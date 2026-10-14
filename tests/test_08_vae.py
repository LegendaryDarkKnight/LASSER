"""VAE graph learning and the cell-signal variant, on synthetic data (no dataset, CPU, seconds)."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytest.importorskip("torch_geometric")

from lasser.config import LasserConfig, VAEConfig, stable_hash
from lasser.vae import edge_confidence, gene_embeddings, train_vae


def _cos(E):
    En = torch.nn.functional.normalize(E.float(), dim=1)
    return (En @ En.T).numpy()


# ------------------------------------------------------------- similarities
def test_denoise_and_resp_embeddings_are_exact_correlations():
    torch.manual_seed(0)
    W = torch.randn(12, 4)
    Z = torch.randn(300, 4)
    conds = np.array(["ctrl"] * 100 + [f"p{i}" for i in range(4) for _ in range(50)])
    Z[100:] += torch.repeat_interleave(torch.randn(4, 4), 50, dim=0)          # perturbations shift the latent
    embs = gene_embeddings(W, Z, conds, torch.zeros(12, dtype=torch.bool), ("load", "denoise", "resp"))
    recon = (Z @ W.T).numpy()
    assert np.allclose(_cos(embs["denoise"]), np.corrcoef(recon.T), atol=1e-4)    # denoised Pearson
    means = np.stack([recon[conds == c].mean(0) for c in ["p0", "p1", "p2", "p3"]])
    delta = means - recon[conds == "ctrl"].mean(0)                                   # decoded change per pert
    assert np.allclose(_cos(embs["resp"]), np.corrcoef(delta.T), atol=1e-4)         # response correlation
    assert np.allclose(_cos(embs["load"]), _cos(W), atol=1e-6)


def test_missing_genes_get_zero_embeddings():
    W, Z = torch.randn(5, 3), torch.randn(50, 3)
    conds = np.array(["ctrl"] * 20 + ["a"] * 15 + ["b"] * 15)
    miss = torch.tensor([False, True, False, False, True])
    embs = gene_embeddings(W, Z, conds, miss, ("load", "denoise", "resp"))
    for e in embs.values():
        assert torch.all(e[miss] == 0) and torch.all(e[~miss].abs().sum(1) > 0)


# -------------------------------------------------------------- VAE training
def _modules(n_cells=1500, seed=0):
    """30 genes in 3 modules of 10; each module driven by one latent factor; two 'perturbations'."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n_cells, 3))
    conds = np.array(["ctrl"] * (n_cells // 3) + ["pA"] * (n_cells // 3) + ["pB"] * (n_cells - 2 * (n_cells // 3)))
    f[conds == "pA", 0] += 2.0
    f[conds == "pB", 2] += 2.0
    load = np.zeros((30, 3))
    for m in range(3):
        load[10 * m:10 * (m + 1), m] = rng.uniform(0.8, 1.2, 10)
    X = np.clip(f @ load.T + 3.0 + rng.normal(scale=0.5, size=(n_cells, 30)), 0, None).astype(np.float32)
    return sp.csr_matrix(X), conds


def test_vae_recovers_modules():
    X, conds = _modules()
    cfg = VAEConfig(latent_dim=4, hidden_dim=32, epochs=40, batch_size=128, kl_warmup_epochs=5, patience=10)
    model, hist, best = train_vae(X, cfg, seed=0, device=torch.device("cpu"))
    assert hist[-1]["val_recon"] < hist[0]["val_recon"] and 1 <= best <= len(hist)
    with torch.no_grad():
        mu, _ = model.encode(torch.from_numpy(X.toarray()))
    embs = gene_embeddings(model.W.detach(), mu, conds, torch.zeros(30, dtype=torch.bool), ("denoise",))
    S = _cos(embs["denoise"])
    block = np.repeat(np.arange(3), 10)
    same = (block[:, None] == block[None, :]) & ~np.eye(30, dtype=bool)
    diff = block[:, None] != block[None, :]
    print(f"denoised similarity within modules {S[same].mean():.3f}, between {S[diff].mean():.3f}")
    assert S[same].mean() > S[diff].mean() + 0.3


def test_edge_confidence_range():
    torch.manual_seed(0)
    W, mu, sd = torch.randn(40, 4), torch.randn(200, 4), torch.full((200, 4), 0.1)
    conds = np.array(["ctrl"] * 50 + ["a"] * 50 + ["b"] * 50 + ["c"] * 50)
    cfg = VAEConfig(n_posterior_samples=4, confidence_k=5, chunk_size=16)
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    c = edge_confidence(W, mu, sd, conds, torch.zeros(40, dtype=torch.bool), src, dst, cfg, 0, torch.device("cpu"))
    assert c.shape == (3,) and torch.all((c >= 0) & (c <= 1))


# ------------------------------------------------------------ cell signal
def _tiny_gears(n_genes=6, n_perts=3, hidden=8):
    from lasser.gears.model import GEARS_Model

    ei = torch.tensor([[1, 2, 3, 0], [0, 0, 1, 2]])
    args = dict(num_genes=n_genes, num_perts=n_perts, hidden_size=hidden, uncertainty=False,
                num_go_gnn_layers=1, decoder_hidden_size=4, num_gene_gnn_layers=1, no_perturb=False,
                G_coexpress=ei, G_coexpress_weight=torch.ones(4), G_go=torch.tensor([[0, 1], [1, 0]]),
                G_go_weight=torch.ones(2), device="cpu")
    torch.manual_seed(0)
    return GEARS_Model(args).eval()


def _batch(xs, n_genes=6):
    from torch_geometric.data import Batch, Data

    return Batch.from_data_list([Data(x=torch.tensor(x, dtype=torch.float32).view(n_genes, 1), pert_idx=[0],
                                      pert="A+ctrl") for x in xs])


def test_upstream_delta_is_cell_independent_but_cell_signal_is_not():
    """The paper's mechanism: in GEARS the predicted change (output minus control) does not depend
    on the cell's expression. Upstream, it differs only by batch position (cell 0 is the one that
    gets co-expression messages); with the batching fix it is identical for every cell. With the
    cell-signal variant it depends on the cell's expression."""
    import copy

    from lasser.batching import enable_batched_coexpress
    from lasser.cellsignal import CellSignalGEARSModel, enable_cell_signal

    rng = np.random.default_rng(0)
    xs = [rng.uniform(0, 3, 6) for _ in range(3)]
    b = _batch(xs)
    up = _tiny_gears()
    with torch.no_grad():
        delta_up = up(b) - b.x.view(3, 6)
    assert torch.allclose(delta_up[1], delta_up[2], atol=1e-6)               # cells 1.. : self-loop only
    fixed = copy.deepcopy(up)
    enable_batched_coexpress(fixed, 6)
    with torch.no_grad():
        delta_fix = fixed(b) - b.x.view(3, 6)
    assert torch.allclose(delta_fix[0], delta_fix[1], atol=1e-6) and torch.allclose(delta_fix[1], delta_fix[2], atol=1e-6)
    assert torch.allclose(delta_fix[0], delta_up[0], atol=1e-6)              # the fix gives every cell cell 0's view

    cs = copy.deepcopy(up)
    n_params = sum(p.numel() for p in cs.parameters())
    assert enable_cell_signal(cs, 6, seed=1) == 1 and isinstance(cs, CellSignalGEARSModel)
    assert sum(p.numel() for p in cs.parameters()) == n_params + 2 * 8      # u and c per conv
    with torch.no_grad():
        cs.layers_emb_pos[0].cell_u.fill_(1.0)
        delta_cs = cs(b) - b.x.view(3, 6)
    assert not torch.allclose(delta_cs[0], delta_cs[1], atol=1e-4)
    assert cs.layers_emb_pos[0]._lasser_cell_x is None                      # nothing kept between calls


def test_cell_signal_zero_init_matches_batched_fix():
    import copy

    from lasser.batching import enable_batched_coexpress
    from lasser.cellsignal import enable_cell_signal

    rng = np.random.default_rng(1)
    b = _batch([rng.uniform(0, 3, 6) for _ in range(2)])
    fixed, cs = _tiny_gears(), _tiny_gears()
    enable_batched_coexpress(fixed, 6)
    enable_cell_signal(cs, 6, seed=1)
    with torch.no_grad():
        cs.layers_emb_pos[0].cell_u.zero_()
        assert torch.allclose(fixed(b), cs(b), atol=1e-6)
    copy.deepcopy(cs)                                                        # GEARS deep-copies models


# ------------------------------------------------------------ config/hash
def test_config_compatibility():
    base = LasserConfig()
    d = {k: v for k, v in base.to_dict().items()
         if k not in {"data_dir", "out_dir", "cache_dir", "log_level", "repo_dir", "device", "run_tag", "gsr", "vae",
                      "fix_coexpress_batching", "train_seed", "graph_mode", "cell_signal"}}
    assert base.config_hash() == stable_hash(d)                              # baseline hashes unchanged
    gl = LasserConfig(graph_learning=True)
    d_gl = dict(d, graph_learning=True, gsr=gl.to_dict()["gsr"])
    assert gl.config_hash() == stable_hash(d_gl)                            # GSR-run hashes unchanged
    assert gl.graph_cache_key() == gl.gsr_cache_key()
    vae = LasserConfig(graph_learning=True, graph_mode="vae")
    assert vae.config_hash() != gl.config_hash() and vae.graph_cache_key() == vae.vae_cache_key()
    assert LasserConfig.from_dict(vae.to_dict()).config_hash() == vae.config_hash()
    with pytest.raises(ValueError):
        LasserConfig(cell_signal=True).validate()                           # needs the batching fix
    with pytest.raises(ValueError):
        LasserConfig(graph_mode="nope").validate()
    with pytest.raises(ValueError):
        LasserConfig(graph_learning=True, graph_mode="vae", vae=VAEConfig(beta={"load": 1.0})).validate()


def test_vae_plan_jobs():
    from lasser.study import VAE_PLAN, make_jobs

    jobs = make_jobs(VAE_PLAN, dict(data_dir="/d", out_dir="/o", cache_dir="/c", epochs=10))
    ids = [j["job_id"] for j in jobs]
    assert ids[:2] == ["GEARS-cs_seed1", "VAE-cs_seed1"]
    assert len(ids) == len(set(ids)) == len({j["config_hash"] for j in jobs})
    j = {x["job_id"]: x for x in jobs}
    assert j["VAE-random-cs_seed1"]["config"]["vae"]["edge_control"] == "random"
    assert j["VAE-cs_seed2"]["config"]["graph_mode"] == "vae" and j["VAE-cs_seed2"]["config"]["cell_signal"]
    assert j["LASSER-cs_seed1"]["config"]["graph_mode"] == "gsr"
    assert not j["VAE-fix_seed1"]["config"]["cell_signal"]
    a = LasserConfig.from_dict(j["VAE-cs_seed1"]["config"])
    b = LasserConfig.from_dict(j["VAE-cs_seed1_ts101"]["config"])
    assert a.vae_cache_key() == b.vae_cache_key()                           # replicates share the VAE graph
