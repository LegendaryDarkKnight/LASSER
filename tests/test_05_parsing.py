"""GEARS printout parsing, on lines copied from Kaggle run 02. Needs no data or GPU."""

import math

from lasser.tracking import parse_gears_output

LINES = """Start Training...
Epoch 1 Step 1 Train Loss: 0.4788
Epoch 1 Step 51 Train Loss: 0.3012
Epoch 1: Train Overall MSE: 0.0460 Validation Overall MSE: 0.0381.
Train Top 20 DE MSE: 0.6215 Validation Top 20 DE MSE: 0.7050.
Epoch 2 Step 1 Train Loss: 0.2500
Epoch 2: Train Overall MSE: 0.0100 Validation Overall MSE: 0.0099.
Train Top 20 DE MSE: 0.2000 Validation Top 20 DE MSE: 0.7100.
Done!
Start Testing...
Best performing model: Test Top 20 DE MSE: 0.2039
Start doing subgroup analysis for simulation split...
test_combo_seen0_mse: 0.005785305466916826
test_combo_seen1_pearson_delta: -0.004211393
test_unseen_single_frac_opposite_direction_top20_non_dropout: 0.18333333333333332
test_x_mse: 1e-05
test_y_mse: nan
Done!""".split("\n")


def test_parse_gears_output():
    out = parse_gears_output(LINES)
    ep = out["epochs"]
    assert list(ep.epoch) == [1, 2]
    assert ep.val_mse.tolist() == [0.0381, 0.0099]
    assert ep.train_mse.tolist() == [0.0460, 0.0100]
    assert ep.val_mse_de.tolist() == [0.7050, 0.7100]
    assert ep.train_loss_logged_mean.tolist() == [(0.4788 + 0.3012) / 2, 0.25]
    assert ep.is_best.tolist() == [True, False]
    t = out["gears_test"]
    assert t["test_mse_de"] == 0.2039
    assert t["test_combo_seen0_mse"] == 0.005785305466916826
    assert t["test_combo_seen1_pearson_delta"] == -0.004211393
    assert t["test_unseen_single_frac_opposite_direction_top20_non_dropout"] == 0.18333333333333332
    assert t["test_x_mse"] == 1e-05 and math.isnan(t["test_y_mse"])
