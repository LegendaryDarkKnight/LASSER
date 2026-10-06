# LASSER: Learning Adaptive Sparse Structures for gene Effect Reasoning

We saw that GEARS uses static co-expression and GO graph. So our thoughts were:
1. what if we learnt the co-expression graph instead of building just once and use it?
2. what if we learnt the GO graphs instead of building just once and use it?
3. can you make some reasoning of relationships among the genes from learnt graphs?

So to find this questions we made a progressive plan.
We first tried to incorporate IDGL framework to learn cosine similarity from embedding and use that to build graph.
But we faced several problems.
1. Sparsification was really bad and the graph became so huge to fit in memory
2. We need to convert adjacency matrix to edgelist for SGConv and it took way more time to build the things.
3. We couldn't actually find any minima
4. in the long run there were way too many node embeddings that had higher similarities with each other(maybe the embeddings become flattened)

We added poincare embedding learning for GO graph also.
For now avoid that.

I have already provided you the papers and one of our sample implementation.

Now you try to make a plan on how to work to gain better results and accomplish our purpose. Avoid GO graph learning. We will only focus on coexpression graph learning. Keeping all memory restrictions in mind try to make design choices