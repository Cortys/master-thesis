# Learning to Aggregate on Structured Data

## ToC:

1. Introduction
   1. Motivation
   2. Goals
   3. Structure
2. Related Work
   1. Learning to Aggregate
   2. GC/GR
      - Static:
        - Fingerprint 
        - WL-kernel
        - random walk kernel
        - shortest path kernel
      - Unsupervised:
        - graph2vec (maybe also sub2vec, node2vec)
      - Supervised and Semi-Supervised:
        - Graph Neural Networks
          - Spatial GNNs
          - Spectral GNNs
3. Theoretical Foundations
   1. Graph Similarity
      1. WL
      2. Laplacians
   1. Static
   2. 
   3. GNN
      1. Spatial
      2. Spectral
4. LTA on Graphs
   1. A General LTA Formalization
      1. Steps:
         1. Decompose (Disaggregate)
         2. Evaluate  (Disaggregate)
         3. Aggregate
      2. Properties:
         1. Structural independence
         2. Constituent connectivity
         3. Dynamic decomposition (currently not)
         4. Dynamic evaluation (usually always)
         5. Dynamic aggregation (sometimes)
   2. An LTA Interpretation of Graph Methods 
   3. LTA on Dynamically Decomposed Graphs
5. Evaluation
6. Conclusion
   1. Review
   2. Future Directions

## GC/GN work selection:

embedding
  laplacian factorization
  proximity matrix factorization
kernel
  wl
  shortest path
  graphlet
deep learning
  neural network (take from proposal)
    spatial
    spectral
