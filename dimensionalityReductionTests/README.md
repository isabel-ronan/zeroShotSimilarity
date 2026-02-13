# Dimensionality Reduction

## Algorithms Used
| Algorithm | Description | Advantages | Disadvantages | Suitable Use-Cases | 
| --------- | --------- | --------- | --------- | --------- |
| PCA | Projects data onto orthogonal components that explain maximum variance. | Fast, interpretable, unsupervised, works well for linear data | Assumes linearity, sensitive to scaling | Visualization, noise reduction, feature decorrelation | 
| t-SNE | Models pairwise similarities to preserve local structure in 2D/3D space. | Great for visualization, captures local structure | Poor global structure, not scalable, non-deterministic | Visualizing high-dimensional clusters | 
| UMAP | Preserves local and some global structure using manifold approximation. | Faster & better global structure than t-SNE, scalable | Still stochastic, may need tuning | Visualization, preprocessing for clustering | 
| Random Projection (RP) | Projects data using a random matrix while preserving distance properties. | Extremely fast, low memory usage | Results are approximate, not interpretable | Large-scale data reduction, fast preprocessing | 
| Independent Component Analysis (ICA) | Finds independent non-Gaussian components from mixed signals. | Useful for signal separation, handles non-Gaussian data | Not good for Gaussian noise, can be unstable | Audio separation, EEG/MEG analysis | 
| Factor Analysis (FA) | Models observed variables as linear combinations of latent factors. | Handles noise well, interpretable factor loadings | Assumes linearity, sensitive to overfitting | Latent structure discovery, psychometrics | 

## Why Use So Many Dimensionality Reduction Techniques?
- To see which technique leads to better results.
- Autoencoder not used because it is very computationally intensive.

|Metric | Measures | Type | Scale | Advantages | Disadvantages |
| ------- | ------- | ------- | ------- | ------- | ------- |
|Trustworthiness | Local Neighbor Fidelity | Local | 0 to 1 (higher == better) | Local Structure Sensitivity, No Labels Needed, Standardized and Well Accepted for DR | Neighbour Choice Dependent |
|KNN Preservation |	Neighbor Overlap | Local | 0 to 1 (higher == better) | Intuitive, Easy to Compute | Neighbour Choice Dependent, No Relative Ranking of Neighbours |
|Continuity | Global Neighbor Fidelity | Global | 0 to 1 (higher == better) | Captures Global Fidelity, No Labels Needed | Neighbour Choice Dependent, Computationally Expensive |