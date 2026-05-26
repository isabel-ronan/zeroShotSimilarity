# Zero-Shot Learning vs MDA for Corpus Descriptions

- Dimension-based method satisfies one of the areas of future work proposed by [Kour et al. in Measuring the Measuring Tools](https://doi.org/10.18653/v1/2022.gem-1.35). There are separate scores for each of Biber's 6 dimensions. 

# pybiber
- Open-source implementation of Biber's 67 lexicogrammatical and functional features, as described in ["Variation across Speech and Writing"](https://doi.org/10.1017/CBO9780511621024).
- These 67 features are widely used for text-type, register, and genre classification tasks in corpus linguistics.
- Advantages include local running and quick computation (in comparison to more intensive machine learning methods).
- Theoretically-grounded library which is used for comparison in other Biber-style papers (as the fastest library which does not used advanced neural methods) (such as [Neurobiber](https://doi.org/10.48550/arXiv.2502.18590)). 
- ["One powerful feature of the BiberAnalyzer is the ability to project your data onto Biber’s original dimensions, allowing for direct comparison with established research."](https://browndw.github.io/pybiber/biber-analyzer.html#comparison-with-bibers-original-dimensions). [Exact quote can be found here.](https://browndw.github.io/pybiber/biber-analyzer.html#comparison-with-bibers-original-dimensions:~:text=One%20powerful%20feature%20of%20the%20BiberAnalyzer%20is%20the%20ability%20to%20project%20your%20data%20onto%20Biber%E2%80%99s%20original%20dimensions%2C%20allowing%20for%20direct%20comparison%20with%20established%20research)


# Zero-Shot Classification
- Use a mixture of small zero-shot classification models running locally (to facilitate the use of this method in small-resource settings or with sensitive data).
- [Ensembled small models can perform competitively with larger models](https://doi.org/10.18653/v1/2023.arabicnlp-1.51).
- [Ensembles can perform better than any single model on its own](https://doi.org/10.18653/v1/2024.wassa-1.49).

## Model Selection
<!-- - Models were selected based on their appearance in the [BTZSC benchmark](https://doi.org/10.48550/arXiv.2603.11991) (on 28th March, 2026). 
- Criteria included that the model was within the top 10 highest performers (with the exclusion of a custom-trained deberta model (`deberta-v3-large-nli-triplet`), which we replaced with `cross-encoder/nli-deberta-v3-large` as a substitute), had less than 1B parameters, and were pre-trained for natural language inference or zero-shot classification as assessed on the [Hugging Face BTZSC Leaderboard](https://huggingface.co/spaces/btzsc/btzsc-leaderboard).  -->
- Taken from the most downloaded zero-shot classification models on the [HuggingFace models page](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads).
- Models considered if >= 100k downloads and parameters less than 0.1B.
<!-- - Filtered based on [memory calculations](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator). -->
- Focused on diverse range of architectures with civersity, capacity, and efficiency. 

# Analysis Scores

## Classification Metrics
| Metric | Purpose | Pros | Cons | Scale | Interpretation |
| ------- | ------- | ------- | ------- | ------- | ------- |
| **Accuracy** | % of correct predictions | Simple; easy to compute | Misleading for imbalanced classes | 0–1 | high = better; low = worse |
| **Precision** | Correct positives/predicted positives | Measures false positives; good for selective prediction | Ignores false negatives | 0–1 | high = better; low = worse |
| **Recall (Sensitivity)** | Correct positives/actual positives | Measures false negatives; good for catching positives | Ignores false positives | 0–1 | high = better; low = worse |
| **F1 Score** | Harmonic mean of precision & recall | Balances precision & recall; best for imbalanced data | Harder to interpret than individual precision/recall | 0–1 | high = better; low = worse |
| **Cohen’s Kappa** | Chance-corrected agreement (inter-rater reliability). | Adjusts for chance; good with uneven label distribution | Less intuitive than accuracy | -1 to 1 | high = better; low = worse ; 0 = exactly what is expected by chance |
| **Matthews Correlation Coefficient (MCC)** | Balanced correlation measure for binary classification. | Handles imbalanced data well (considers both true/false positives and negatives), making it better than F1 score for imbalanced datasets. | Complex interpretation (but one of the best ways to summarize a confusion matrix into a single value) | -1 to 1 | high = better; low = worse; 0 = no better than random guessing |

## Continuous Scores
| Metric | Purpose | Pros | Cons | Scale | What High/Low Means |
| ------- | ------- | ------- | ------- | ------- | ------- |
| **Pearson Correlation** | Measures linear association between model and Biber scores | Captures linear trends; easy to interpret | Only linear; sensitive to outliers | -1 to 1 | 1 == strong positive correlation; 0 == negative or weak correlation |
| **Spearman Correlation** | Measures rank-order association | Captures monotonic trends; robust to outliers | Ignores exact differences; less sensitive to linearity | -1 to 1 | 1 == strong monotonic relationship; 0 == weak or inverse relationship |
| **Mean Squared Error (MSE)** | Measures average squared deviation | Penalizes large errors strongly | Harder to interpret; sensitive to outliers | 0 to infinity  | 0 == better; High == worse |
| **Root Mean Squared Error (RMSE)** | root MSE, interpretable in same scale as factor (units are units of original data (e.g. dollars of error if using dollars))| Same units as target; easier to interpret | Still sensitive to outliers | 0 to infinity  | 0 == better; High == worse |
| **Mean Absolute Error (MAE)** | Average absolute deviation | Intuitive; less sensitive to outliers than MSE | Does not penalize large errors as strongly as MSE | 0 to infinity  | 0 == better; high == worse |

# Hypotheses
- Biber dimensions can be derived from a local small zero-shot model ensemble in the same way as they can be derived from rule-based features. **Zero-shot and biber-based dimensions are not comparable and result in very low interrater agreeability**. - 10x10 (dataset x 10 documents) x 100 with classification and continuous measures measuring the difference between outputs. 
- A local small zero-shot model ensemble with corpus-linguistic-informed labels can distinguish corpora of different types better than traditional corpus linguistic measures (biber rule-based features). **Yes, the ensemble distinguishes corpora of different types better than traditional measures.** - train a series of machine learning models on each set of dimensions (also plot averages of 10x10 using UMAP) to show that zero-shot dimensions results in better separation than traditional dimensions (models performed nearly 10 percentage points better using zero-shot in comparison to traditional dimensions across the range of classification scores (classification into dataset categories)). 

# Methodology
1. Divide datasets into train, test, val. 
2. Sample 10 notes from set. For each note from each dataset, classify into biber's 6 dimensions using zero-shot model (zero-shot dimensions) and biber features (traditional biber dimensions). Make sure each set of 10x10 is z score normalized to ensure accurate comparison between zero-shot and traditional biber dimensions (essentially storing how informed or informational each note is relative to all other notes in the 10x10 subset). Compute continuous and classification metrics between zero-shot and traditional biber dimensions (from analysis scores metrics outlined above). Save to folder (named after random seed). 
3. Repeat step 2 100 times for the train, test, and val sets separately (100 x 100 x 100). 
4. Visualise using UMAP (both zero-shot and traditional biber dimensions separately for averaged and non-averaged points). for train, test, val -> (get the mean dimensions for any duplicates for non-averaged implementation), reduce 6 dimensions to 2 dimensions using UMAP and plot on a graph with the colour of each point being the dataset. This helps visualize dataset separation. 
4. Set up machine learning pipeline with classical models. Train model on train set (averaged dimensions 10x10 becomes 1x10 for each dataset). Evaluate on test dataset. Store results. Do this for both zero-shot and traditional biber dimensions separately. 