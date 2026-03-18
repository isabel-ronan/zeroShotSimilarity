# PEFT (Parameter-Efficient Fine-Tuning)
## Why PEFT?
- "The traditional paradigm is to finetune all of a model’s parameters for each downstream task, but this is becoming exceedingly costly and impractical because of the enormous number of parameters in models today." - [HuggingFace](https://huggingface.co/docs/peft/quicktour)
- Efficient adaptation of LLMs for various downstream applications without fine-tuning all of the model's parameters.
- Significant decrease in computational and storage costs compared to fully fine-tuned models. 
- More accessible to train and store large language models on consumer hardware. 

## Model - `distilbert-base-uncased` or `sentence-transformers/all-MiniLM-L6-v2` or `google/embeddinggemma-300m` or `google/medgemma-1.5-4b-it` or `emilyalsentzer/Bio_ClinicalBERT` or `nlpie/distil-clinicalbert` or `nlpie/tiny-clinicalbert`
- `distilbert-base-uncased` - slightly more popular, but larger and slower to train.
- `sentence-transformers/all-MiniLM-L6-v2` - slightly less common for fine-tuned sentence classification tasks, but smaller and faster to train.
- `distilbert-base-uncased` performance is slightly better but takes double the training time (60mins) compared to `sentence-transformers/all-MiniLM-L6-v2` taking 30 mins. `google/embeddinggemma-300m` performance is better than `distilbert-base-uncased` but is much slower (101 minutes and 15.0seconds) and does not come with token-level explainability. `emilyalsentzer/Bio_ClinicalBERT` performs best but took longest to train (289 minutes and 42.6 seconds). `nlpie/distil-clinicalbert` performed well and took a shorter amount of time to train (91 minutes and 5.1 seconds). `nlpie/tiny-clinicalbert` took the shortest amount of time to train (24 minutes and 55 seconds) with acceptable performance.
- Using TF-IDF and Logistic Regression as the baseline. 

### Logistic Regression Results
#### Confusion Matrix

| 514 | 67 |

| 28 | 548 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.95      | 0.88   | 0.92     | 581     |
|        | 1      | 0.89      | 0.95   | 0.92     | 576     |
| ------       | ------ | ------    | ------ | ------   | ------ |
| accuracy     | ------ | ------    | ------ | 0.92     | 1157   |
| macro avg    | ------ | 0.92      | 0.92   | 0.92     | 1157   |
| weighted avg | ------ | 0.92      | 0.92   | 0.92     | 1157   |


### `nlpie/tiny-clinicalbert` Results
#### Confusion Matrix

| 269 | 22 |

| 13 | 275 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.95      | 0.92   | 0.94     | 291     |
|        | 1      | 0.93      | 0.95   | 0.94     | 288     |
| ------       | ------ | ------    | ------ | ------   | ------  |
| accuracy     | ------ | ------    | ------ | 0.94     | 579     |
| macro avg    | ------ | 0.94   | 0.94      | 0.94   | 579|
| weighted avg | ------ | 0.94   | 0.94      | 0.94   | 579|

### `nlpie/distil-clinicalbert` Results
#### Confusion Matrix

| 265 | 26 |

| 7 | 281 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.97      | 0.91   | 0.94     | 291     |
|        | 1      | 0.92      | 0.98   | 0.94     | 288     |
| ------       | ------ | ------    | ------ | ------   | ------  |
| accuracy     | ------ | ------    | ------ | 0.94     | 579     |
| macro avg    | ------ | 0.94   | 0.94      | 0.94   | 579|
| weighted avg | ------ | 0.94   | 0.94      | 0.94   | 579|



### `emilyalsentzer/Bio_ClinicalBERT` Results
#### Confusion Matrix

|274 | 17 |

| 7 | 281 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.98      | 0.94   | 0.96     | 291     |
|        | 1      | 0.94      | 0.98   | 0.96     | 288     |
| ------       | ------ | ------    | ------ | ------   | ------  |
| accuracy     | ------ | ------    | ------ | 0.96     | 579     |
| macro avg    | ------ | 0.96   | 0.96      | 0.96   | 579|
| weighted avg | ------ | 0.96   | 0.96      | 0.96   | 579|

### `google/embeddinggemma-300m` Results
#### Confusion Matrix

|284 | 7 |

| 21 | 267 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.93      | 0.98   | 0.95     | 291     |
|        | 1      | 0.97      | 0.93   | 0.95     | 288     |
| ------       | ------ | ------    | ------ | ------   | ------  |
| accuracy     | ------ | ------    | ------ | 0.95     | 579     |
| macro avg    | ------ | 0.95   | 0.95      | 0.95   | 579|
| weighted avg | ------ | 0.95   | 0.95      | 0.95   | 579|


### `distilbert-base-uncased` Results
#### Confusion Matrix

|264 | 27 |

| 12 | 276 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.96      | 0.91   | 0.93     | 291     |
|        | 1      | 0.91      | 0.96   | 0.93     | 288     |
| ------       | ------ | ------    | ------ | ------   | ------  |
| accuracy     | ------ | ------    | ------ | 0.93     | 579     |
| macro avg    | ------ | 0.93   | 0.93      | 0.93   | 579|
| weighted avg | ------ | 0.93   | 0.93      | 0.93   | 579|


### `sentence-transformers/all-MiniLM-L6-v2` Results
#### Confusion Matrix

|263 | 28 |

| 13 | 275 |

 #### Classification Report
| metric | label  | precision | recall | f1-score | support |
| ------ | ------ | ------    | ------ | ------   | ------  |
|        | 0      | 0.95      | 0.90   | 0.93     | 291     |
|        | 1      | 0.91      | 0.95   | 0.93     | 288     |
| ------       | ------ | ------    | ------ | ------   | ------  |
| accuracy     | ------ | ------    | ------ | 0.93     | 579     |
| macro avg    | ------ | 0.93   | 0.93      | 0.93   | 579|
| weighted avg | ------ | 0.93   | 0.93      | 0.93   | 579|

## Explainability
### [SHAP](https://doi.org/10.48550/arXiv.1705.07874)
- Using the [SHAP package](https://github.com/shap/shap?tab=readme-ov-file); this is constantly GitHub updated (which cannot be said for LIME). 
- SHAP provides high accuracy and reliable text explanations. It's the gold standard for 'fair' credit assignment. 
- SHAP can be computationally expensive for transformers and slow if the input sequence is long.
- Use 'partition' algorithm as it uses a hierarchy to group tokens and systematically masks them to see the change in output (this method is more efficient for text). It is model-agnostic and works very well with Hugging Face models. 