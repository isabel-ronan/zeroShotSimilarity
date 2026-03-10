# PEFT (Parameter-Efficient Fine-Tuning)
## Why PEFT?
- "The traditional paradigm is to finetune all of a model’s parameters for each downstream task, but this is becoming exceedingly costly and impractical because of the enormous number of parameters in models today." - [HuggingFace](https://huggingface.co/docs/peft/quicktour)
- Efficient adaptation of LLMs for various downstream applications without fine-tuning all of the model's parameters.
- Significant decrease in computational and storage costs compared to fully fine-tuned models. 
- More accessible to train and store large language models on consumer hardware. 

## Model - `distilbert-base-uncased` or `sentence-transformers/all-MiniLM-L6-v2` or `google/embeddinggemma-300m` or `google/medgemma-1.5-4b-it` or `emilyalsentzer/Bio_ClinicalBERT`
- `distilbert-base-uncased` - slightly more popular, but larger and slower to train.
- `sentence-transformers/all-MiniLM-L6-v2` - slightly less common for fine-tuned sentence classification tasks, but smaller and faster to train.
- `distilbert-base-uncased` performance is slightly better but takes double the training time (60mins) compared to `sentence-transformers/all-MiniLM-L6-v2` taking 30 mins. `google/embeddinggemma-300m` performance is better than `distilbert-base-uncased` but is much slower (101 minutes and 15.0seconds) and does not come with token-level explainability. `emilyalsentzer/Bio_ClinicalBERT` performs best but took longest to train (289 minutes and 42.6 seconds).

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