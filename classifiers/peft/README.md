# PEFT (Parameter-Efficient Fine-Tuning)
## Why PEFT?
- "The traditional paradigm is to finetune all of a model’s parameters for each downstream task, but this is becoming exceedingly costly and impractical because of the enormous number of parameters in models today." - [HuggingFace](https://huggingface.co/docs/peft/quicktour)
- Efficient adaptation of LLMs for various downstream applications without fine-tuning all of the model's parameters.
- Significant decrease in computational and storage costs compared to fully fine-tuned models. 
- More accessible to train and store large language models on consumer hardware. 

## Model - `distilbert-base-uncased` or `microsoft/MiniLM-L12-H384-uncased` 
- `distilbert-base-uncased` - slightly more popular, but larger and slower to train.
- `microsoft/MiniLM-L12-H384-uncased` - slightly less common for fine-tuned sentence classification tasks, but smaller and faster to train.