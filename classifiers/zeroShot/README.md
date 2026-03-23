# Zero-Shot Classification

## Zero-Shot Hypothesis Templates
- "This shows that a pre-trained language model can inherit zero-shot capabilities when given appropriate prompts, even without using any human-annotated examples." - [Lu et al.](https://doi.org/10.18653/v1/2023.acl-long.128)
- "To address this weakness, we propose meta-tuning, which directly optimizes the zero-shot learning objective by fine-tuning pre-trained language models on a collection of datasets ... we can potentially first use meta-tuning as a probe to make them adapted to answering prompts before measuring their performance." - [Zhong et al.](https://doi.org/10.18653/v1/2021.findings-emnlp.244)
- "combine hypotheses to create more accurate NLI-based zero-shot hate speech detection systems. Specifically, we develop four simple strategies, filtering by target, filtering counter speech, filtering reclaimed-slurs, and catching de-humanizing comparisons, that target specific model weaknesses" - [Goldzycher et al.](https://aclanthology.org/2022.trac-1.10/)
- "However, the precise choice of the prompt template and label words can largely influence performance, with semantically equivalent settings often showing notable performance difference. ... inherent class bias is a significant factor that influences the sensitivity of the system to prompt and label words" - [Liusie et al.](https://doi.org/10.18653/v1/2023.findings-ijcnlp.29)
- "prompting is known to be sensitive to the choice of the pattern and the verbalizer, yet practitioners are blind when designing them in true zero-shot setting" - [van de Kar et al.](https://doi.org/10.48550/arXiv.2210.14803)
- Has examples of various styles of hypothesis template in [Yudanto et al.](https://aclanthology.org/2024.paclic-1.57/)

## Chosen Models
- Taken from the most downloaded zero-shot classification models on the [HuggingFace models page](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads).
- Models considered if >= 1k downloads.
- Filtered based on [memory calculations](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator).