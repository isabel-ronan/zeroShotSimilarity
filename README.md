# Towards Zero-Shot Corpus-Level Text Similarity
## Environment Details
- Python Version: 3.12.12
- Pip Version: 25.3
- Creating a conda environment for this project is recommended.
- Required packages can be installed from the `./requirements.txt` file.
- `requirements.txt` made using the `pip list --format=freeze > requirements.txt` command.
- `spacy` is used in this project. Run `python -m spacy download en_core_web_sm` after installing packages from `requirements.txt` to prevent errors.


## Google Colab Runtime Details
Google Colab G4 GPU with [2026.04](https://research.google.com/colaboratory/runtime-version-faq.html#2026.04) runtime. 
- Ubuntu 22.04.5 LTS
- Python 3.12.13
- numpy 2.0.2
- PyTorch 2.10.0
- Jax 0.7.2
- TensorFlow 2.19.0 (not included in TPU runtimes)
- R version 4.5.3 (2026-03-11) -- "Reassured Reassurer"
- julia version 1.11.5