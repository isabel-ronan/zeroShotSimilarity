# Datasets

## Medical Datasets

### SimSUM
- [SimSUM](https://doi.org/10.48550/arXiv.2409.08936) which was downloaded from [GitHub](https://github.com/prabaey/SimSUM). 
- `advanced_text` column is used; these are more 'challenging' compact representations of the textual data designed to replicate real-life note taking settings.
- 10,000 simulated patient records in the domain of respiratory diseases. 

<!-- ### Augmented Clinical Notes 
- [Augmented Clinical Notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes/blob/main/report.pdf) which was downloaded from [HuggingFace](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) with the relevant methodology found at the [Medinote GitHub repository](https://github.com/EPFL-IC-Make-Team/medinote).
- 30,000 structured clinical notes. 
- `full_note` column is used; these represent full notes that are not truncated.  -->

### Clinical Dialogue Summarizations
- [Clinical Dialogue Summarizations](https://doi.org/10.18653/v1/2023.eacl-main.168), which were downloaded from [GitHub](https://github.com/abachaa/MTS-Dialog/tree/main); specifically, we are using the [full augmented dataset](https://github.com/abachaa/MTS-Dialog/blob/main/Augmented-Data/MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv). 
- 3,600 pairs of medical conversations and associated summaries.
- `section_text` column is used; these represent summarized dialogues. 

### DementiaAudio
- [Pitt Corpus](https://talkbank.org/dementia/access/0docs/Becker1994.pdf) which was downloaded from DementiaBank(https://talkbank.org/dementia/access/English/Pitt.html).
- Contains audio recordings and transcripts of various Dementia-related assessments.
- We focus on the transcripts of the Cookie Theft stimulus photo assessment from both the Control and Dementia groups as they are the most consistent and descriptive of all assessments.

### Medical Abstracts
- [Medical Abstracts](https://doi.org/10.1145/3582768.3582795) which was downloaded from [GitHub](https://github.com/sebischair/Medical-Abstracts-TC-Corpus/tree/main) 

### Synthetic Care Home Nurse Notes
- [Synthetically generated nursing home notes](https://doi.org/10.1016/j.jbi.2025.104936) which were downloaded from [GitHub](https://github.com/isabel-ronan/SyntheticNotes); specifically, the [GPT3-generated notes](https://github.com/isabel-ronan/SyntheticNotes/tree/main/data/gpt3) were used.


## Other Datasets

<!-- ### 20 NewsGroups
- [20 NewsGroups](https://doi.org/10.1016/B978-1-55860-377-6.50048-7) dataset which was downloaded from [Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html).
- Contains approximately 18,000 news articles.

### Trump Tweets
- [Trump Twitter Archive V2](https://www.thetrumparchive.com/) dataset downloaded from GoogleDrive as a [CSV](https://drive.google.com/file/d/1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6/view?usp=sharing) file and a [JSON](https://drive.google.com/file/d/16wm-2NTKohhcA26w-kaWfhLIGwl_oX95/view?usp=sharing) file.

### BBC News
- [BBC News](https://doi.org/10.1145/1143844.1143892) dataset which was downloaded from [Kaggle](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive), but is based on the [original source website](http://mlg.ucd.ie/datasets/bbc.html).
- Consists of 2225 documents with 5 topical areas from 2004-2005. 
- Class Labels: 5 (business, entertainment, politics, sport, tech). -->

## HuffPost News
- [HuffPost](https://doi.org/10.48550/arXiv.2209.11429) dataset which was downloaded from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset).
- Used in the paper, ["Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora"](https://doi.org/10.18653/v1/2022.gem-1.35).

## Banking77
- [Banking77](https://doi.org/10.18653/v1/2020.nlp4convai-1.5) dataset which was downloaded from [HuggingFace](https://huggingface.co/datasets/PolyAI/banking77).
- Used in the paper, ["Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora"](https://doi.org/10.18653/v1/2022.gem-1.35).

## Atis
- [Atis](https://aclanthology.org/H90-1021/) dataset which was downloaded from [Kaggle](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem?select=atis_intents.csv).
- Used in the paper, ["Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora"](https://doi.org/10.18653/v1/2022.gem-1.35).

## Yahoo 
- Yahoo dataset which was downloaded from [this website](https://ciir.cs.umass.edu/downloads/nfL6/).
- Used in the paper, ["Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora"](https://doi.org/10.18653/v1/2022.gem-1.35).

## Clinc150
- [Clinc150](https://doi.org/10.18653/v1/D19-1131) dataset which was downloaded from [this website](https://doi.org/10.24432/C5MP58).
- Used in the paper, ["Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora"](https://doi.org/10.18653/v1/2022.gem-1.35).
z