# Import packages.
import os
import json
import spacy
import warnings
import pandas as pd

# Ignore UserWarnings.
warnings.filterwarnings("ignore", category=UserWarning)


# Initialize constant variables.
INPUT_FOLDER = 'data'
OUTPUT_FOLDER = 'dataProcessed'

# Make Spacy NLP object.
nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]  # speed
)

# Make function to remove punctuation, make lowercase, remove stopwords, punctuation, lemmatize, remove documents with less than 5 tokens.
def preprocessing(texts, min_words = 5):
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=1000):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.lemma_ != "-PRON-"
            and token.is_alpha
        ]
        if len(tokens) >= min_words:
            cleaned_texts.append(" ".join(tokens))
    return cleaned_texts

# Process all T1 files (this can be adjusted later for T2 also).
nurse_notes = {}
nurse_notes_processed = {}
for folder in os.listdir(f'./{INPUT_FOLDER}/'):
    if '.' not in folder:
        for sub_folder in os.listdir(f'./{INPUT_FOLDER}/{folder}'):
            if '.' not in sub_folder and 'T1' in sub_folder:
                for sub_sub_folder in os.listdir(f'./{INPUT_FOLDER}/{folder}/{sub_folder}'):
                    if '.' not in sub_sub_folder: 
                        for file in os.listdir(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}'):
                            if file[0] != '.' and 'xlsx' in file:
                                if 'dailyNurseNotes' in file:
                                    nurse_notes_processed[sub_sub_folder.split(' ')[0]] = preprocessing(list(pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/{file}')['Note'].dropna()))
                                    nurse_notes[sub_sub_folder.split(' ')[0]] = list(pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/{file}')['Note'].dropna())


# If saving folder does not exist, make it.
if not os.path.exists(f'./{OUTPUT_FOLDER}/'):
    os.makedirs(f'./{OUTPUT_FOLDER}/', exist_ok=False)  

# Save JSON output.
json_str = json.dumps(nurse_notes, indent=4)
with open(f'./{OUTPUT_FOLDER}/nurseNotes.json', "w") as f:
    f.write(json_str)

# Save JSON output.
json_str = json.dumps(nurse_notes_processed, indent=4)
with open(f'./{OUTPUT_FOLDER}/nurseNotesProcessed.json', "w") as f:
    f.write(json_str)