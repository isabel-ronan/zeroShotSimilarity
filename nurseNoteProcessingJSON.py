# Import packages.
import os
import json
import spacy
import warnings
import pandas as pd

def main():
    # Ignore UserWarnings.
    warnings.filterwarnings("ignore", category=UserWarning)


    # Initialize constant variables.
    INPUT_FOLDER = 'data'
    OUTPUT_FOLDER = 'dataProcessedJSON'

    # Make Spacy NLP object.
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["ner", "parser"]  # speed
    )

    # Make function to remove punctuation, make lowercase, remove stopwords, punctuation, lemmatize, remove documents with less than 5 tokens.
    def preprocessing(notes, dates, times, min_words=5):
        cleaned_notes = []
        cleaned_dates = []
        cleaned_times = []

        for note, date, time in zip(notes, dates, times):
            doc = nlp(note)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and token.lemma_ != "-PRON-"
                and token.is_alpha
            ]
            if len(tokens) >= min_words:
                cleaned_notes.append(" ".join(tokens))
                cleaned_dates.append(date)
                cleaned_times.append(time)

        return cleaned_notes, cleaned_dates, cleaned_times



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
                                        temp_df = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/{file}')
                                        temp_df = temp_df.dropna(subset=['Note', 'Date', 'Time'])
                                        temp_notes, temp_dates, temp_times = list(temp_df['Note']), list(temp_df['Date'].astype(str)), list(temp_df['Time'].astype(str))
                                        processed_temp_notes, processed_temp_dates, processed_temp_times = preprocessing(list(temp_df['Note']), list(temp_df['Date'].astype(str)), list(temp_df['Time'].astype(str)))
                                        nurse_notes_processed[sub_sub_folder.split(' ')[0]] = {'Date': processed_temp_dates, 'Time': processed_temp_times, 'Note': processed_temp_notes}
                                        nurse_notes[sub_sub_folder.split(' ')[0]] = {'Date': temp_dates, 'Time': temp_times, 'Note': temp_notes}


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


if __name__ == '__main__':
    main()