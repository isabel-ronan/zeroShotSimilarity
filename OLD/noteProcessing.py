import os
import pandas as pd
import numpy as np
import random
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Warnings
import warnings

def main():

    # Ignore user warnings.
    warnings.filterwarnings("ignore", category=UserWarning)

    # Initialize constant variables.
    INPUT_FOLDER = 'data'
    OUTPUT_FOLDER = 'dataProcessed'

    # Random States
    def set_random_states(random_state):
        # Set various random seeds.
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        os.environ["PYTHONHASHSEED"] = str(random_state)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        return random_state
    RANDOM_STATE = set_random_states(1618)

    # Load classifiers
    device = 0 if torch.cuda.is_available() else -1

    # https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
    pos_neg_class = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True
    )

    # https://huggingface.co/agentlans/snowflake-arctic-xs-grammar-classifier
    grammar_class = pipeline(
        "text-classification",
        model="agentlans/snowflake-arctic-xs-grammar-classifier",
        device=device,
        truncation=True
    )

    # Fine-Tuned Met/Unmet Needs Model
    # Load fine-tuned model.
    # Make constant variables.
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    PEFT_HEAD = "./classifiers/peft/synth_lora_model_bioclinicalbert"
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model = PeftModel.from_pretrained(base_model, PEFT_HEAD)

    tokenizer = AutoTokenizer.from_pretrained(PEFT_HEAD)
    palliative_class = pipeline(
        "text-classification",
        model=model,
        device=device,
        tokenizer=tokenizer,
        truncation=True
    )
    palliative_class.model.config.max_position_embeddings = 512

    classifier_dict = {"Positive Negative": pos_neg_class, "Grammar": grammar_class, "Met Unmet": palliative_class}

    # Process all T1 files (this can be adjusted later for T2 also). 
    # Minor processing of data points to make separate date and time columns where applicable and removing NaN values.
    all_data = {}
    for folder in os.listdir(f'./{INPUT_FOLDER}/'):
        if '.' not in folder:
            for sub_folder in os.listdir(f'./{INPUT_FOLDER}/{folder}'):
                if '.' not in sub_folder and 'T1' in sub_folder:
                    for sub_sub_folder in os.listdir(f'./{INPUT_FOLDER}/{folder}/{sub_folder}'):
                        if '.' not in sub_sub_folder: 
                            if 'P15' not in sub_sub_folder:
                                # -------- Carer Notes --------
                                carer_notes_1 = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/carerNotes1_{sub_sub_folder.split(' ')[0]}.xlsx', skiprows= 3, header=[0, 1])
                                carer_notes_2 = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/carerNotes2_{sub_sub_folder.split(' ')[0]}.xlsx', skiprows= 3, header=[0, 1])
                                carer_notes_1.columns = [
                                "_".join([str(x) for x in col if "Unnamed" not in str(x)]).strip()
                                for col in carer_notes_1.columns
                                ]
                                carer_notes_2.columns = [
                                "_".join([str(x) for x in col if "Unnamed" not in str(x)]).strip()
                                for col in carer_notes_2.columns
                                ]
                                carer_notes_1 = carer_notes_1[['Date', 'Activity']]
                                carer_notes_2 = carer_notes_2[['Date', 'Activity']]
                                carer_notes = pd.concat([carer_notes_1, carer_notes_2])
                                carer_notes['datetime'] = pd.to_datetime(carer_notes['Date'], format='%d/%m/%y %H:%M')
                                carer_notes['Date'] = carer_notes['datetime'].dt.date
                                carer_notes['Time'] = carer_notes['datetime'].dt.time
                                carer_notes = carer_notes[['Date', 'Time', 'Activity']]
                                carer_notes = carer_notes.dropna(subset=['Date', 'Time', 'Activity'])
                                carer_notes['Carer Note'] = carer_notes['Activity'].values.tolist()
                                carer_notes = carer_notes.drop(columns=['Activity'])
                            else:
                                # -------- Carer Notes --------
                                carer_notes = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/carerNotes_{sub_sub_folder.split(' ')[0]}.xlsx', skiprows= 3, header=[0, 1])
                                carer_notes.columns = [
                                "_".join([str(x) for x in col if "Unnamed" not in str(x)]).strip()
                                for col in carer_notes.columns
                                ]
                                carer_notes = carer_notes[['Date', 'Activity']]
                                carer_notes['datetime'] = pd.to_datetime(carer_notes['Date'], format='%d/%m/%y %H:%M')
                                carer_notes['Date'] = carer_notes['datetime'].dt.date
                                carer_notes['Time'] = carer_notes['datetime'].dt.time
                                carer_notes = carer_notes[['Date', 'Time', 'Activity']]
                                carer_notes = carer_notes.dropna(subset=['Date', 'Time', 'Activity'])
                                carer_notes['Carer Note'] = carer_notes['Activity'].values.tolist()
                                carer_notes = carer_notes.drop(columns=['Activity'])
                            
                            # -------- Nurse Notes --------
                            daily_nurse = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/dailyNurseNotes_{sub_sub_folder.split(' ')[0]}.xlsx')
                            daily_nurse = daily_nurse[['Date', 'Time', 'Note']]
                            daily_nurse = daily_nurse.dropna(subset=['Note', 'Date', 'Time'])
                            daily_nurse['Nurse Note'] = daily_nurse['Note'].values.tolist()
                            daily_nurse = daily_nurse.drop(columns=['Note'])   
                            
                            # -------- Monthly --------
                            monthly = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/monthly_{sub_sub_folder.split(' ')[0]}.xlsx')
                            monthly = monthly.drop(columns=['Resident Study Number'])
                            # -------- Multi-Disciplinary Notes --------
                            multi = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/multiDisciplinaryNotes_{sub_sub_folder.split(' ')[0]}.xlsx')
                            multi['Multi-Disciplinary Note'] = multi['Note'].values.tolist()
                            multi = multi.drop(columns=['Resident Study Number', 'Delirium Indicated', 'Note', 'Note Type'])
                            multi = multi.dropna(subset=['Multi-Disciplinary Note'])
                            # -------- Quarterly --------
                            quarterly = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/quarterly_{sub_sub_folder.split(' ')[0]}.xlsx')
                            quarterly = quarterly.drop(columns=['Resident Study Number'])
                            # Big Concatenation 
                            big_df = pd.concat([carer_notes, daily_nurse, multi, monthly, quarterly])
                            cleaned_dates = []
                            for date in big_df['Date']:
                                date = (str(date)).strip()
                                cleaned_dates.append(date.split(' ')[0])
                            big_df['Date'] = cleaned_dates
                            big_df['DateTime'] = pd.to_datetime(big_df['Date'].astype(str) + ' ' + big_df['Time'].astype(str), format='mixed')
                            big_columns = [(column.split('\n')[0]).replace("'", '') for column in big_df.columns]
                            big_df.columns = big_columns
                            big_df = big_df.sort_values('DateTime')
                            # -------- Meds --------
                            meds = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/meds_{sub_sub_folder.split(' ')[0]}.xlsx')
                            meds = meds.dropna(subset=['Medications in Use in previous 6 to 9 months'])
                            meds = meds[['Medications in Use in previous 6 to 9 months', 'Regular, PRN, or short course', 'Date started (all meds)', 'Date discontinued']]
                            meds_columns = [(column.split('\n')[0]).replace("'", '') for column in meds.columns]
                            meds.columns = meds_columns
                            meds.columns = [
                                "".join([str(x) for x in col if "Unnamed" not in str(x)]).strip()
                                for col in meds.columns
                                ]
                            meds = meds.drop(columns=[column for column in meds.columns if 'Unnamed' in column])
                            # -------- Demographics --------
                            t1 = pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/t1_{sub_sub_folder.split(' ')[0]}.xlsx')   
                            t1_columns = [(column.split('\n')[0]).replace("'", '') for column in t1.columns]
                            t1.columns = t1_columns
                            t1.columns = [
                                "".join([str(x) for x in col if "Unnamed" not in str(x)]).strip()
                                for col in t1.columns
                                ]
                            t1 = t1.drop(columns=[column for column in t1.columns if 'Unnamed' in column])
                            all_data[sub_sub_folder.split(' ')[0]] = {'Temporal Information': big_df, 'Medications': meds, 'Demographics': t1}                      


    for patient, temp_all_data in all_data.items():
        texts = temp_all_data['Temporal Information']['Nurse Note']
        mask = texts.notna()
        for key, classifier in classifier_dict.items():
            encoded_values = classifier(texts[mask].tolist(), trunaction = True, max_length = 512, batch_size=32)
            encoded_series = pd.Series(index=texts.index, dtype=object)
            encoded_series[mask] = list(encoded_values)
            temp_all_data['Temporal Information'][f"Nurse Notes - {key}"] = encoded_series

    # If saving folder does not exist, make it.
    if not os.path.exists(f'./{OUTPUT_FOLDER}/'):
        os.makedirs(f'./{OUTPUT_FOLDER}/', exist_ok=False)

    for patient, values in all_data.items():
        for value_type, value in values.items():
            value.to_json(f'./{OUTPUT_FOLDER}/{value_type[:1].lower() + value_type.replace(' ', '')[1:]}{patient}.json', orient='records', date_format='iso')


if __name__ == '__main__':
    main()