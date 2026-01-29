import os
import re
import json
import pandas as pd

INPUT_FOLDER = 'data'
OUTPUT_FOLDER = 'dataProcessed'

def preprocessing(texts):
    temp_list = []
    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;])\s*", r"\1 ", text)
        text = text.replace(':`', '')
        text = text.replace('[', '')
        text = text.replace(']', '')
        temp_list.append(text)
    return temp_list

nurse_notes = {}
for folder in os.listdir(f'./{INPUT_FOLDER}/'):
    if '.' not in folder:
        for sub_folder in os.listdir(f'./{INPUT_FOLDER}/{folder}'):
            if '.' not in sub_folder and 'T1' in sub_folder:
                for sub_sub_folder in os.listdir(f'./{INPUT_FOLDER}/{folder}/{sub_folder}'):
                    if '.' not in sub_sub_folder: 
                        for file in os.listdir(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}'):
                            if file[0] != '.' and 'xlsx' in file:
                                if 'dailyNurseNotes' in file:
                                    nurse_notes[sub_sub_folder.split(' ')[0]] = preprocessing(list(pd.read_excel(f'./{INPUT_FOLDER}/{folder}/{sub_folder}/{sub_sub_folder}/{file}')['Note'].dropna()))

if not os.path.exists(f'./{OUTPUT_FOLDER}/'):
    os.makedirs(f'./{OUTPUT_FOLDER}/', exist_ok=False)  

json_str = json.dumps(nurse_notes, indent=4)
with open(f'./{OUTPUT_FOLDER}/nurseNotes.json', "w") as f:
    f.write(json_str)