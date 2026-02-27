import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as opy
import plotly.express as px
from django.shortcuts import render
from django.conf import settings
import glob

import plotly.graph_objects as go
import plotly.offline as opy
import plotly.express as px   # <-- add this

def dashboard_view(request):
    person = request.GET.get('person', 'P1')
    normalize = request.GET.get('normalize', 'norm')

    file_path = os.path.join(
        settings.BASE_DIR,
        'dashboard',
        'dataProcessed',
        f'temporalInformation{person}.json'
    )

    time_column = 'DateTime'
    df = pd.read_json(file_path)
    df = df.sort_values(time_column)

    columns_to_plot = [
        'Abbey Pain Scale','MUST','Weight (in KG)','Medley',
        'ABC for Challenging Behaviour','Barthel Index',
        'Cannard Assessment (FRASE scale)','Clinical Frailty Scale',
        "Cornell’s Scale",'Dewing Wandering',
        'Glasgow Coma Scale (GCS)','Mental Test Score',
        'Oral Cavity Assessment'
    ]

    nurse_columns = [
        'Nurse Notes - Positive Negative',
        'Nurse Notes - Grammar'
    ]

    df_copy = df.copy()
    df = df.dropna(subset=columns_to_plot, how='all')
    note_df = df_copy.dropna(subset=nurse_columns, how='all')

    # -------------------------
    # NORMALIZE CLINICAL SCALES
    # -------------------------
    if normalize == 'norm':
        for column in columns_to_plot:
            if column in df.columns:
                try:
                    col_min = df[column].min()
                    col_max = df[column].max()

                    if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
                        df[column] = (df[column] - col_min) / (col_max - col_min)
                    else:
                        df[column] = 0
                except:
                    pass
    

    # -------------------------
    # SIGNED NURSE SCORES
    # -------------------------
    def signed_score(cell):
        if isinstance(cell, dict):
            label = cell.get("label")
            score = cell.get("score")

            if score is None:
                return None, None

            # signed value: +1 for positive, -1 for negative
            if normalize == 'norm':
                signed_value = 1 if label in ["POSITIVE", "grammatical"] else -1
            else:
                signed_value = 100 if label in ["POSITIVE", "grammatical"] else 0
            return signed_value, score  # tuple
        return None, None

    fig = go.Figure()
    colors = px.colors.qualitative.Alphabet

    # -------------------------
    # ADD NURSE TRACES
    # -------------------------
    offset = len(columns_to_plot)

    for i, column in enumerate(nurse_columns):
        if column in note_df.columns:
            signed_results = note_df[column].apply(signed_score)
            signed_values = signed_results.apply(lambda x: x[0])
            probabilities = signed_results.apply(lambda x: x[1])
            opacity = probabilities * 0.5
            fig.add_trace(go.Scatter(
                x=note_df[time_column],
                y=signed_values,
                mode='lines+markers',
                opacity=0.5,
                marker=dict(
                    size=8,
                    color=colors[(offset + i) % len(colors)],
                    opacity=opacity
                    
                ),
                text=note_df['Nurse Note'],  # display note text
                hovertemplate='%{text}<br>Score: %{y}<extra></extra>',
                name=column
            ))

    # -------------------------
    # ADD CLINICAL TRACES
    # -------------------------
    for i, column in enumerate(columns_to_plot):
        if column in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_column],
                y=df[column],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)]
                ),
                name=column
            ))



    fig.update_layout(
        title=f'Data for {person}',
        xaxis_title='Time',
        yaxis_title='Values',
        template='plotly_white',
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d %H:%M'
        )
    )

    graph_div = opy.plot(fig, auto_open=False, output_type='div')
    print(normalize)
    context = {
        'graph': graph_div,
        'selected_person': person,
        'persons': [f'P{i}' for i in range(1, 21)],
        'selected_norm': normalize,
    }

    return render(request, 'dashboard/dashboard.html', context)




def demographics_view(request):
    data_folder = os.path.join(settings.BASE_DIR, 'dashboard', 'dataProcessed')
    file_pattern = os.path.join(data_folder, 'demographics*.json')

    files = sorted(
        glob.glob(file_pattern),
        key=lambda x: int(
            os.path.basename(x)
            .replace('demographics', '')
            .replace('.json', '')
            .replace('P', '')
        )
    )

    variable_data = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        patient_id = int(
            filename.replace('demographics', '').replace('.json', '').replace('P', '')
        )

        df = pd.read_json(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        row = df.iloc[0]

        for col in df.columns:
            value = pd.to_numeric(row[col], errors='coerce')
            value = 0 if pd.isna(value) else value   
            
            if col not in variable_data:
                variable_data[col] = []

            variable_data[col].append((patient_id, value))

    graphs = []
    colors = px.colors.qualitative.Alphabet  # same large palette

    for idx, (variable, values) in enumerate(variable_data.items()):
        if (variable == 'Resident Age') or (variable == 'Gender') or (variable == 'Number of Hospital Admissions in Previous 6 to 9 Months - Planned') \
            or (variable == 'Number of Falls in Previous 6 to 9 Months') or (variable == 'Number of Infections in Previous 6 to 9 Months'): 
            values = sorted(values, key=lambda x: x[0])
            patients = [f'Patient {v[0]}' for v in values]
            y_values = [v[1] for v in values]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=patients,
                y=y_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[idx % len(colors)]
                )
            ))

            fig.update_layout(
                title=f'{variable}',
                xaxis_title='Patient',
                yaxis_title=variable,
                template='plotly_white',
                xaxis=dict(
                    type='category',
                    categoryorder='array',
                    categoryarray=patients
                )
            )

            graph_div = opy.plot(fig, auto_open=False, output_type='div')
            graphs.append(graph_div)

    return render(request, 'dashboard/demographics.html', {
        'graphs': graphs
    })

def medications_view(request):

    def patient_sort_key(pid):
        return int(''.join(filter(str.isdigit, pid)) or 0)

    selected_patient = request.GET.get("patient", "all")

    data_folder = os.path.join(settings.BASE_DIR, 'dashboard', 'dataProcessed')
    file_pattern = os.path.join(data_folder, 'medications*.json')

    files = sorted(
        glob.glob(file_pattern),
        key=lambda x: patient_sort_key(
            os.path.basename(x).replace('medications', '').replace('.json', '')
        )
    )

    fig = go.Figure()

    symbol_map = {
        "regular": "circle",
        "prn": "diamond",
        "short course": "square",
        "unknown": "x"
    }

    color_map = {
        "regular": "green",
        "prn": "orange",
        "short course": "purple",
        "unknown": "gray"
    }

    shown_legends = set()

    for file_path in files:
        filename = os.path.basename(file_path)
        patient_id = filename.replace('medications', '').replace('.json', '')

        if selected_patient != "all" and selected_patient != patient_id:
            continue

        df = pd.read_json(file_path)

        # remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # strip spaces
        df.columns = df.columns.str.strip()

        # parse dates
        df['Date started (all meds)'] = pd.to_datetime(df['Date started (all meds)'], errors='coerce')
        df['Date discontinued'] = pd.to_datetime(df['Date discontinued'], errors='coerce')
        df['Date discontinued'] = df['Date discontinued'].fillna(pd.Timestamp.today())

        # sort meds
        df = df.sort_values('Date started (all meds)')

        for _, row in df.iterrows():
            start = row['Date started (all meds)']
            end = row['Date discontinued']

            if pd.isna(start) or pd.isna(end):
                continue

            med_name = str(row.get("Medications in Use in previous 6 to 9 months", "Unknown Med")).strip()
            if med_name == "":
                med_name = "Unknown Med"

            med_type_raw = row.get("Regular, PRN, or short course")
            if pd.isna(med_type_raw) or str(med_type_raw).strip() == "":
                med_type = "unknown"
            else:
                med_type = str(med_type_raw).strip().lower()

            symbol = symbol_map.get(med_type, "x")
            line_color = color_map.get(med_type, "gray")
            legend_group = med_type.title()

            label = f'{patient_id} — {med_name}'

            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[label, label],
                mode="lines",
                line=dict(width=3, color=line_color),
                showlegend=False,
                legendgroup=legend_group
            ))

            show_legend = legend_group not in shown_legends
            shown_legends.add(legend_group)

            fig.add_trace(go.Scatter(
                x=[start],
                y=[label],
                mode='markers',
                marker=dict(size=12, symbol=symbol, color=line_color),
                name=legend_group,
                legendgroup=legend_group,
                showlegend=show_legend,
                hovertemplate=f"{med_name}<br>Start: {start:%Y-%m-%d}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=[end],
                y=[label],
                mode='markers',
                marker=dict(size=12, symbol='triangle-down', color=line_color),
                name='',  
                legendgroup=legend_group,
                showlegend=False,
                hovertemplate=f"{med_name}<br>End: {end:%Y-%m-%d}<extra></extra>"
            ))


    fig.update_layout(
        title='Medication Timeline',
        xaxis_title='Date',
        yaxis_title='Patient / Medication',
        template='plotly_white',
        xaxis=dict(type="date")
    )

    graph_div = opy.plot(fig, auto_open=False, output_type='div')

    patient_ids = [
        os.path.basename(f).replace('medications','').replace('.json','')
        for f in files
    ]

    return render(request, 'dashboard/medications.html', {
        'graph': graph_div,
        'patients': sorted(patient_ids, key=patient_sort_key),
        'selected_patient': selected_patient
    })
