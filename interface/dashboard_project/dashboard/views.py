import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as opy
from django.shortcuts import render
from django.conf import settings
import glob
from textblob import TextBlob


def dashboard_view(request):
    person = request.GET.get('person', 'P1')
    file_path = os.path.join(
        settings.BASE_DIR,
        'dashboard',
        'dataProcessedINTERFACE',
        f'temporalInformation{person}.json'
    )
    time_column = 'DateTime'
    df = pd.read_json(file_path)
    df = df.sort_values('DateTime')

    columns_to_plot = ['Abbey Pain Scale','MUST','Weight (in KG)','Medley', 'ABC for Challenging Behaviour','Barthel Index','Cannard Assessment (FRASE scale)','Clinical Frailty Scale',"Cornell’s Scale",'Dewing Wandering','Glasgow Coma Scale (GCS)','Mental Test Score','Oral Cavity Assessment']
    
    df = df.dropna(subset=columns_to_plot, how='all')
    
    fig = go.Figure()

    for column in columns_to_plot:
        fig.add_trace(go.Scatter(
            x=df[time_column],
            y=df[column],
            mode='markers',
            marker=dict(size=12),
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

    context = {
        'graph': graph_div,
        'selected_person': person,
        'persons': [f'P{i}' for i in range(1, 21)]
    }

    return render(request, 'dashboard/dashboard.html', context)

def demographics_view(request):
    data_folder = os.path.join(settings.BASE_DIR, 'dashboard', 'dataProcessedINTERFACE')
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

            if not pd.isna(value):

                if col not in variable_data:
                    variable_data[col] = []

                variable_data[col].append((patient_id, value))

    graphs = []

    for variable, values in variable_data.items():

        # Sort by patient ID
        values = sorted(values, key=lambda x: x[0])

        patients = [f'Patient {v[0]}' for v in values]
        y_values = [v[1] for v in values]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=patients,
            y=y_values,
            mode='markers',
            marker=dict(size=10)
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

    data_folder = os.path.join(settings.BASE_DIR, 'dashboard', 'dataProcessedINTERFACE')
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

def sentiment_view(request):

    def patient_sort_key(pid):
        return int(''.join(filter(str.isdigit, pid)) or 0)

    selected_patient = request.GET.get("patient", "P1")
    selected_note_type = request.GET.get("note_type", "Nurse Note")

    note_columns = [
        "Nurse Note",
        "Carer Note",
        "Multi-Disciplinary Note"
    ]

    # fallback if URL param invalid
    if selected_note_type not in note_columns:
        selected_note_type = "Nurse Note"

    # -------------------------
    # load files
    # -------------------------
    data_folder = os.path.join(settings.BASE_DIR, 'dashboard', 'dataProcessedINTERFACE')

    files = sorted(
        glob.glob(os.path.join(data_folder, "temporalInformation*.json")),
        key=lambda x: patient_sort_key(
            os.path.basename(x).replace("temporalInformation","").replace(".json","")
        )
    )

    fig = go.Figure()

    # -------------------------
    # loop files
    # -------------------------
    for file_path in files:

        filename = os.path.basename(file_path)
        patient_id = filename.replace("temporalInformation","").replace(".json","")

        if selected_patient != "all" and selected_patient != patient_id:
            continue

        df = pd.read_json(file_path)

        df.columns = df.columns.str.strip()

        # skip if chosen column doesn't exist
        if selected_note_type not in df.columns:
            continue

        # parse datetime
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime", selected_note_type])

        # compute sentiment
        sentiments = []
        hover_text = []

        for note in df[selected_note_type]:

            score = TextBlob(note).sentiment.polarity

            sentiments.append(score)

            hover_text.append(
                f"<b>{patient_id}</b><br>"
                f"{selected_note_type}:<br>"
                f"{note}<br>"
                f"Sentiment: {score:.2f}"
            )

        df["Sentiment"] = sentiments

        # plot trace
        fig.add_trace(go.Scatter(
            x=df["DateTime"],
            y=df["Sentiment"],
            mode="markers+lines",
            name=patient_id,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>"
        ))

    # -------------------------
    # layout
    # -------------------------
    fig.update_layout(
        title=f"Sentiment Over Time — {selected_note_type}",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_white",
        xaxis=dict(type="date"),
        yaxis=dict(range=[-1,1])
    )

    graph_div = opy.plot(fig, auto_open=False, output_type='div')

    # patient list
    patient_ids = [
        os.path.basename(f).replace("temporalInformation","").replace(".json","")
        for f in files
    ]

    return render(request, "dashboard/sentiment.html", {
        "graph": graph_div,
        "patients": sorted(patient_ids, key=patient_sort_key),
        "selected_patient": selected_patient,
        "note_types": note_columns,
        "selected_note_type": selected_note_type
    })
