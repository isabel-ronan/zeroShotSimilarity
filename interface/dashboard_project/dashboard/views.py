import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as opy
from django.shortcuts import render
from django.conf import settings
import glob


def dashboard_view(request):
    person = request.GET.get('person', 'P1')
    file_path = os.path.join(
        settings.BASE_DIR,
        'dashboard',
        'dataProcessedCSV',
        f'temporalInformation{person}.csv'
    )
    time_column = 'DateTime'
    df = pd.read_csv(file_path, parse_dates=[time_column], low_memory=False)
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
    data_folder = os.path.join(settings.BASE_DIR, 'dashboard', 'dataProcessedCSV')
    file_pattern = os.path.join(data_folder, 'demographics*.csv')

    files = sorted(
        glob.glob(file_pattern),
        key=lambda x: int(
            os.path.basename(x)
            .replace('demographics', '')
            .replace('.csv', '')
            .replace('P', '')
        )
    )

    variable_data = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        patient_id = int(
            filename.replace('demographics', '').replace('.csv', '').replace('P', '')
        )

        df = pd.read_csv(file_path)

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
            title=f'{variable} Comparison',
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