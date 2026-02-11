import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as opy
from django.shortcuts import render
from django.conf import settings


def dashboard_view(request):
    # Get file and sort based on DateTime column.
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
    df = df.dropna(subset=['DateTime', 'Date', 'Time'])
    
    columns_to_plot = ['Abbey Pain Scale','MUST','Weight (in KG)','Medley', 'ABC for Challenging Behaviour','Barthel Index','Cannard Assessment (FRASE scale)','Clinical Frailty Scale',"Cornell’s Scale",'Dewing Wandering','Glasgow Coma Scale (GCS)','Mental Test Score','Oral Cavity Assessment']

    # Create Plotly figure
    fig = go.Figure()

    for column in columns_to_plot:
        fig.add_trace(go.Scatter(
            x=df[time_column],
            y=df[column],
            mode='markers',
            marker=dict(size=6),
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

    # Convert to HTML
    graph_div = opy.plot(fig, auto_open=False, output_type='div')

    context = {
        'graph': graph_div,
        'selected_person': person,
        'persons': [f'P{i}' for i in range(1, 21)]
    }

    return render(request, 'dashboard/dashboard.html', context)
