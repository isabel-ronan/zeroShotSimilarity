from django.shortcuts import render

import pandas as pd
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def dashboard_view(request):
    charts = []

    for csv_file in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        chart_data = {
            "x": df['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
            "y": df['value1'].tolist(),
            "name": csv_file.name
        }
        charts.append(chart_data)

    return render(request, "dashboard/dashboard.html", {"charts": charts})