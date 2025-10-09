import pandas as pd
import os
import re
from datetime import datetime

# Папки
aggregated_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aggregated')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'forecast')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')

def load_aggregated_data():
    data = {}
    for file in ['city_tourism_rating.csv', 'federal_districts_summary.csv', 'travel_recommendations.csv']:
        file_path = os.path.join(aggregated_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'as_of_date' in df.columns:
                df['as_of_date'] = pd.to_datetime(df['as_of_date'])
                max_date = df['as_of_date'].max()
                df = df[df['as_of_date'] == max_date]
            data[file.split('.')[0]] = df
        else:
            print(f"Файл {file} не найден.")
    return data

def load_forecast_data():
    forecast_file = os.path.join(forecasts_dir, 'Forecast.csv')
    if os.path.exists(forecast_file):
        df = pd.read_csv(forecast_file)
        if 'as_of_date' in df.columns:
            df['as_of_date'] = pd.to_datetime(df['as_of_date'])
            max_date = df['as_of_date'].max()
            df = df[df['as_of_date'] == max_date]
        return df
    return pd.DataFrame()

def generate_markdown(data, forecast_df):
    md = "### Данные о погоде\n\n"
    
    # Графики
    md += "#### Графики\n"
    md += "![Динамика температуры](data/visualizations/temperature_trend.png)\n\n"
   
