import pandas as pd
import os
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
    # Добавьте другие графики, если нужно: ![Comfort Index](data/visualizations/comfort_index_and_temp_by_city.png)\n\n
    
    # Витрины
    if 'city_tourism_rating' in data:
        md += "#### Рейтинг туризма по городам\n"
        md += data['city_tourism_rating'].to_markdown(index=False) + "\n\n"
    
    if 'federal_districts_summary' in data:
        md += "#### Сводка по федеральным округам\n"
        md += data['federal_districts_summary'].to_markdown(index=False) + "\n\n"
    
    if 'travel_recommendations' in data:
        md += "#### Рекомендации по путешествиям\n"
        md += data['travel_recommendations'].to_markdown(index=False) + "\n\n"
    
    # Прогнозы
    if not forecast_df.empty:
        md += "#### Прогноз температуры на завтра\n"
        forecast_md = forecast_df[['city', 'forecast_date', 'predicted_temp_day', 'predicted_temp_night']].to_markdown(index=False)
        md += forecast_md + "\n\n"
    
    return md

def update_readme(md_content):
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        start_marker = "<!-- WEATHER DATA START -->"
        end_marker = "<!-- WEATHER DATA END -->"
        
        if start_marker in content and end_marker in content:
            before = content.split(start_marker)[0]
            after = content.split(end_marker)[1]
            new_content = before + start_marker + "\n" + md_content + "\n" + end_marker + after
        else:
            # Если маркеров нет, добавить в конец
            new_content = content + "\n" + start_marker + "\n" + md_content + "\n" + end_marker
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("README.md обновлён.")
    else:
        print("README.md не найден.")

if __name__ == "__main__":
    data = load_aggregated_data()
    forecast_df = load_forecast_data()
    md_content = generate_markdown(data, forecast_df)
    update_readme(md_content)
