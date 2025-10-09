import pandas as pd
import os
import re
from datetime import datetime

# Папки (добавил проверки существования папок)
aggregated_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aggregated')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'forecast')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')

def load_aggregated_data():
    data = {}
    if not os.path.exists(aggregated_dir):
        print(f"Папка {aggregated_dir} не найдена. Пропускаем загрузку aggregated данных.")
        return data
    for file in ['city_tourism_rating.csv', 'federal_districts_summary.csv', 'travel_recommendations.csv']:
        file_path = os.path.join(aggregated_dir, file)
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'as_of_date' in df.columns:
                    df['as_of_date'] = pd.to_datetime(df['as_of_date'], errors='coerce')
                    max_date = df['as_of_date'].max()
                    df = df[df['as_of_date'] == max_date]
                data[file.split('.')[0]] = df
            else:
                print(f"Файл {file} не найден.")
        except Exception as e:
            print(f"Ошибка загрузки {file}: {e}")
    return data

def load_forecast_data():
    if not os.path.exists(forecasts_dir):
        print(f"Папка {forecasts_dir} не найдена. Пропускаем загрузку forecast данных.")
        return pd.DataFrame()
    forecast_file = os.path.join(forecasts_dir, 'Forecast.csv')
    try:
        if os.path.exists(forecast_file):
            df = pd.read_csv(forecast_file)
            if 'as_of_date' in df.columns:
                df['as_of_date'] = pd.to_datetime(df['as_of_date'], errors='coerce')
                max_date = df['as_of_date'].max()
                df = df[df['as_of_date'] == max_date]
            return df
    except Exception as e:
        print(f"Ошибка загрузки Forecast.csv: {e}")
    return pd.DataFrame()

def generate_markdown(data, forecast_df):
    md = "### Данные о погоде\n\n"
    
    # Графики
    md += "#### Графики\n"
    md += "![Динамика температуры](data/visualizations/temperature_trend.png)\n\n"
    
    # Добавьте здесь генерацию таблиц из data и forecast_df
    # Например:
    if 'city_tourism_rating' in data:
        md += "#### Рейтинг туризма по городам\n"
        md += data['city_tourism_rating'].to_markdown(index=False) + "\n\n"
    # Аналогично для других DataFrame
    
    return md

# Основная логика
try:
    # Загрузка данных
    data = load_aggregated_data()
    forecast_df = load_forecast_data()
    
    # Генерация Markdown
    md = generate_markdown(data, forecast_df)
    
    # Проверка README на конфликты
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README.md не найден: {readme_path}")
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверка на merge-конфликты
    if '<<<<<<< HEAD' in content or '=======' in content or '>>>>>>>' in content:
        raise ValueError("README.md содержит неразрешённые merge-конфликты. Разрешите их вручную перед запуском скрипта.")
    
    # Замена секции (предполагаемая логика; добавьте, если отсутствует)
    pattern = r'<!-- WEATHER DATA START -->.*?<!-- WEATHER DATA END -->'
    replacement = f'<!-- WEATHER DATA START -->\n{md}\n<!-- WEATHER DATA END -->'
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        print("Секция WEATHER DATA не найдена в README. Проверьте маркеры.")
    else:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("README.md успешно обновлён.")
        
except Exception as e:
    print(f"Ошибка в update_readme.py: {e}")
    exit(1)  # Остановить workflow, если ошибка
