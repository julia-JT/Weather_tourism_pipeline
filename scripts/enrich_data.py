import pandas as pd
import os
from datetime import datetime

# Папки (предполагаем, что cleaned_data находится в data/cleaned/, а enriched в data/enriched/)
cleaned_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')

# Создаем папки
os.makedirs(enriched_dir, exist_ok=True)

# Маппинг федеральных округов по городам (расширьте при необходимости)
federal_districts = {
    'Москва': 'Центральный федеральный округ',
    'Санкт-Петербург': 'Северо-Западный федеральный округ',
    'Новосибирск': 'Сибирский федеральный округ',
    'Казань': 'Приволжский федеральный округ',
    'Сочи': 'Южный федеральный округ',
    # Добавьте другие города, если нужно
}

def calculate_comfort_index(row):
    try:
        # Преобразуем в float, обрабатывая пустые строки как 0 или дефолт
        temp = float(row['temperature']) if pd.notna(row['temperature']) and row['temperature'] != '' else 20  # Дефолт 20°C
        hum = float(row['humidity']) if pd.notna(row['humidity']) and row['humidity'] != '' else 50   # Дефолт 50%
        clouds = float(row['clouds']) if pd.notna(row['clouds']) and row['clouds'] != '' else 50    # Дефолт 50%
        pop = float(row['pop']) if pd.notna(row['pop']) and row['pop'] != '' else 0               # Дефолт 0 (пустые строки -> 0)
        wind = float(row['wind_speed']) if pd.notna(row['wind_speed']) and row['wind_speed'] != '' else 5  # Дефолт 5 м/с
        
        # Формула comfort_index (адаптируйте веса при необходимости)
        comfort = (temp * 0.4) + (hum * -0.2) + (clouds * -0.1) + (pop * -0.3) + (wind * -0.1) + 20
        return round(comfort, 2)
    except (ValueError, TypeError) as e:
        print(f"WARNING: Ошибка расчёта comfort_index для строки {row.name}: {e}. Данные: temp={row['temperature']}, hum={row['humidity']}, clouds={row['clouds']}, pop={row['pop']}, wind={row['wind_speed']}")
        return 10  # Дефолт

def determine_recommended_activity(comfort_index, pop):
    try:
        pop_val = float(pop) if pd.notna(pop) and pop != '' else 0
        if pd.notna(comfort_index) and comfort_index > 15 and pop_val < 0.3:
            return "активный туризм"
        elif pd.notna(comfort_index) and comfort_index > 10:
            return "культурный туризм"
        else:
            return "домашний отдых"
    except (ValueError, TypeError) as e:
        print(f"WARNING: Ошибка определения activity: comfort={comfort_index}, pop={pop}. {e}")
        return "домашний отдых"

def determine_season_match(current_month, tourism_season):
    # Предполагаем tourism_season как строка месяцев, например "май-август"
    # Простая проверка: если текущий месяц в сезоне (адаптируйте логику)
    if tourism_season and str(current_month) in tourism_season:
        return "да"
    return "нет"

def enrich_weather_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Ошибка чтения {file_path}: {e}")
        return
    
    if df.empty:
        print(f"WARNING: Файл {file_path} пустой")
        return
    
    # Проверяем наличие необходимых столбцов (убрал federal_district из проверки, добавлю его ниже)
    required_cols = ['city_name', 'temperature', 'humidity', 'clouds', 'pop', 'wind_speed']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Недостающие столбцы в {file_path}: {required_cols}")
        return
    
    # Заменяем пустые строки на NaN для корректной обработки
    df = df.replace('', pd.NA)
    
    # Добавляем federal_district на основе city_name
    df['federal_district'] = df['city_name'].map(federal_districts).fillna('Неизвестный федеральный округ')
    
    # Рассчитываем comfort_index
    df['comfort_index'] = df.apply(calculate_comfort_index, axis=1)
    
    # Определяем recommended_activity
    df['recommended_activity'] = df.apply(lambda row: determine_recommended_activity(row['comfort_index'], row['pop']), axis=1)
    
    # Добавляем tourism_season (статический, адаптируйте под реальные данные)
    df['tourism_season'] = "май-август"  # Пример: сезон для большинства городов
    
    # Определяем tourist_season_match (на основе текущего месяца)
    current_month = datetime.now().month  # Текущий месяц (1-12)
    df['tourist_season_match'] = df.apply(lambda row: determine_season_match(current_month, row['tourism_season']), axis=1)
    
    # Формируем имя файла: извлекаем YYYYMMDD из cleaned файла и добавляем HHMM
    basename = os.path.basename(file_path)  # Например, "weather_cleaned_20231001.csv"
    if basename.startswith("weather_cleaned_") and basename.endswith(".csv"):
        date_part = basename[len("weather_cleaned_"):-len(".csv")]  # Извлекаем "20231001"
        current_time = datetime.now().strftime("%H%M")  # Текущее время, например "1430"
        enriched_file = f"weather_enriched_{date_part}_{current_time}.csv"
    else:
        # Fallback, если формат не совпадает
        enriched_file = f"weather_enriched_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    enriched_path = os.path.join(enriched_dir, enriched_file)
    
    # Сохраняем enriched файл
    try:
        df.to_csv(enriched_path, index=False, encoding='utf-8')
        print(f"SUCCESS: Enriched data saved to {enriched_path}")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения в {enriched_path}: {e}")

# Основная логика: обрабатываем все файлы в cleaned_dir
if __name__ == "__main__":
    if os.path.exists(cleaned_dir):
        for file in os.listdir(cleaned_dir):
            if file.startswith("weather_cleaned_") and file.endswith(".csv"):
                file_path = os.path.join(cleaned_dir, file)
                enrich_weather_data(file_path)
    else:
        print(f"ERROR: Папка {cleaned_dir} не существует")
