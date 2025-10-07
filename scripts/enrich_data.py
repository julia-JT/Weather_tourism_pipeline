import pandas as pd
import os
from datetime import datetime
from collections import defaultdict

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

def enrich_weather_data_for_date(date_str, file_paths):
    all_data = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if not df.empty:
                all_data.append(df)
            else:
                print(f"WARNING: Файл {file_path} пустой, пропускаем")
        except Exception as e:
            print(f"ERROR: Ошибка чтения {file_path}: {e}, пропускаем")
    
    if not all_data:
        print(f"WARNING: Нет данных для даты {date_str}")
        return
    
    # Объединяем все файлы за дату
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Удаляем дубликаты, если есть (на случай повторяющихся строк)
    combined_df = combined_df.drop_duplicates()
    
    # Проверяем наличие необходимых столбцов
    required_cols = ['city_name', 'temperature', 'humidity', 'clouds', 'pop', 'wind_speed']
    if not all(col in combined_df.columns for col in required_cols):
        print(f"ERROR: Недостающие столбцы для даты {date_str}: {required_cols}")
        return
    
    # Заменяем пустые строки на NaN для корректной обработки
    combined_df = combined_df.replace('', pd.NA)
    
    # Добавляем federal_district на основе city_name
    combined_df['federal_district'] = combined_df['city_name'].map(federal_districts).fillna('Неизвестный федеральный округ')
    
    # Рассчитываем comfort_index
    combined_df['comfort_index'] = combined_df.apply(calculate_comfort_index, axis=1)
    
    # Определяем recommended_activity
    combined_df['recommended_activity'] = combined_df.apply(lambda row: determine_recommended_activity(row['comfort_index'], row['pop']), axis=1)
    
    # Добавляем tourism_season (статический, адаптируйте под реальные данные)
    combined_df['tourism_season'] = "май-август"  # Пример: сезон для большинства городов
    
    # Определяем tourist_season_match (на основе текущего месяца)
    current_month = datetime.now().month  # Текущий месяц (1-12)
    combined_df['tourist_season_match'] = combined_df.apply(lambda row: determine_season_match(current_month, row['tourism_season']), axis=1)
    
    # Формируем имя файла: weather_enriched_YYYYMMDD.csv (без времени)
    enriched_file = f"weather_enriched_{date_str}.csv"
    enriched_path = os.path.join(enriched_dir, enriched_file)
    
    # Сохраняем enriched файл
    try:
        combined_df.to_csv(enriched_path, index=False, encoding='utf-8')
        print(f"SUCCESS: Enriched data for date {date_str} saved to {enriched_path} (объединено {len(file_paths)} файлов)")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения в {enriched_path}: {e}")

# Основная логика: группируем файлы по дате и обрабатываем
if __name__ == "__main__":
    if os.path.exists(cleaned_dir):
        # Группируем файлы по дате
        date_to_files = defaultdict(list)
        for file in os.listdir(cleaned_dir):
            if file.startswith("weather_cleaned_") and file.endswith(".csv"):
                # Извлекаем YYYYMMDD
                date_part = file[len("weather_cleaned_"):-len(".csv")]
                if len(date_part) == 8 and date_part.isdigit():  # Проверяем формат YYYYMMDD
                    date_to_files[date_part].append(os.path.join(cleaned_dir, file))
                else:
                    print(f"WARNING: Неверный формат даты в файле {file}, пропускаем")
        
        # Обрабатываем каждую дату
        for date_str, file_paths in date_to_files.items():
            enrich_weather_data_for_date(date_str, file_paths)
    else:
        print(f"ERROR: Папка {cleaned_dir} не существует")
