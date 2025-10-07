import pandas as pd
import os
from datetime import datetime
from collections import defaultdict

# Папки (предполагаем, что cleaned_data находится в data/cleaned/, а enriched в data/enriched/)
cleaned_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
cities_ref_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cities_reference.csv')

# Создаем папки
os.makedirs(enriched_dir, exist_ok=True)

# Словарь месяцев для парсинга диапазонов (на русском)
month_dict = {
    'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
    'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
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
    if not tourism_season or pd.isna(tourism_season):
        return "нет"
    
    tourism_season = tourism_season.strip().lower()  # Убираем пробелы и точки, приводим к нижнему
    
    if tourism_season == 'круглогодично':
        return "да"
    
    # Парсим диапазон, например "май-сентябрь" или "июнь-август."
    parts = tourism_season.replace('.', '').split('-')
    if len(parts) == 2:
        start_month_name = parts[0].strip()
        end_month_name = parts[1].strip()
        start_month = month_dict.get(start_month_name, 0)
        end_month = month_dict.get(end_month_name, 0)
        if start_month > 0 and end_month > 0 and start_month <= current_month <= end_month:
            return "да"
    return "нет"

def enrich_weather_data_for_date(date_str, file_paths):
    # Загружаем справочник городов
    if not os.path.exists(cities_ref_path):
        print(f"ERROR: Файл {cities_ref_path} не найден. Пропускаем обогащение для даты {date_str}.")
        return
    try:
        df_cities = pd.read_csv(cities_ref_path, encoding='utf-8')
        required_city_cols = ['city_name', 'federal_district', 'tourism_season']
        if not all(col in df_cities.columns for col in required_city_cols):
            print(f"ERROR: Недостающие столбцы в {cities_ref_path}: {required_city_cols}. Пропускаем.")
            return
        # Нормализуем city_name в справочнике для точного совпадения
        df_cities['city_name'] = df_cities['city_name'].str.lower().str.strip()
    except Exception as e:
        print(f"ERROR: Ошибка чтения {cities_ref_path}: {e}. Пропускаем.")
        return
    
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
    
    # Сохраняем оригинальный city_name (с заглавной русской буквы)
    combined_df['original_city_name'] = combined_df['city_name']
    
    # Нормализуем city_name в combined_df для точного совпадения
    combined_df['city_name'] = combined_df['city_name'].str.lower().str.strip()
    
    # Мерджим с данными из cities_reference.csv по city_name
    combined_df = combined_df.merge(
        df_cities[['city_name', 'federal_district', 'tourism_season', 'timezone', 'population']],
        on='city_name',
        how='left'
    )
    
    # Для городов, не найденных в справочнике, устанавливаем дефолты
    combined_df['federal_district'] = combined_df['federal_district'].fillna('Неизвестный федеральный округ')
    combined_df['tourism_season'] = combined_df['tourism_season'].fillna('добавьте город в справочник cities_reference.csv')
    combined_df['timezone'] = combined_df['timezone'].fillna('UTC+3')  # Дефолт
    combined_df['population'] = combined_df['population'].fillna(0)  # Дефолт
    
    # Восстанавливаем оригинальный city_name
    combined_df['city_name'] = combined_df['original_city_name']
    combined_df = combined_df.drop(columns=['original_city_name'])
    
    # Рассчитываем comfort_index
    combined_df['comfort_index'] = combined_df.apply(calculate_comfort_index, axis=1)
    
    # Определяем recommended_activity
    combined_df['recommended_activity'] = combined_df.apply(lambda row: determine_recommended_activity(row['comfort_index'], row['pop']), axis=1)
    
    # Определяем tourist_season_match (на основе месяца из date_str, а не текущего времени)
    current_month = int(date_str[4:6])  # Извлекаем месяц из YYYYMMDD (MM)
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
