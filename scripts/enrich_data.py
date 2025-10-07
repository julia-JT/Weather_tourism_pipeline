import pandas as pd
import os
from datetime import datetime

# Папки (относительные пути от scripts/ к data/)
raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
log_dir = enriched_dir  # Лог в той же папке

# Создаем папки
os.makedirs(enriched_dir, exist_ok=True)

def calculate_comfort_index(row):
    # Формула comfort_index (адаптируйте веса под реальные данные)
    # Базовый сдвиг +20, чтобы значения были положительными
    comfort = (
        (row['temperature'] * 0.4) +
        (row['humidity'] * -0.2) +  # Высокая влажность снижает комфорт
        (row['clouds'] * -0.1) +   # Облачность снижает
        (row['pop'] * -0.3) +      # Осадки сильно снижают
        (row['wind_speed'] * -0.1) +  # Ветер снижает
        20  # Базовый сдвиг для положительных значений
    )
    return round(comfort, 2)

def determine_recommended_activity(comfort_index, pop):
    if comfort_index > 15 and (pop < 0.3 or pop is None):
        return "активный туризм"
    elif comfort_index > 10:
        return "культурный туризм"
    else:
        return "домашний отдых"

def determine_season_match(current_month, tourism_season):
    # Предполагаем tourism_season как строка месяцев, например "май-август"
    # Простая проверка: если текущий месяц в сезоне
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
    
    # Проверяем наличие необходимых столбцов
    required_cols = ['city_name', 'temperature', 'humidity', 'clouds', 'pop', 'wind_speed', 'federal_district']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Недостающие столбцы в {file_path}: {required_cols}")
        return
    
    # Рассчитываем comfort_index
    df['comfort_index'] = df.apply(calculate_comfort_index, axis=1)
    
    # Определяем recommended_activity
    df['recommended_activity'] = df.apply(lambda row: determine_recommended_activity(row['comfort_index'], row['pop']), axis=1)
    
    # Добавляем tourism_season (статический, адаптируйте под реальные данные)
    df['tourism_season'] = "май-август"  # Пример: сезон для большинства городов
    
    # Определяем tourist_season_match (на основе текущего месяца)
    current_month = datetime.now().month  # Текущий месяц (1-12)
    df['tourist_season_match'] = df.apply(lambda row: determine_season_match(current_month, row['tourism_season']), axis=1)
    
    # Сохраняем enriched файл
    enriched_file = f"weather_enriched_{os.path.basename(file_path)}"
    enriched_path = os.path.join(enriched_dir, enriched_file)
    df.to_csv(enriched_path, index=False, encoding='utf-8')
    print(f"Enriched data saved to {enriched_path}")

def main():
    # Найти все raw CSV файлы
    raw_files = [f for f in os.listdir(raw_dir) if f.startswith("weather_raw_") and f.endswith(".csv")]
    if not raw_files:
        print("ERROR: Нет raw CSV файлов в data/raw/")
        return
    
    # Обработать каждый файл
    processed_count = 0
    for file in raw_files:
        file_path = os.path.join(raw_dir, file)
        enrich_weather_data(file_path)
        processed_count += 1
    
    # Лог
    log_path = os.path.join(log_dir, "enriched_log.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Обработано файлов: {len(raw_files)} ({', '.join(raw_files)})\n")
        log_file.write(f"Успешно обработано: {processed_count}\n")
        log_file.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write("Comfort_index рассчитан по формуле; Recommended_activity: активный/культурный туризм или домашний отдых\n")
        log_file.write("Tourist_season_match: да/нет на основе текущего месяца\n")
        if processed_count < len(raw_files):
            log_file.write("Предупреждение: Некоторые файлы не обработаны\n")
    
    print(f"Enrichment completed. Log: {log_path}")

if __name__ == "__main__":
    main()
