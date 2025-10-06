import pandas as pd
import os
from datetime import datetime

# Папки (относительные пути от scripts/ к data/)
cleaned_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
log_dir = enriched_dir  # Лог в той же папке

# Создаем папки
os.makedirs(enriched_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Справочник городов (создаем CSV)
def create_city_reference():
    reference_data = [
        {"city_name": "Москва", "federal_district": "Центральный", "timezone": "UTC+3", "population": 12500000, "tourism_season": "Круглогодично"},
        {"city_name": "Санкт-Петербург", "federal_district": "Северо-Западный", "timezone": "UTC+3", "population": 5400000, "tourism_season": "Май-Сентябрь"},
        {"city_name": "Сочи", "federal_district": "Южный", "timezone": "UTC+3", "population": 400000, "tourism_season": "Май-Октябрь"},
        {"city_name": "Казань", "federal_district": "Приволжский", "timezone": "UTC+3", "population": 1250000, "tourism_season": "Май-Сентябрь"},
        {"city_name": "Новосибирск", "federal_district": "Сибирский", "timezone": "UTC+7", "population": 1620000, "tourism_season": "Июнь-Август"}
    ]
    df_ref = pd.DataFrame(reference_data)
    ref_path = os.path.join(enriched_dir, "cities_reference.csv")
    df_ref.to_csv(ref_path, index=False, encoding='utf-8')
    print(f"Справочник городов создан: {ref_path}")
    return df_ref

# Функция для определения сезона по месяцу
def get_season(month):
    if month in [12, 1, 2]:
        return "зима"
    elif month in [3, 4, 5]:
        return "весна"
    elif month in [6, 7, 8]:
        return "лето"
    else:
        return "осень"

# Функция для comfort_index (обновлена: добавлены visibility и pop)
def calculate_comfort_index(temp, humidity, wind_speed, visibility, pop):
    # Пример: штраф за отклонение от идеала (20°C, 50% влажность, 5 м/с ветер, 10000 м видимость, 0% осадков)
    temp_penalty = abs(temp - 20) * 0.5  # Больше штраф за холод/жар
    humidity_penalty = abs(humidity - 50) / 10  # Нормализуем
    wind_penalty = wind_speed * 2  # Ветер сильно влияет
    visibility_penalty = max(0, (10000 - visibility) / 1000)  # Штраф за низкую видимость (идеал 10000 м)
    pop_penalty = pop * 10  # Штраф за вероятность осадков (0-1, умножаем на 10 для масштаба)
    return round(temp_penalty + humidity_penalty + wind_penalty + visibility_penalty + pop_penalty, 2)

# Функция для recommended_activity (обновлена: явно не рекомендуем прогулки при pop >= 0.5)
def get_recommended_activity(comfort_index, season, pop):
    if pop >= 0.5:
        return "домашний отдых"  # Явно не рекомендуем прогулки при высокой вероятности осадков
    if comfort_index < 10 and season in ["лето", "весна"]:
        return "прогулки"
    elif comfort_index < 20:
        return "музеи"
    else:
        return "домашний отдых"

# Функция для tourist_season_match
def check_season_match(tourism_season, current_month):
    if tourism_season == "Круглогодично":
        return "да"
    elif tourism_season == "Май-Сентябрь":
        return "да" if current_month in [5,6,7,8,9] else "нет"
    elif tourism_season == "Май-Октябрь":
        return "да" if current_month in [5,6,7,8,9,10] else "нет"
    elif tourism_season == "Июнь-Август":
        return "да" if current_month in [6,7,8] else "нет"
    else:
        return "нет"

# Основная функция
def enrich_weather_data():
    # Создать справочник
    df_ref = create_city_reference()
    
    # Найти последний cleaned CSV (по дате)
    cleaned_files = [f for f in os.listdir(cleaned_dir) if f.startswith("weather_cleaned_") and f.endswith(".csv")]
    if not cleaned_files:
        print("ERROR: Нет cleaned CSV файлов в data/cleaned/")
        return
    cleaned_files.sort(reverse=True)  # Самый свежий файл
    cleaned_file = cleaned_files[0]
    cleaned_path = os.path.join(cleaned_dir, cleaned_file)
    
    # Прочитать cleaned CSV
    try:
        df_cleaned = pd.read_csv(cleaned_path, encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Ошибка чтения {cleaned_path}: {e}")
        return
    
    # Объединить с справочником
    df_enriched = pd.merge(df_cleaned, df_ref, on='city_name', how='left')
    
    # Извлечь месяц из collection_time для сезона
    df_enriched['collection_time'] = pd.to_datetime(df_enriched['collection_time'], format="%d.%m.%Y %H:%M:%S")
    df_enriched['current_season'] = df_enriched['collection_time'].dt.month.apply(get_season)
    
    # Вычислить comfort_index (обновлено: добавлены visibility и pop)
    df_enriched['comfort_index'] = df_enriched.apply(
        lambda row: calculate_comfort_index(row['temperature'], row['humidity'], row['wind_speed'], row['visibility'], row['pop']), axis=1
    )
    
    # Вычислить recommended_activity (обновлено: передаем pop для явной проверки)
    df_enriched['recommended_activity'] = df_enriched.apply(
        lambda row: get_recommended_activity(row['comfort_index'], row['current_season'], row['pop']), axis=1
    )
    
    # Вычислить tourist_season_match
    df_enriched['tourist_season_match'] = df_enriched.apply(
        lambda row: check_season_match(row['tourism_season'], row['collection_time'].month), axis=1
    )
    
    # Дата для имени файла
    date_str = df_enriched['collection_time'].iloc[0].strftime("%Y%m%d") if not df_enriched.empty else datetime.now().strftime("%Y%m%d")
    
    # Сохранить enriched CSV
    enriched_filename = f"weather_enriched_{date_str}.csv"
    enriched_path = os.path.join(enriched_dir, enriched_filename)
    df_enriched.to_csv(enriched_path, index=False, encoding='utf-8')
    
    # Лог
    log_filename = f"enrichment_log_{date_str}.txt"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Обработан файл: {cleaned_file}\n")
        log_file.write(f"Количество записей: {len(df_enriched)}\n")
        log_file.write("Добавленные поля:\n")
        log_file.write("- Из справочника: federal_district, timezone, population, tourism_season\n")
        log_file.write("- Вычисляемые: current_season, comfort_index, recommended_activity, tourist_season_match\n")
        log_file.write("Формулы:\n")
        log_file.write("- comfort_index: (abs(temp-20)*0.5 + abs(humidity-50)/10 + wind_speed*2 + max(0, (10000-visibility)/1000) + pop*10)\n")
        log_file.write("- recommended_activity: на основе comfort_index, сезона и pop (прогулки не рекомендуются при pop >= 0.5)\n")
        log_file.write("- tourist_season_match: сравнение текущего месяца с tourism_season\n")
    
    print(f"Enriched данные сохранены в {enriched_path}")
    print(f"Лог сохранен в {log_path}")

# Запуск
if __name__ == "__main__":
    enrich_weather_data()
