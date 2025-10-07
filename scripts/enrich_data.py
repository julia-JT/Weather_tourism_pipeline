import pandas as pd
import os
from datetime import datetime

# Папки (относительные пути от scripts/ к data/)
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
aggregated_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aggregated')
log_dir = aggregated_dir  # Лог в той же папке

# Создаем папки
os.makedirs(aggregated_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Основная функция
def create_aggregated_reports():
    # Найти последний enriched CSV (по дате)
    enriched_files = [f for f in os.listdir(enriched_dir) if f.startswith("weather_enriched_") and f.endswith(".csv")]
    if not enriched_files:
        print("ERROR: Нет enriched CSV файлов в data/enriched/")
        return
    enriched_files.sort(reverse=True)  # Самый свежий файл
    enriched_file = enriched_files[0]
    enriched_path = os.path.join(enriched_dir, enriched_file)
    
    # Прочитать enriched CSV
    try:
        df = pd.read_csv(enriched_path, encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Ошибка чтения {enriched_path}: {e}")
        return
    
    # Дата для имени файлов и as_of_date
    as_of_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    date_str = df['collection_time'].iloc[0].strftime("%Y%m%d") if not df.empty else datetime.now().strftime("%Y%m%d")
    
    # 1. travel_recommendations.csv
    # Топ-3 городов по comfort_index (уникальные, без повторов)
    df_sorted = df.dropna(subset=['comfort_index']).sort_values('comfort_index', ascending=False)
    top_3_cities = ', '.join(df_sorted['city_name'].head(3).drop_duplicates().tolist())  # Уникальные
    
    # Города для stay_home: comfort_index < 10 (неуютно)
    stay_home_cities = '; '.join(df[df['comfort_index'] < 10]['city_name'].unique().tolist())
    
    # Специальные рекомендации: для топ-3, на основе comfort_index и сезона
    special_recs = []
    for _, row in df_sorted.head(3).iterrows():
        if row['comfort_index'] > 15 and row['current_season'] in ['лето', 'весна']:
            special_recs.append(f"Активный отдых в {row['city_name']}")
        elif row['comfort_index'] > 10:
            special_recs.append(f"Прогулки в {row['city_name']}")
        else:
            special_recs.append(f"Музеи в {row['city_name']}")
    special_recommendations = '; '.join(special_recs)
    
    # Погодные предупреждения: на основе pop и visibility
    warnings = []
    for _, row in df.iterrows():
        if row['pop'] > 0.7:
            warnings.append(f"Высокая вероятность осадков в {row['city_name']}")
        if row['visibility'] < 5000:
            warnings.append(f"Низкая видимость в {row['city_name']}")
    weather_warnings = '; '.join(set(warnings))  # Уникальные
    
    travel_df = pd.DataFrame([{
        'top_3_cities': top_3_cities,
        'stay_home_cities': stay_home_cities,
        'special_recommendations': special_recommendations,
        'weather_warnings': weather_warnings,
        'as_of_date': as_of_date
    }])
    travel_path = os.path.join(aggregated_dir, 'travel_recommendations.csv')
    travel_df.to_csv(travel_path, index=False, encoding='utf-8')
    
    # 2. city_tourism_rating.csv
    # Группировка по городу, расчет среднего comfort_index
    city_ratings = df.groupby('city_name').agg(
        avg_comfort_index=('comfort_index', 'mean'),
        federal_district=('federal_district', 'first'),  # Берем первое значение
        population=('population', 'first'),
        tourism_season=('tourism_season', 'first')
    ).reset_index()
    # recommended_activity: на основе среднего comfort_index (с учетом pop из enriched, но усредняем)
    avg_pop = df.groupby('city_name')['pop'].mean().reset_index()
    city_ratings = pd.merge(city_ratings, avg_pop, on='city_name', how='left')
    city_ratings['recommended_activity'] = city_ratings.apply(
        lambda row: 'домашний отдых' if row['avg_pop'] >= 0.5 else (
            'активный туризм' if row['avg_comfort_index'] > 15 else (
                'прогулки' if row['avg_comfort_index'] > 10 else 'музеи'
            )
        ), axis=1
    )
    city_ratings.drop(columns=['avg_pop'], inplace=True)  # Убираем временное поле
    city_path = os.path.join(aggregated_dir, 'city_tourism_rating.csv')
    city_ratings.to_csv(city_path, index=False, encoding='utf-8')
    
    # 3. federal_districts_summary.csv
    # Группировка по федеральному округу
    district_summary = df.groupby('federal_district').agg(
        total_cities=('city_name', 'nunique'),
        comfortable_cities_count=('comfort_index', lambda x: (x > 10).sum()),  # Порог 10, адаптируйте
        avg_population=('population', 'mean'),
        avg_comfort_index=('comfort_index', 'mean')
    ).reset_index()
    district_path = os.path.join(aggregated_dir, 'federal_districts_summary.csv')
    district_summary.to_csv(district_path, index=False, encoding='utf-8')
    
    # Лог
    log_filename = f"aggregation_log_{date_str}.txt"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Обработан файл: {enriched_file}\n")
        log_file.write(f"Количество записей: {len(df)}\n")
        log_file.write("Созданные файлы:\n")
        log_file.write("- travel_recommendations.csv: топ-3 городов, stay_home, рекомендации, предупреждения\n")
        log_file.write("- city_tourism_rating.csv: средний comfort_index и активность по городам\n")
        log_file.write("- federal_districts_summary.csv: подсчет комфортных городов по округам\n")
        log_file.write("Логика:\n")
        log_file.write("- Топ-3: сортировка по comfort_index, уникальные\n")
        log_file.write("- Stay_home: comfort_index < 10\n")
        log_file.write("- Рекомендации: на основе comfort_index и сезона\n")
        log_file.write("- Предупреждения: pop > 0.7 или visibility < 5000\n")
    
    print(f"Aggregated отчеты сохранены в {aggregated_dir}")
    print(f"Лог сохранен в {log_path}")

# Запуск
if __name__ == "__main__":
    create_aggregated_reports()
