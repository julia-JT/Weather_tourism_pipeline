import pandas as pd
import os
from datetime import datetime

# Папки (относительные пути от scripts/ к data/)
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
reports_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aggregated')  # Переименовано
log_dir = reports_dir  # Лог в той же папке

# Создаем папки
os.makedirs(reports_dir, exist_ok=True)

# Основная функция
def create_reports():
    # Найти все enriched CSV файлы
    enriched_files = [f for f in os.listdir(enriched_dir) if f.startswith("weather_enriched_") and f.endswith(".csv")]
    if not enriched_files:
        print("ERROR: Нет enriched CSV файлов в data/enriched/")
        return
    
    # Прочитать и объединить все enriched файлы
    dfs = []
    for file in enriched_files:
        file_path = os.path.join(enriched_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            dfs.append(df)
        except Exception as e:
            print(f"ERROR: Ошибка чтения {file_path}: {e}")
            continue
    if not dfs:
        print("ERROR: Не удалось прочитать ни одного файла")
        return
    
    # Объединить все данные
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Добавить столбец as_of_date (формат YYYY-MM-DD hh:mm)
    df_all['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Теперь df_all содержит все данные за весь период с as_of_date
    
    # Витрина 1: Рейтинг городов для туризма
    df_city_rating = df_all.groupby('city_name').agg({
        'comfort_index': 'mean',
        'recommended_activity': lambda x: x.mode()[0] if not x.mode().empty else 'неизвестно',  # Самая частая активность
        'tourist_season_match': lambda x: x.mode()[0] if not x.mode().empty else 'неизвестно',
        'tourism_season': 'first'
    }).reset_index()
    df_city_rating['comfort_index'] = df_city_rating['comfort_index'].round(2)
    df_city_rating['tour_recommendation'] = df_city_rating.apply(
        lambda row: f"{row['recommended_activity']} в сезон" if row['tourist_season_match'] == 'да' else f"{row['recommended_activity']} вне сезона", axis=1
    )
    df_city_rating = df_city_rating.sort_values('comfort_index').rename(columns={'comfort_index': 'avg_comfort_index'})
    # Добавить as_of_date в витрину
    df_city_rating['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Перезапись витрины (без аккумуляции истории)
    city_rating_path = os.path.join(reports_dir, "city_tourism_rating.csv")
    df_city_rating.to_csv(city_rating_path, index=False, encoding='utf-8')
    
    # Витрина 2: Сводка по федеральным округам
    # Сначала группируем по city_name для уникальных городов
    df_city_agg = df_all.groupby('city_name').agg({
        'federal_district': 'first',
        'comfort_index': 'mean',
        'temperature': 'mean',
        'recommended_activity': lambda x: x.mode()[0] if not x.mode().empty else 'неизвестно'
    }).reset_index()
    df_city_agg['avg_comfort_index'] = df_city_agg['comfort_index'].round(2)
    df_city_agg['avg_temperature'] = df_city_agg['temperature'].round(2)
    
    # Фильтруем города, подходящие для туризма: comfort_index > 15 и recommended_activity != "домашний отдых"
    df_city_agg_filtered = df_city_agg[df_city_agg['recommended_activity'] != 'домашний отдых']
    # df_city_agg[(df_city_agg['avg_comfort_index'] > 15) & (df_city_agg['recommended_activity'] != 'домашний отдых')]
    
    # Получить все уникальные округа
    all_districts = df_city_agg['federal_district'].unique()
    df_district_summary = pd.DataFrame({'federal_district': all_districts})
    
    # Средняя температура по ВСЕМ городам в округе
    temp_summary = df_city_agg.groupby('federal_district')['avg_temperature'].mean().round(2).reset_index()
    df_district_summary = df_district_summary.merge(temp_summary, on='federal_district', how='left')
    
    # Количество комфортных городов (из filtered)
    comfortable_count = df_city_agg_filtered.groupby('federal_district').size().reset_index(name='comfortable_cities_count')
    df_district_summary = df_district_summary.merge(comfortable_count, on='federal_district', how='left').fillna(0)
    df_district_summary['comfortable_cities_count'] = df_district_summary['comfortable_cities_count'].astype(int)
    
    # Рекомендация
    df_district_summary['general_recommendation'] = df_district_summary.apply(
        lambda row: "Рекомендуется посетить" if row['avg_temperature'] > 10 and row['comfortable_cities_count'] > 0 else "Лучше остаться дома", axis=1
    )
    
    # Добавить as_of_date
    df_district_summary['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Перезапись витрины (без аккумуляции истории)
    district_summary_path = os.path.join(reports_dir, "federal_districts_summary.csv")
    df_district_summary.to_csv(district_summary_path, index=False, encoding='utf-8')
    
    # Витрина 3: Отчет для турагентств (travel_recommendations.csv)
    # Группируем по city_name для уникальных
    df_city_agg2 = df_all.groupby('city_name').agg({
        'comfort_index': 'mean',
        'recommended_activity': lambda x: x.mode()[0] if not x.mode().empty else 'неизвестно',
        'pop': 'mean',
        'temperature': 'mean',
        'clouds': 'mean',
        'humidity': 'mean'
    }).reset_index()
    
    # Топ-3 для поездок: Только города с recommended_activity != "домашний отдых", сортировка по avg_comfort_index descending
    df_for_travel = df_city_agg2[df_city_agg2['recommended_activity'] != "домашний отдых"]
    top_cities = df_for_travel.sort_values('comfort_index', ascending=False).head(3)[['city_name', 'comfort_index']]
    top_cities['comfort_index'] = top_cities['comfort_index'].round(2)
    
    stay_home_cities = df_city_agg2[df_city_agg2['recommended_activity'] == "домашний отдых"][['city_name', 'comfort_index']]
    stay_home_cities['comfort_index'] = stay_home_cities['comfort_index'].round(2)
    
    # Объединяем special_recommendations и weather_warnings в additional_notes с маской <Город>: <рекомендация>
    df_city_agg2['additional_notes'] = df_city_agg2.apply(
        lambda row: (
            ("Взять зонт" if row['pop'] > 0.5 else "") +
            ("; Взять теплую одежду" if row['temperature'] < 10 else "") +
            ("; Солнцезащитный крем" if row['temperature'] > 25 else "") +
            ("; Очень холодно, риск обморожения" if row['temperature'] < 0 else "") +
            ("; Сильные осадки, возможно снег/дождь" if row['pop'] > 0.8 else "") +
            ("; Плохая видимость из-за тумана/облачности" if row['clouds'] > 80 or row['humidity'] > 90 else "")
        ).strip("; "), axis=1
    )
    # Теперь формируем строку с городом: <Город>: <рекомендация>; <Город>: <рекомендация>...
    additional_notes_str = '; '.join([f"{row['city_name']}: {row['additional_notes']}" for _, row in df_city_agg2[df_city_agg2['additional_notes'] != ''].iterrows()])
    
    # Создать сводный DataFrame для витрины
    mart3_data = {
        'top_3_cities': [', '.join(top_cities['city_name'].tolist())],
        'stay_home_cities': [', '.join(stay_home_cities['city_name'].tolist())],
        'additional_notes': [additional_notes_str]
    }
    df_mart3 = pd.DataFrame(mart3_data)
    # Добавить as_of_date в витрину
    df_mart3['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Перезапись витрины (без аккумуляции истории)
    travel_rec_path = os.path.join(reports_dir, "travel_recommendations.csv")
    df_mart3.to_csv(travel_rec_path, index=False, encoding='utf-8')
    
    # Лог
    log_path = os.path.join(log_dir, "reports_log.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Обработано файлов: {len(enriched_files)} ({', '.join(enriched_files)})\n")
        log_file.write(f"Всего строк данных: {len(df_all)}\n")
        log_file.write(f"Дата загрузки: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write("Витрина 1: Рейтинг городов (city_tourism_rating.csv) - сортировка по avg_comfort_index\n")
        log_file.write("Витрина 2: Сводка по округам (federal_districts_summary.csv) - средняя temp по всем городам, комфортные города (comfort > 15 и не домашний отдых), рекомендация\n")
        log_file.write("Витрина 3: Рекомендации (travel_recommendations.csv) - топ-3, дома, дополнительные заметки\n")
    
    print(f"Отчеты созданы в {reports_dir} на основе всех данных за период")
    print(f"Лог: {log_path}")

# Запуск
if __name__ == "__main__":
    create_reports()
