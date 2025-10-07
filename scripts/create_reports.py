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
    df_city_rating.to_csv(os.path.join(reports_dir, "city_tourism_rating.csv"), index=False, encoding='utf-8')
    
    # Витрина 2: Сводка по федеральным округам
    df_district_summary = df_all.groupby('federal_district').agg({
        'temperature': 'mean',
        'comfort_index': lambda x: (x < 20).sum()  # Количество комфортных городов
    }).reset_index()
    df_district_summary['avg_temperature'] = df_district_summary['temperature'].round(2)
    df_district_summary['comfortable_cities_count'] = df_district_summary['comfort_index']
    df_district_summary['general_recommendation'] = df_district_summary.apply(
        lambda row: "Рекомендуется посетить" if row['avg_temperature'] > 10 and row['comfortable_cities_count'] > 0 else "Лучше остаться дома", axis=1
    )
    df_district_summary = df_district_summary[['federal_district', 'avg_temperature', 'comfortable_cities_count', 'general_recommendation']]
    # Добавить as_of_date в витрину
    df_district_summary['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    df_district_summary.to_csv(os.path.join(reports_dir, "federal_districts_summary.csv"), index=False, encoding='utf-8')
    
    # Витрина 3: Отчет для турагентств
    df_travel_rec = df_all.copy()
    # Топ-3 для поездок: Сортировка по comfort_index, фильтр по recommended_activity
    top_cities = df_travel_rec[df_travel_rec['recommended_activity'] != "домашний отдых"].sort_values('comfort_index').head(3)[['city_name', 'comfort_index']]
    stay_home_cities = df_travel_rec[df_travel_rec['recommended_activity'] == "домашний отдых"][['city_name', 'comfort_index']]
    # Специальные рекомендации
    df_travel_rec['special_recommendations'] = df_travel_rec.apply(
        lambda row: (
            ("Взять зонт" if row['pop'] > 0.5 else "") +
            ("; Теплую одежду" if row['temperature'] < 10 else "") +
            ("; Солнцезащитный крем" if row['temperature'] > 25 else "")
        ).strip("; "), axis=1
    )
    # Уведомления о плохой погоде
    df_travel_rec['weather_warnings'] = df_travel_rec.apply(
        lambda row: (
            ("Очень холодно, риск обморожения" if row['temperature'] < 0 else "") +
            ("; Сильные осадки, возможно снег/дождь" if row['pop'] > 0.8 else "") +
            ("; Плохая видимость из-за тумана/облачности" if row['clouds'] > 80 or row['humidity'] > 90 else "")
        ).strip("; "), axis=1
    )
    special_recs = df_travel_rec[['city_name', 'special_recommendations']].drop_duplicates()
    weather_warns = df_travel_rec[['city_name', 'weather_warnings']].drop_duplicates()
    
    # Создать сводный DataFrame для витрины
    mart3_data = {
        'top_3_cities': [', '.join(top_cities['city_name'].tolist())],
        'stay_home_cities': [', '.join(stay_home_cities['city_name'].tolist())],
        'special_recommendations': ['; '.join(special_recs['special_recommendations'].tolist())],
        'weather_warnings': ['; '.join(weather_warns['weather_warnings'].tolist())]  # Новое поле
    }
    df_mart3 = pd.DataFrame(mart3_data)
    # Добавить as_of_date в витрину
    df_mart3['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    df_mart3.to_csv(os.path.join(reports_dir, "travel_recommendations.csv"), index=False, encoding='utf-8')
    
    # Лог
    log_path = os.path.join(log_dir, "reports_log.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Обработано файлов: {len(enriched_files)} ({', '.join(enriched_files)})\n")
        log_file.write(f"Всего строк данных: {len(df_all)}\n")
        log_file.write(f"Дата загрузки: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write("Витрина 1: Рейтинг городов (city_tourism_rating.csv) - сортировка по avg_comfort_index\n")
        log_file.write("Витрина 2: Сводка по округам (federal_districts_summary.csv) - средняя temp, комфортные города\n")
        log_file.write("Витрина 3: Рекомендации (travel_recommendations.csv) - топ-3, дома, спец. советы, уведомления о погоде\n")
    
    print(f"Отчеты созданы в {reports_dir} на основе всех данных за период")
    print(f"Лог: {log_path}")

# Запуск
if __name__ == "__main__":
    create_reports()
