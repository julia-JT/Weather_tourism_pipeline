import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Папки
reports_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aggregated')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для фильтрации данных по as_of_date (например, последние 30 дней)
def filter_recent_data(df, days=30):
    df['as_of_date'] = pd.to_datetime(df['as_of_date'], errors='coerce')  # Преобразовать в datetime
    cutoff = datetime.now() - timedelta(days=days)
    return df[df['as_of_date'] >= cutoff]

# Основная функция
def generate_visualizations():
    # Витрина 1: Рейтинг городов
    city_rating_path = os.path.join(reports_dir, "city_tourism_rating.csv")
    if os.path.exists(city_rating_path):
        df_city = pd.read_csv(city_rating_path, encoding='utf-8')
        df_city = filter_recent_data(df_city, days=30)  # Фильтр по последним 30 дням (адаптируй)
        
        if not df_city.empty:
            # График: Топ-10 городов по avg_comfort_index
            top_10 = df_city.nlargest(10, 'avg_comfort_index')
            plt.figure(figsize=(10, 6))
            plt.barh(top_10['city_name'], top_10['avg_comfort_index'])
            plt.xlabel('Средний индекс комфорта')
            plt.title(f'Топ-10 городов по комфорту (данные на {df_city["as_of_date"].max().strftime("%Y-%m-%d")})')
            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, "top_cities_comfort.png"))
            plt.close()
            print("График top_cities_comfort.png создан")
    
    # Витрина 2: Сводка по округам
    district_summary_path = os.path.join(reports_dir, "federal_districts_summary.csv")
    if os.path.exists(district_summary_path):
        df_district = pd.read_csv(district_summary_path, encoding='utf-8')
        df_district = filter_recent_data(df_district, days=30)
        
        if not df_district.empty:
            # График: Средняя температура по округам
            plt.figure(figsize=(10, 6))
            plt.bar(df_district['federal_district'], df_district['avg_temperature'])
            plt.ylabel('Средняя температура (°C)')
            plt.title(f'Средняя температура по федеральным округам (данные на {df_district["as_of_date"].max().strftime("%Y-%m-%d")})')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, "district_temperature.png"))
            plt.close()
            print("График district_temperature.png создан")
    
    # Витрина 3: Рекомендации (здесь данные сводные, так что график может быть простым)
    travel_rec_path = os.path.join(reports_dir, "travel_recommendations.csv")
    if os.path.exists(travel_rec_path):
        df_rec = pd.read_csv(travel_rec_path, encoding='utf-8')
        df_rec = filter_recent_data(df_rec, days=30)
        
        if not df_rec.empty:
            # Простой текстовый вывод или график (например, количество городов в топ-3)
            top_3_count = len(df_rec['top_3_cities'].iloc[0].split(', ')) if not df_rec.empty else 0
            stay_home_count = len(df_rec['stay_home_cities'].iloc[0].split(', ')) if not df_rec.empty else 0
            
            labels = ['Топ-3 для поездок', 'Рекомендуется остаться дома']
            sizes = [top_3_count, stay_home_count]
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title(f'Распределение рекомендаций (данные на {df_rec["as_of_date"].max().strftime("%Y-%m-%d")})')
            plt.axis('equal')
            plt.savefig(os.path.join(visualizations_dir, "travel_recommendations_pie.png"))
            plt.close()
            print("График travel_recommendations_pie.png создан")
    
    print(f"Визуализации созданы в {visualizations_dir}")

# Запуск
if __name__ == "__main__":
    generate_visualizations()
