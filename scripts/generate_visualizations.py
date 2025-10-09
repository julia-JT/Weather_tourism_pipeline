import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Папки
aggregated_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aggregated')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')  # Путь к README.md
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для загрузки данных из aggregated слоя
def load_aggregated_data(filename):
    file_path = os.path.join(aggregated_dir, filename)
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # Преобразуем as_of_date в datetime, если есть
        if 'as_of_date' in df.columns:
            df['as_of_date'] = pd.to_datetime(df['as_of_date'], errors='coerce')
        return df
    except Exception as e:
        print(f"Ошибка при загрузке {file_path}: {e}")
        return pd.DataFrame()

# Функция для генерации графика динамики avg_comfort_index из city_tourism_rating.csv
def generate_comfort_index_trend(df):
    if df.empty or 'as_of_date' not in df.columns or 'avg_comfort_index' not in df.columns:
        print("Нет данных для графика динамики avg_comfort_index.")
        return "", ""
    
    # Группировка по as_of_date (предполагаем, что данные уже агрегированы по городам, но если нужно, добавьте группировку)
    df_trend = df.groupby('as_of_date')['avg_comfort_index'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_trend['as_of_date'], df_trend['avg_comfort_index'], marker='o', label='Средний Comfort Index')
    plt.xlabel('Дата')
    plt.ylabel('Средний Comfort Index')
    plt.title('Динамика изменений avg_comfort_index от as_of_date')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(visualizations_dir, 'comfort_index_trend.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"График динамики avg_comfort_index сохранён в {plot_path}")
    
    # Таблица на последний as_of_date
    latest_date = df['as_of_date'].max()
    df_latest = df[df['as_of_date'] == latest_date]
    table_md = df_latest.to_markdown(index=False) if not df_latest.empty else "Нет данных на последнюю дату."
    
    return plot_path, table_md

# Функция для генерации гистограммы из federal_districts_summary.csv
def generate_district_histogram(df):
    if df.empty or 'as_of_date' not in df.columns:
        print("Нет данных для гистограммы federal_districts_summary.")
        return "", ""
    
    # Фильтр на максимальную as_of_date
    latest_date = df['as_of_date'].max()
    df_latest = df[df['as_of_date'] == latest_date]
    
    if df_latest.empty or 'avg_temperature' not in df_latest.columns or 'comfortable_cities' not in df_latest.columns:
        print("Нет столбцов avg_temperature или comfortable_cities.")
        return "", ""
    
    # Гистограмма: столбцы для avg_temperature и comfortable_cities (предполагаем, что comfortable_cities - числовой, например, count)
    plt.figure(figsize=(12, 6))
    plt.bar(df_latest.index, df_latest['avg_temperature'], label='Средняя температура', alpha=0.7)
    plt.bar(df_latest.index, df_latest['comfortable_cities'], label='Комфортные города', alpha=0.7)
    plt.xlabel('Записи')
    plt.ylabel('Значения')
    plt.title(f'Гистограмма avg_temperature и comfortable_cities на {latest_date.date()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(visualizations_dir, 'district_histogram.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Гистограмма сохранена в {plot_path}")
    
    return plot_path, ""

# Функция для вывода значений из travel_recommendations.csv
def generate_recommendations_text(df):
    if df.empty or 'as_of_date' not in df.columns:
        print("Нет данных для travel_recommendations.")
        return ""
    
    # Фильтр на максимальную as_of_date
    latest_date = df['as_of_date'].max()
    df_latest = df[df['as_of_date'] == latest_date]
    
    if df_latest.empty:
        return "Нет данных на максимальную дату."
    
    # Вывести значения столбцов в отдельной строке (предполагаем, что одна строка, или агрегировать)
    row = df_latest.iloc[0] if not df_latest.empty else {}
    text = f"Рекомендации на {latest_date.date()}: " + ", ".join([f"{col}: {row[col]}" for col in df_latest.columns if col != 'as_of_date'])
    return text

# Основная функция для генерации визуализаций и обновления README.md
def generate_visualizations():
    readme_content = "# Визуализации данных\n\n"
    
    # 1. city_tourism_rating.csv
    df_rating = load_aggregated_data('city_tourism_rating.csv')
    plot_path, table_md = generate_comfort_index_trend(df_rating)
    if plot_path:
        readme_content += f"## Динамика avg_comfort_index\n\n![График динамики]({plot_path})\n\n"
    if table_md:
        readme_content += f"### Сводная таблица на последний as_of_date\n\n{table_md}\n\n"
    
    # 2. federal_districts_summary.csv
    df_district = load_aggregated_data('federal_districts_summary.csv')
    plot_path, _ = generate_district_histogram(df_district)
    if plot_path:
        readme_content += f"## Гистограмма по федеральным округам\n\n![Гистограмма]({plot_path})\n\n"
    
    # 3. travel_recommendations.csv
    df_recommend = load_aggregated_data('travel_recommendations.csv')
    text = generate_recommendations_text(df_recommend)
    if text:
        readme_content += f"## Рекомендации для путешествий\n\n{text}\n\n"
    
    # Запись в README.md
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"README.md обновлён с визуализациями в {readme_path}")

# Запуск
if __name__ == "__main__":
    generate_visualizations()
