import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Папки
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'forecasts')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для загрузки исторических данных
def load_historical_data():
    all_files = [f for f in os.listdir(enriched_dir) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        file_path = os.path.join(enriched_dir, file)
        df = pd.read_csv(file_path, encoding='utf-8')
        if 'collection_time' in df.columns:
            df['as_of_date'] = pd.to_datetime(df['collection_time'])
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True).sort_values('as_of_date').drop_duplicates()
    return df_all

# Функция для загрузки прогноза
def load_forecast():
    forecast_path = os.path.join(forecasts_dir, 'forecast_tomorrow.csv')
    if os.path.exists(forecast_path):
        return pd.read_csv(forecast_path, encoding='utf-8')
    else:
        print("Файл прогноза не найден. Пропускаем прогноз в графике.")
        return pd.DataFrame()

# Функция для генерации графиков
def generate_plots():
    # Загрузить данные
    df_hist = load_historical_data()
    df_forecast = load_forecast()
    
    if df_hist.empty:
        print("Нет исторических данных для визуализаций.")
        return
    
    # Предполагаем, что данные по городам; для примера берем один город или усредняем
    # Группировка по дате (день) для упрощения графика
    df_hist['date'] = df_hist['as_of_date'].dt.date
    daily_comfort = df_hist.groupby('date')['comfort_index'].mean().reset_index()
    
    # График 1: Исторический comfort_index
    plt.figure(figsize=(10, 5))
    plt.plot(daily_comfort['date'], daily_comfort['comfort_index'], label='Исторический comfort_index', color='blue')
    
    # Добавить прогноз на завтра, если есть
    if not df_forecast.empty:
        forecast_date = pd.to_datetime(df_forecast['forecast_date'].iloc[0]).date()
        forecast_value = df_forecast['predicted_comfort_index'].iloc[0]
        plt.scatter(forecast_date, forecast_value, color='red', label='Прогноз на завтра', s=100)
    
    plt.xlabel('Дата')
    plt.ylabel('Comfort Index')
    plt.title('Comfort Index: Исторические данные и прогноз')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(visualizations_dir, 'comfort_index.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"График сохранён в {plot_path}")
    
    # Можно добавить больше графиков, например, для температуры и т.д.
    # Пример: График температуры
    daily_temp = df_hist.groupby('date')['temperature'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(daily_temp['date'], daily_temp['temperature'], label='Средняя температура', color='orange')
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.title('Средняя температура по дням')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    temp_plot_path = os.path.join(visualizations_dir, 'temperature.png')
    plt.savefig(temp_plot_path)
    plt.close()
    print(f"График температуры сохранён в {temp_plot_path}")

# Запуск
if __name__ == "__main__":
    generate_plots()
