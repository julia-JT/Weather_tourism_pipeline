import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Папки
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'forecast')  # Исправлено: папка 'forecast' (без 's'), как в train_weather_model.py
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
    # Определяем дату прогноза (завтра)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
    forecast_path = os.path.join(forecasts_dir, f'forecast_{tomorrow}.csv')
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path, encoding='utf-8')
        # Переименуем колонки для совместимости (если нужно)
        if 'predicted_temperature' in df.columns:
            df = df.rename(columns={'predicted_temperature': 'predicted_temp'})  # Для удобства
        return df
    else:
        print(f"Файл прогноза {forecast_path} не найден. Пропускаем прогноз в графике.")
        return pd.DataFrame()

# Функция для генерации графиков
def generate_plots():
    # Загрузить данные
    df_hist = load_historical_data()
    df_forecast = load_forecast()
    
    if df_hist.empty:
        print("Нет исторических данных для визуализаций.")
        return
    
    # Группировка по дате (день) для упрощения графика
    df_hist['date'] = df_hist['as_of_date'].dt.date
    daily_comfort = df_hist.groupby('date')['comfort_index'].mean().reset_index()
    
    # График 1: Исторический comfort_index + прогноз температуры (поскольку comfort_index не прогнозируется напрямую)
    plt.figure(figsize=(10, 5))
    plt.plot(daily_comfort['date'], daily_comfort['comfort_index'], label='Исторический comfort_index', color='blue')
    
    # Добавить прогноз температуры на завтра (comfort_index рассчитать нельзя без влажности)
    if not df_forecast.empty:
        forecast_date = pd.to_datetime(df_forecast['forecast_date'].iloc[0]).date()
        forecast_temp = df_forecast['predicted_temp'].iloc[0]  # Теперь используем predicted_temperature
        # Для примера: отображаем прогноз температуры как точку (comfort_index не прогнозируем)
        plt.scatter(forecast_date, forecast_temp, color='red', label='Прогноз температуры на завтра', s=100)
        # Если хотите, можно добавить аннотацию: plt.annotate(f'Прогноз temp: {forecast_temp:.1f}°C', (forecast_date, forecast_temp))
    
    plt.xlabel('Дата')
    plt.ylabel('Comfort Index / Температура (°C)')
    plt.title('Comfort Index: Исторические данные и прогноз температуры')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(visualizations_dir, 'comfort_index.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"График сохранён в {plot_path}")
    
    # График 2: Температура с прогнозом
    daily_temp = df_hist.groupby('date')['temperature'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(daily_temp['date'], daily_temp['temperature'], label='Историческая температура', color='orange')
    
    # Добавить прогноз температуры
    if not df_forecast.empty:
        forecast_date = pd.to_datetime(df_forecast['forecast_date'].iloc[0]).date()
        forecast_temp = df_forecast['predicted_temp'].iloc[0]
        plt.scatter(forecast_date, forecast_temp, color='red', label='Прогноз на завтра', s=100)
    
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.title('Средняя температура по дням с прогнозом')
    plt.legend()
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
