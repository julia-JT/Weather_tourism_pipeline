import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import pickle

# Директория скрипта (где лежит train_weather_model.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Директории (теперь собираются относительно SCRIPT_DIR)
ENRICHED_DIR = os.path.join(SCRIPT_DIR, "..", "data", "enriched")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "models", "forecast")
VISUALIZATIONS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "visualizations")

# Порог для переобучения (MAE > этого значения -> переобучить)
MAE_THRESHOLD = 5.0  # В градусах Цельсия

def create_directories():
    """Создаём все необходимые директории."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    print("Директории созданы или уже существуют.")

def load_all_enriched_data(directory, prefix="weather_enriched_"):
    """Загружаем и объединяем все файлы weather_enriched_*.csv."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No files found matching {prefix}*.csv in {directory}")
    
    all_data = []
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        # Добавляем дату из имени файла для разделения
        date_str = file.split('_')[-1].split('.')[0]  # YYYYMMDD
        df['file_date'] = date_str
        all_data.append(df)
    
    # Объединяем и удаляем дубликаты (по timestamp и city_name)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'city_name'])
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.sort_values(by=['city_name', 'timestamp'])
    
    return combined_df

def split_data_by_date(df, today_str):
    """Разделяем данные на исторические и сегодняшние."""
    historical = df[df['file_date'] < today_str]
    today_data = df[df['file_date'] == today_str]
    return historical, today_data

def load_and_prepare_data(df):
    """Подготавливаем данные для Prophet (фильтр столбцов)."""
    df = df[['timestamp', 'city_name', 'temperature', 'humidity']].dropna()
    # Рассчитываем comfort_index (упрощённая формула: temperature - humidity/10)
    df['comfort_index'] = df['temperature'] - (df['humidity'] / 10)
    return df

def get_model_path(city_name):
    """Путь к файлу модели для города."""
    return os.path.join(MODELS_DIR, f"{city_name}_model.pkl")

def save_model(model, city_name):
    """Сохраняем модель Prophet."""
    model_path = get_model_path(city_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель для {city_name} сохранена в {model_path}")

def load_model(city_name):
    """Загружаем модель Prophet, если существует."""
    model_path = get_model_path(city_name)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Модель для {city_name} загружена из {model_path}")
        return model
    return None

def forecast_temperature_for_city(city_data, forecast_date, retrain=False):
    """Предсказываем температуру на указанную дату для одного города. Загружаем/сохраняем модель."""
    city_name = city_data['city_name'].iloc[0]
    df_prophet = city_data[['timestamp', 'temperature']].rename(columns={'timestamp': 'ds', 'temperature': 'y'})
    
    model = load_model(city_name)
    if model is None or retrain:
        print(f"Обучение модели для {city_name}...")
        model = Prophet(yearly_seasonality=True, daily_seasonality=True, weekly_seasonality=True)
        model.fit(df_prophet)
        save_model(model, city_name)
    
    future = model.make_future_dataframe(periods=1, freq='D')  # Прогноз на 1 день
    future = future[future['ds'] == pd.to_datetime(forecast_date)]
    forecast = model.predict(future)
    
    if not forecast.empty:
        return forecast['yhat'].iloc[0]
    return None

def evaluate_model_on_today(historical_df, today_df, forecast_date_str):
    """Оцениваем модель на сегодняшних данных: прогноз на forecast_date_str и сравнение с today_df."""
    city_groups_hist = historical_df.groupby('city_name')
    city_groups_today = today_df.groupby('city_name')
    
    mae_list = []
    for city_name in city_groups_hist.groups.keys():
        if city_name not in city_groups_today.groups:
            continue  # Нет сегодняшних данных для города
        
        hist_data = city_groups_hist.get_group(city_name)
        today_data = city_groups_today.get_group(city_name)
        
        # Прогноз на сегодняшнюю дату (используем retrain=False, чтобы загрузить существующую модель)
        predicted_temp = forecast_temperature_for_city(hist_data, forecast_date_str, retrain=False)
        if predicted_temp is None:
            continue
        
        # Реальная температура на сегодня (средняя по дню, если несколько записей)
        real_temp = today_data['temperature'].mean()
        
        mae = abs(predicted_temp - real_temp)
        mae_list.append(mae)
        print(f"MAE для {city_name}: {mae:.2f}°C (предсказ: {predicted_temp:.2f}, реал: {real_temp:.2f})")
    
    if mae_list:
        avg_mae = np.mean(mae_list)
        print(f"Средний MAE по городам: {avg_mae:.2f}°C")
        return avg_mae > MAE_THRESHOLD
    return False  # Если нет данных для оценки, не переобучаем

def generate_visualizations(df, forecast_df, forecast_date_str):
    """Генерируем и сохраняем визуализации."""
    # График температуры по городам (история + прогноз)
    plt.figure(figsize=(12, 8))
    for city in df['city_name'].unique():
        city_data = df[df['city_name'] == city]
        plt.plot(city_data['timestamp'], city_data['temperature'], label=f'{city} (история)')
    
    # Добавляем прогнозы
    for _, row in forecast_df.iterrows():
        plt.scatter(pd.to_datetime(row['forecast_date']), row['predicted_temperature'], label=f"{row['city_name']} (прогноз)", marker='x', s=100)
    
    plt.title(f'Прогноз температуры на {forecast_date_str}')
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'temperature_forecast.png'))
    plt.close()
    print("Визуализация температуры сохранена.")
    
    # График comfort index по городам
    plt.figure(figsize=(12, 8))
    for city in df['city_name'].unique():
        city_data = df[df['city_name'] == city]
        plt.plot(city_data['timestamp'], city_data['comfort_index'], label=city)
    
    plt.title('Индекс комфорта по городам')
    plt.xlabel('Дата')
    plt.ylabel('Индекс комфорта')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'comfort_index.png'))
    plt.close()
    print("Визуализация comfort index сохранена.")

def train_and_forecast(historical_df, today_df, forecast_date_str, retrain=False):
    """Основная функция: обучаем/переобучаем и предсказываем на завтра."""
    if retrain:
        # Переобучение на всех данных
        full_df = pd.concat([historical_df, today_df], ignore_index=True)
        print("Переобучение на всех данных...")
    else:
        full_df = historical_df
    
    df = load_and_prepare_data(full_df)
    city_groups = df.groupby('city_name')
    
    results = []
    for city_name, city_data in city_groups:
        if len(city_data) < 2:
            print(f"Недостаточно данных для {city_name}. Пропускаем.")
            continue
        
        try:
            # Прогноз на завтра
            tomorrow_temp = forecast_temperature_for_city(city_data, forecast_date_str, retrain=retrain)
            if tomorrow_temp is not None:
                # Рассчитываем comfort_index для прогноза (на основе средней влажности)
                avg_humidity = city_data['humidity'].mean()
                comfort_index = tomorrow_temp - (avg_humidity / 10)
                results.append({
                    'city_name': city_name,
                    'forecast_date': forecast_date_str,
                    'predicted_temperature': round(tomorrow_temp, 2),
                    'predicted_comfort_index': round(comfort_index, 2)
                })
                print(f"Прогноз для {city_name}: {tomorrow_temp:.2f}°C, comfort: {comfort_index:.2f} на {forecast_date_str}")
        except Exception as e:
            print(f"Ошибка для {city_name}: {e}")
    
    # Сохраняем результаты
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(OUTPUT_DIR, f"forecast_{forecast_date_str}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Прогнозы сохранены в {output_file}")
        
        # Генерируем визуализации
        generate_visualizations(df, results_df, forecast_date_str)
    else:
        print("Нет прогнозов для сохранения.")

# Запуск
if __name__ == "__main__":
    try:
        create_directories()
        
        # Получаем сегодняшнюю дату
        today = datetime.now()
        today_str = today.strftime("%Y%m%d")
        forecast_date = today + timedelta(days=1)
        forecast_date_str = forecast_date.strftime("%Y%m%d")
        
        # Загружаем все данные
        all_df = load_all_enriched_data(ENRICHED_DIR)
        print(f"Загружено {len(all_df)} записей из всех файлов.")
        
        # Разделяем на исторические и сегодняшние
        historical_df, today_df = split_data_by_date(all_df, today_str)
        print(f"Исторических записей: {len(historical_df)}, сегодняшних: {len(today_df)}")
        
        # Оцениваем модель на сегодняшних данных
        needs_retrain = evaluate_model_on_today(historical_df, today_df, today_str)
        
        # Обучаем/переобучаем и предсказываем
        train_and_forecast(historical_df, today_df, forecast_date_str, retrain=needs_retrain)
        
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
