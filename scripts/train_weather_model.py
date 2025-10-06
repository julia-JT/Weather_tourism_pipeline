import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Папки (относительные пути от scripts/ к data/)
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
models_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'forecasts')

# Создаем папки
os.makedirs(models_dir, exist_ok=True)
os.makedirs(forecasts_dir, exist_ok=True)

# Основная функция обучения и прогноза
def train_and_forecast():
    # Найти все enriched CSV
    enriched_files = [f for f in os.listdir(enriched_dir) if f.startswith("weather_enriched_") and f.endswith(".csv")]
    if not enriched_files:
        print("ERROR: Нет enriched CSV файлов в data/enriched/")
        return
    
    # Объединить все файлы в один DataFrame
    dfs = []
    for file in enriched_files:
        path = os.path.join(enriched_dir, file)
        try:
            df = pd.read_csv(path, encoding='utf-8')
            dfs.append(df)
        except Exception as e:
            print(f"WARNING: Ошибка чтения {path}: {e}")
    
    if not dfs:
        print("ERROR: Нет валидных данных для обучения")
        return
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values(by=['city_name', 'date']).reset_index(drop=True)
    
    # Создать фичи и таргет
    combined_df['temperature_next_day'] = combined_df.groupby('city_name')['temperature'].shift(-1)  # Таргет: temp на следующий день
    # Фичи: текущие значения (предыдущий день для прогноза)
    features = ['temperature', 'pop', 'clouds', 'humidity', 'comfort_index']
    combined_df = combined_df.dropna(subset=['temperature_next_day'] + features)  # Убрать строки без таргета
    
    X = combined_df[features]
    y = combined_df['temperature_next_day']
    
    if X.empty or y.empty:
        print("ERROR: Недостаточно данных для обучения")
        return
    
    # Обучить модель
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Оценка качества (MSE на тренировочных данных)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    # Сохранить модель
    model_path = os.path.join(models_dir, 'weather_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Прогноз на завтра: Взять последний день для каждого города
    today = datetime.now().date()
    last_day_data = combined_df[combined_df['date'] == pd.to_datetime(today - timedelta(days=1))]  # Последний полный день (вчера)
    if last_day_data.empty:
        print("WARNING: Нет данных за последний день для прогноза")
        forecast_df = pd.DataFrame()
    else:
        forecast_X = last_day_data[features]
        forecast_temp = model.predict(forecast_X)
        forecast_df = last_day_data[['city_name']].copy()
        forecast_df['forecast_date'] = today + timedelta(days=1)  # Завтра
        forecast_df['predicted_temperature'] = forecast_temp.round(2)
    
    # Сохранить прогноз
    date_str = datetime.now().strftime("%Y%m%d")
    forecast_path = os.path.join(forecasts_dir, f"forecast_{date_str}.csv")
    forecast_df.to_csv(forecast_path, index=False, encoding='utf-8')
    
    # Лог
    log_path = os.path.join(models_dir, f"training_log_{date_str}.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Обучена модель на {len(combined_df)} записях из {len(enriched_files)} файлов\n")
        log_file.write(f"MSE на тренировочных данных: {mse:.4f}\n")
        log_file.write(f"Модель сохранена: {model_path}\n")
        log_file.write(f"Прогноз на завтра сохранен: {forecast_path}\n")
        if not forecast_df.empty:
            log_file.write(f"Прогноз для {len(forecast_df)} городов\n")
    
    print(f"Модель обучена, прогноз сохранен. Лог: {log_path}")

# Запуск
if __name__ == "__main__":
    train_and_forecast()
