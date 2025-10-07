import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Папки
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
enriched_dir = os.path.join(data_dir, 'enriched')
models_dir = os.path.join(data_dir, 'models')
forecasts_dir = os.path.join(models_dir, 'forecast')
visualizations_dir = os.path.join(data_dir, 'visualizations')
os.makedirs(forecasts_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для загрузки данных
def load_data_from_directory(directory):
    all_data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            # Извлекаем город из имени файла
            city_name = file.split('_')[0]
            df['city'] = city_name
            # Переименовываем колонку даты в 'date' (если есть 'Date', 'date' или 'collection_time')
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            elif 'date' not in df.columns and 'collection_time' in df.columns:
                # Используем collection_time как дату измерения
                df.rename(columns={'collection_time': 'date'}, inplace=True)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                print(f"Предупреждение: В файле {file} нет подходящей колонки даты ('date', 'Date' или 'collection_time'). Пропускаем файл.")
                continue  # Пропускаем файл, если нет даты
            all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Определяем интервалы: day (11:00-18:00), night (остальное)
        combined_df['hour'] = combined_df['date'].dt.hour
        combined_df['interval'] = combined_df['hour'].apply(lambda h: 'day' if 11 <= h < 18 else 'night')
        # Агрегируем по городу, дню и интервалу: средняя температура
        combined_df['date_day'] = combined_df['date'].dt.date
        combined_df = combined_df.groupby(['city', 'date_day', 'interval'])['temperature'].mean().reset_index()
        # Pivot: получаем temp_day и temp_night для каждого дня
        combined_df = combined_df.pivot(index=['city', 'date_day'], columns='interval', values='temperature').reset_index()
        combined_df.rename(columns={'day': 'temp_day', 'night': 'temp_night'}, inplace=True)
        combined_df['date'] = pd.to_datetime(combined_df['date_day'])
        combined_df.drop('date_day', axis=1, inplace=True)
        # Заполняем NaN (если интервал отсутствует) средними значениями или 0 (но лучше средними для точности)
        combined_df['temp_day'] = combined_df['temp_day'].fillna(combined_df['temp_day'].mean())
        combined_df['temp_night'] = combined_df['temp_night'].fillna(combined_df['temp_night'].mean())
        return combined_df
    return pd.DataFrame()

# Функция для вычисления comfort index (опционально, если есть humidity)
def calculate_comfort_index(temp, humidity):
    return temp - (humidity / 10)

# Функция для обучения модели и прогноза
def train_and_forecast(df, city, tomorrow_date):
    if df.empty:
        print(f"Нет данных для города {city}")
        return pd.DataFrame()
    
    df_city = df[df['city'] == city]
    if df_city.empty:
        print(f"Нет данных для города {city}")
        return pd.DataFrame()
    
    if 'date' not in df_city.columns or 'temp_day' not in df_city.columns or 'temp_night' not in df_city.columns:
        print(f"Необходимые колонки отсутствуют в данных для города {city}")
        return pd.DataFrame()
    
    df_city['date'] = pd.to_datetime(df_city['date'])
    df_city = df_city.sort_values('date')
    df_city['day_of_year'] = df_city['date'].dt.dayofyear
    X = df_city[['day_of_year']]
    
    if len(df_city) < 2:
        print(f"Недостаточно данных для модели в городе {city}")
        return pd.DataFrame()
    
    # Обучаем модель для temp_day
    y_day = df_city['temp_day']
    X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(X, y_day, test_size=0.2, random_state=42)
    model_day = LinearRegression()
    model_day.fit(X_train_day, y_train_day)
    
    # Обучаем модель для temp_night
    y_night = df_city['temp_night']
    X_train_night, X_test_night, y_train_night, y_test_night = train_test_split(X, y_night, test_size=0.2, random_state=42)
    model_night = LinearRegression()
    model_night.fit(X_train_night, y_train_night)
    
    # Прогноз на завтра
    tomorrow_day = pd.to_datetime(tomorrow_date).dayofyear
    predicted_day = model_day.predict([[tomorrow_day]])[0]
    predicted_night = model_night.predict([[tomorrow_day]])[0]
    
    # Сохранение моделей
    model_path_day = os.path.join(models_dir, f'{city}_model_day.pkl')
    model_path_night = os.path.join(models_dir, f'{city}_model_night.pkl')
    with open(model_path_day, 'wb') as f:
        pickle.dump(model_day, f)
    with open(model_path_night, 'wb') as f:
        pickle.dump(model_night, f)
    
    # Возврат прогноза как DataFrame
    forecast_df = pd.DataFrame({
        'city': [city],
        'forecast_date': [tomorrow_date],
        'predicted_temp_day': [predicted_day],
        'predicted_temp_night': [predicted_night],
        'model_type': ['LinearRegression']
    })
    return forecast_df

# Функция для создания визуализаций
def create_visualizations(df):
    if df.empty:
        print("Нет данных для визуализаций.")
        return
    
    if 'date' not in df.columns:
        print("Колонка 'date' отсутствует для визуализаций.")
        return
    
    # Визуализация дневной температуры по городам
    plt.figure(figsize=(10, 6))
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        plt.plot(city_data['date'], city_data['temp_day'], label=f"{city} - Day")
    plt.title('Day Temperature Over Time by City')
    plt.xlabel('Date')
    plt.ylabel('Day Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, 'temperature_day.png'))
    plt.close()
    
    # Визуализация ночной температуры по городам
    plt.figure(figsize=(10, 6))
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        plt.plot(city_data['date'], city_data['temp_night'], label=f"{city} - Night")
    plt.title('Night Temperature Over Time by City')
    plt.xlabel('Date')
    plt.ylabel('Night Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, 'temperature_night.png'))
    plt.close()
    
    # Визуализация comfort index (если есть humidity, но сейчас не актуально, так как фокус на temp_day/temp_night)
    # Можно добавить, если есть humidity в агрегированных данных (но сейчас её нет)
    print("Визуализации сохранены в data/visualizations/")

# Основная функция
def main():
    df = load_data_from_directory(enriched_dir)
    if df.empty:
        print("Нет данных для обработки.")
        return
    
    tomorrow = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    as_of_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_forecasts = []
    cities = df['city'].unique()
    for city in cities:
        forecast_df = train_and_forecast(df, city, tomorrow)
        if not forecast_df.empty:
            forecast_df['as_of_date'] = as_of_date
            all_forecasts.append(forecast_df)
    
    if all_forecasts:
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)
        forecast_file = os.path.join(forecasts_dir, 'Forecast.csv')
        file_exists = os.path.exists(forecast_file)
        combined_forecast.to_csv(forecast_file, mode='a', header=not file_exists, index=False)
        print(f"Прогнозы сохранены в {forecast_file} с as_of_date {as_of_date}")
    else:
        print("Нет прогнозов для сохранения.")
    
    # Создание визуализаций на основе всех данных
    create_visualizations(df)

if __name__ == "__main__":
    main()
