import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Папки
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'forecast')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
os.makedirs(forecasts_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для загрузки исторических данных
def load_historical_data():
    if not os.path.exists(enriched_dir):
        print(f"Папка {enriched_dir} не найдена.")
        return pd.DataFrame()
    
    all_files = [f for f in os.listdir(enriched_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"Нет CSV-файлов в {enriched_dir}.")
        return pd.DataFrame()
    
    df_list = []
    for file in all_files:
        file_path = os.path.join(enriched_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            city_name = file.split('_')[0]
            df['city'] = city_name
            if 'collection_time' in df.columns:
                df['as_of_date'] = pd.to_datetime(df['collection_time'])
            df_list.append(df)
        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
            continue
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True).sort_values('as_of_date').drop_duplicates()
        return df_all
    else:
        return pd.DataFrame()

# Функция для загрузки прогнозов (из одного файла Forecast.csv)
def load_forecast():
    forecast_file = os.path.join(forecasts_dir, 'Forecast.csv')
    if not os.path.exists(forecast_file):
        print(f"Файл {forecast_file} не найден.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(forecast_file, encoding='utf-8')
        # Преобразуем as_of_date и forecast_date в datetime
        if 'as_of_date' in df.columns:
            df['as_of_date'] = pd.to_datetime(df['as_of_date'])
        if 'forecast_date' in df.columns:
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
        return df
    except Exception as e:
        print(f"Ошибка при загрузке {forecast_file}: {e}")
        return pd.DataFrame()

# Функция для генерации графиков
def generate_plots():
    # Загрузить данные
    df_hist = load_historical_data()
    df_forecast = load_forecast()
    
    if df_hist.empty:
        print("Нет исторических данных для визуализаций.")
        return
    
    # Группировка по дате и городу для исторических данных
    df_hist['date'] = df_hist['as_of_date'].dt.date
    daily_comfort = df_hist.groupby(['date', 'city'])['comfort_index'].mean().reset_index()
    daily_temp = df_hist.groupby(['date', 'city'])['temperature'].mean().reset_index()
    
    # Фильтруем прогнозы по последнему as_of_date (самому свежему)
    if not df_forecast.empty and 'as_of_date' in df_forecast.columns:
        latest_as_of_date = df_forecast['as_of_date'].max()
        df_forecast = df_forecast[df_forecast['as_of_date'] == latest_as_of_date]
    
    # Рассчитаем ошибки прогноза: для каждого прогноза найти реальную температуру на forecast_date
    forecast_errors = []
    if not df_forecast.empty and 'forecast_date' in df_forecast.columns and 'predicted_temp' in df_forecast.columns:
        for _, forecast_row in df_forecast.iterrows():
            city = forecast_row['city']
            forecast_date = forecast_row['forecast_date'].date()
            predicted_temp = forecast_row['predicted_temp']
            
            # Найти реальную среднюю температуру на эту дату из исторических данных
            real_temp_row = daily_temp[(daily_temp['city'] == city) & (daily_temp['date'] == forecast_date)]
            if not real_temp_row.empty:
                real_temp = real_temp_row['temperature'].iloc[0]
                error = predicted_temp - real_temp
                forecast_errors.append({
                    'city': city,
                    'forecast_date': forecast_date,
                    'predicted_temp': predicted_temp,
                    'real_temp': real_temp,
                    'error': error
                })
    
    df_errors = pd.DataFrame(forecast_errors)
    
    # График 1: Comfort Index с прогнозами температуры по городам (добавляем реальные точки)
    plt.figure(figsize=(12, 6))
    for city in daily_comfort['city'].unique():
        city_data = daily_comfort[daily_comfort['city'] == city]
        plt.plot(city_data['date'], city_data['comfort_index'], label=f'Исторический comfort_index ({city})')
        
        # Добавить прогноз температуры для этого города
        if not df_forecast.empty and 'forecast_date' in df_forecast.columns and 'predicted_temp' in df_forecast.columns:
            city_forecast = df_forecast[df_forecast['city'] == city]
            if not city_forecast.empty:
                forecast_date = city_forecast['forecast_date'].iloc[0].date()
                forecast_temp = city_forecast['predicted_temp'].iloc[0]
                plt.scatter(forecast_date, forecast_temp, label=f'Прогноз temp ({city})', s=100, marker='o', color='red')
                
                # Добавить реальную температуру на forecast_date, если есть
                real_temp_row = daily_temp[(daily_temp['city'] == city) & (daily_temp['date'] == forecast_date)]
                if not real_temp_row.empty:
                    real_temp = real_temp_row['temperature'].iloc[0]
                    plt.scatter(forecast_date, real_temp, label=f'Реальная temp ({city})', s=100, marker='x', color='blue')
    
    plt.xlabel('Дата')
    plt.ylabel('Comfort Index / Температура (°C)')
    plt.title('Comfort Index и прогноз температуры по городам с реальными данными')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(visualizations_dir, 'comfort_index_by_city.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"График comfort index сохранён в {plot_path}")
    
    # График 2: Температура с прогнозами по городам (добавляем реальные точки)
    plt.figure(figsize=(12, 6))
    for city in daily_temp['city'].unique():
        city_data = daily_temp[daily_temp['city'] == city]
        plt.plot(city_data['date'], city_data['temperature'], label=f'Историческая температура ({city})')
        
        # Добавить прогноз температуры для этого города
        if not df_forecast.empty and 'forecast_date' in df_forecast.columns and 'predicted_temp' in df_forecast.columns:
            city_forecast = df_forecast[df_forecast['city'] == city]
            if not city_forecast.empty:
                forecast_date = city_forecast['forecast_date'].iloc[0].date()
                forecast_temp = city_forecast['predicted_temp'].iloc[0]
                plt.scatter(forecast_date, forecast_temp, label=f'Прогноз ({city})', s=100, marker='o', color='red')
                
                # Добавить реальную температуру на forecast_date, если есть
                real_temp_row = daily_temp[(daily_temp['city'] == city) & (daily_temp['date'] == forecast_date)]
                if not real_temp_row.empty:
                    real_temp = real_temp_row['temperature'].iloc[0]
                    plt.scatter(forecast_date, real_temp, label=f'Реальная ({city})', s=100, marker='x', color='blue')
    
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.title('Средняя температура по дням и городам с прогнозами и реальными данными')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    temp_plot_path = os.path.join(visualizations_dir, 'temperature_by_city.png')
    plt.savefig(temp_plot_path)
    plt.close()
    print(f"График температуры сохранён в {temp_plot_path}")
    
    # График 3: Ошибки прогноза (новый)
    if not df_errors.empty:
        plt.figure(figsize=(12, 6))
        for city in df_errors['city'].unique():
            city_errors = df_errors[df_errors['city'] == city]
            plt.scatter(city_errors['forecast_date'], city_errors['error'], label=f'Ошибка прогноза ({city})', s=100)
        
        plt.axhline(y=0, color='black', linestyle='--', label='Идеальный прогноз (ошибка=0)')
        plt.xlabel('Дата прогноза')
        plt.ylabel('Ошибка (°C) (Прогноз - Реальность)')
        plt.title('Ошибки прогноза температуры по городам')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        error_plot_path = os.path.join(visualizations_dir, 'forecast_errors.png')
        plt.savefig(error_plot_path)
        plt.close()
        print(f"График ошибок прогноза сохранён в {error_plot_path}")
    else:
        print("Нет данных для графика ошибок прогноза (возможно, нет совпадений дат).")

# Запуск
if __name__ == "__main__":
    generate_plots()
