from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import requests
import os

app = FastAPI(title="GitHub Data Marts API")

# Настройки GitHub
GITHUB_OWNER = "julia-JT"  # Ваш GitHub username/organization
GITHUB_REPO = "Weather_tourism_pipeline"    # Имя репо
GITHUB_BRANCH = "main"             # Ветка
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Токен для приватных репо (или None для публичных)

# Предопределенные витрины (список имен без .csv)
marts = 'data/aggregated'  # Базовая папка для витрин (можно использовать для путей)
mart_name_list = ['city_tourism_rating', 'federal_districts_summary', 'travel_recommendations']

def get_csv_from_github(file_path: str) -> pd.DataFrame:
    """Скачивает и парсит CSV из GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{file_path}?ref={GITHUB_BRANCH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Файл не найден или нет доступа")
    
    # Получить raw URL и скачать
    raw_url = response.json()["download_url"]
    raw_response = requests.get(raw_url, headers=headers)
    raw_response.raise_for_status()
    
    # Парсить CSV
    df = pd.read_csv(pd.io.common.StringIO(raw_response.text))
    return df

@app.get("/marts")
def list_marts():
    """Список предопределенных витрин."""
    return {"marts": mart_name_list}

@app.get("/marts/{mart_name}")
def get_mart(mart_name: str, limit: int = Query(10, ge=1, le=1000)):
    """Получить данные витрины (первые limit строк)."""
    if mart_name not in mart_name_list:
        raise HTTPException(status_code=404, detail="Витрина не найдена в списке")
    
    try:
        # Используем переменную marts для формирования пути
        df = get_csv_from_github(f"{marts}/{mart_name}.csv")
        return df.head(limit).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
