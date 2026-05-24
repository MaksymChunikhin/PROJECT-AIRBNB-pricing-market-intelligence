"""
utils/cleaning.py
=================
Функции для загрузки и очистки сырых данных Airbnb.
Вся логика соответствует 01_eda.ipynb (cleaning-секции).

Использование:
    from utils.cleaning import load_all

    data = load_all()
    listings = data["listings"]
    calendar = data["calendar"]
    reviews  = data["reviews"]
    geo      = data["geo"]
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import geopandas as gpd

# Координаты центра Амстердама
CITY_CENTER_LAT: float = 52.3676
CITY_CENTER_LON: float = 4.9041

# Путь к сырым данным по умолчанию (относительно корня проекта)
# При вызове из notebooks/ нужен "../data/raw"
_DEFAULT_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: float, lon2: float) -> np.ndarray:
    """Векторизованная haversine-формула. Возвращает расстояние в км."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _parse_percent(series: pd.Series) -> pd.Series:
    """'95%' → 0.95. Используется для host_response_rate / host_acceptance_rate."""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        / 100
    )


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def load_clean_listings(path: str | None = None) -> pd.DataFrame:
    """
    Загружает и очищает listings.csv.gz.

    Выполняемые шаги (в точности как в 01_eda.ipynb):
    1.  Копия сырых данных
    2.  Удаление полностью пустых колонок
    3.  Парсинг price ($-строка → float)
    4.  Конвертация дат в datetime
    5.  Фильтр выбросов (price, min/max_nights)
    6.  Производные признаки:
        - log_price
        - distance_to_center
        - unavailability_rate
        - host_experience_years
        - professional_host (>= 3 listings)

    Возвращает pd.DataFrame.
    """
    if path is None:
        path = os.path.join(_DEFAULT_RAW, "listings.csv.gz")

    df = pd.read_csv(path)

    # 1. Удаляем полностью пустые колонки
    df.drop(
        columns=["neighbourhood_group_cleansed", "calendar_updated"],
        inplace=True,
        errors="ignore",
    )

    # 2. Price: '$1,234.00' → float
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # 3. Даты → datetime
    date_cols = [
        "last_scraped", "host_since", "calendar_last_scraped",
        "first_review", "last_review",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Фильтр выбросов
    if df["price"].notna().any():
        price_cap = df["price"].quantile(0.995)
    else:
        price_cap = 1500.0

    df = df[
        df["price"].isna() | (df["price"] <= price_cap)
    ].copy()
    df = df[df["maximum_nights"] <= 1125].copy()
    df = df[df["minimum_nights"] <= 365].copy()

    # 5. log_price (target для ML)
    df["log_price"] = np.log1p(df["price"])

    # 6. Расстояние до центра города (км)
    df["distance_to_center"] = _haversine_km(
        df["latitude"].values,
        df["longitude"].values,
        CITY_CENTER_LAT,
        CITY_CENTER_LON,
    )

    # 7. Unavailability rate (прокси спроса, НЕ настоящий occupancy)
    #    0 = всегда доступен, 1 = всегда недоступен
    df["unavailability_rate"] = (365 - df["availability_365"]) / 365

    # 8. Опыт хоста (в годах), привязанный к дате last_scraped (воспроизводимо)
    if "last_scraped" in df.columns and "host_since" in df.columns:
        reference_date = df["last_scraped"].max()
        df["host_experience_years"] = (
            (reference_date - df["host_since"]).dt.days / 365.25
        )

    # 9. Флаг профессионального хоста (≥ 3 listings)
    if "host_listings_count" in df.columns:
        df["professional_host"] = df["host_listings_count"] >= 3

    return df


def load_clean_calendar(path: str | None = None) -> pd.DataFrame:
    """
    Загружает и очищает calendar.csv.gz.

    Шаги:
    1. Удаление пустых price/adjusted_price
    2. date → datetime
    3. available: 't'/'f' → 1/0
    4. Добавление временных признаков: day_of_week, is_weekend, month, year

    Возвращает pd.DataFrame.

    ⚠️  Важно: calendar.price и calendar.adjusted_price полностью пусты
        в текущем Inside Airbnb snapshot (нет данных о ценах по дням).
    """
    if path is None:
        path = os.path.join(_DEFAULT_RAW, "calendar.csv.gz")

    df = pd.read_csv(path)

    # 1. Удаляем пустые ценовые колонки
    df.drop(columns=["price", "adjusted_price"], inplace=True, errors="ignore")

    # 2. Дата
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. available: строка → 0/1
    df["available"] = df["available"].map({"t": 1, "f": 0})

    # 4. Временные признаки
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"]  = df["day_of_week"] >= 5
    df["month"]       = df["date"].dt.month
    df["year"]        = df["date"].dt.year

    return df


def load_clean_reviews(path: str | None = None) -> pd.DataFrame:
    """
    Загружает и очищает reviews.csv.gz.

    Шаги:
    1. Удаление пустых/пустострочных comments
    2. date → datetime

    Возвращает pd.DataFrame.
    """
    if path is None:
        path = os.path.join(_DEFAULT_RAW, "reviews.csv.gz")

    df = pd.read_csv(path)

    # date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Удаляем пустые и пустострочные комментарии
    df = df.dropna(subset=["comments"])
    df = df[df["comments"].str.strip() != ""].copy()

    return df


def load_geo(path: str | None = None) -> gpd.GeoDataFrame:
    """
    Загружает и очищает neighbourhoods.geojson.

    Шаги:
    1. Удаление пустой колонки neighbourhood_group
    2. Проверка корректности геометрий (валидны ли полигоны)

    Возвращает GeoDataFrame.
    """
    if path is None:
        path = os.path.join(_DEFAULT_RAW, "neighbourhoods.geojson")

    gdf = gpd.read_file(path)

    # Удаляем пустую колонку
    gdf.drop(columns=["neighbourhood_group"], inplace=True, errors="ignore")

    # Проверяем геометрии (silent — невалидные не удаляем, просто оставляем флаг)
    invalid = (~gdf.geometry.is_valid).sum()
    if invalid > 0:
        print(f"[load_geo] ⚠️  {invalid} невалидных геометрий")

    return gdf


def load_all(
    data_dir: str | None = None,
    with_calendar_unavail: bool = True,
) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Загружает и очищает все 4 датасета, объединяет в словарь.

    Параметры
    ----------
    data_dir : str, optional
        Путь к папке с raw-данными. По умолчанию — ../data/raw/
        относительно корня проекта.
    with_calendar_unavail : bool, default True
        Если True — добавляет calendar_unavail_rate в listings
        (агрегирует из calendar per listing_id).

    Возвращает
    ----------
    dict с ключами: "listings", "calendar", "reviews", "geo"

    Пример
    ------
    >>> from utils.cleaning import load_all
    >>> data = load_all()
    >>> listings = data["listings"]
    >>> calendar = data["calendar"]
    """
    if data_dir is None:
        data_dir = _DEFAULT_RAW

    listings = load_clean_listings(os.path.join(data_dir, "listings.csv.gz"))
    calendar = load_clean_calendar(os.path.join(data_dir, "calendar.csv.gz"))
    reviews  = load_clean_reviews(os.path.join(data_dir, "reviews.csv.gz"))
    geo      = load_geo(os.path.join(data_dir, "neighbourhoods.geojson"))

    # Calendar-based unavailability per listing
    # (более гранулярный прокси спроса, чем availability_365)
    if with_calendar_unavail:
        calendar_unavail = (
            calendar.groupby("listing_id")["available"]
            .apply(lambda x: 1 - x.mean())
            .rename("calendar_unavail_rate")
        )
        listings = listings.merge(
            calendar_unavail, left_on="id", right_index=True, how="left"
        )

    return {
        "listings": listings,
        "calendar": calendar,
        "reviews":  reviews,
        "geo":      geo,
    }
