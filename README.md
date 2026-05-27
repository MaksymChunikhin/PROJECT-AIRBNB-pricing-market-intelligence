# Airbnb Smart Pricing & Market Intelligence System

ML-проект для анализа рынка Airbnb в Амстердаме: прогнозирование цены, market intelligence, сегментация хостов и recommendation system.

Проект построен как modular ML pipeline — от raw данных Inside Airbnb до feature engineering, gradient boosting моделей, explainable AI, кластеризации, recommendation system и подготовки к deployment.

**Dataset:** Inside Airbnb — Amsterdam  
**Snapshot:** 11 September 2025  
**Автор:** Максим Чунихин

---

# Структура проекта

```text
airbnb_project/
├── notebooks/
│   ├── 01_eda.ipynb
│   │   # Data understanding, cleaning, EDA, geo analysis
│   │
│   ├── 02_feature_engineering.ipynb
│   │   # 76 ML features, amenities parsing,
│   │   # NLP features, BERT sentiment,
│   │   # MiniLM embeddings, dataset preparation
│   │
│   ├── 03_pricing_model.ipynb
│   │   # CatBoost / LightGBM / XGBoost,
│   │   # Optuna tuning, ensemble,
│   │   # SHAP explainability
│   │
│   ├── 04_market_intelligence.ipynb
│   │   # Undervalued listings detection,
│   │   # fairness score,
│   │   # host segmentation,
│   │   # KMeans + HDBSCAN
│   │
│   └── 05_recommender.ipynb
│       # Content-based recommender,
│       # cosine similarity,
│       # embeddings + amenities + geo features
│
├── utils/
│   └── cleaning.py
│
├── data/
│   ├── raw/          # Исходные файлы Inside Airbnb (не хранятся в git)
│   └── processed/    # Готовые parquet datasets (не хранятся в git)
│
├── models/           # Обученные модели и артефакты
│
└── app/
    └── streamlit_app.py   # Streamlit dashboard 
```

---

# Обзор проекта

Проект состоит из пяти основных этапов:

## 1. Exploratory Data Analysis

Notebook: `01_eda.ipynb`

Основные задачи:
- понимание данных;
- cleaning и preprocessing;
- анализ цен;
- анализ районов;
- demand analysis;
- host analysis;
- spatial analysis;
- поиск leakage-признаков.

---

## 2. Feature Engineering

Notebook: `02_feature_engineering.ipynb`

Основные задачи:
- создание ML-признаков;
- parsing amenities;
- geo и demand features;
- NLP features из descriptions и reviews;
- multilingual BERT sentiment;
- MiniLM embeddings;
- подготовка train / validation / test datasets.

Результат:
готовые datasets для ML-моделей и recommendation system.

---

## 3. Pricing ML Models

Notebook: `03_pricing_model.ipynb`

Основные задачи:
- baseline regression models;
- CatBoost / LightGBM / XGBoost;
- Optuna hyperparameter tuning;
- ensemble blending;
- SHAP explainability;
- residual analysis;
- selection bias analysis.

Цель:
прогнозирование цены Airbnb listing по характеристикам жилья.

---

## 4. Market Intelligence

Notebook: `04_market_intelligence.ipynb`

Основные задачи:
- поиск undervalued / overpriced listings;
- fairness score;
- revenue potential estimation;
- сегментация хостов;
- KMeans и HDBSCAN clustering.

Цель:
получение бизнес-аналитики на основе ML-модели и поведения хостов.

---

## 5. Recommendation System

Notebook: `05_recommender.ipynb`

Основные задачи:
- content-based recommendation pipeline;
- vector representation listings;
- embeddings + amenities + geo features;
- cosine similarity + nearest neighbors.

Цель:
поиск похожих Airbnb-объектов по характеристикам жилья.

---

# Текущий статус

| Notebook | Описание | Статус |
|---|---|:---:|
| `01_eda.ipynb` | EDA + cleaning + geo analysis | ✅ |
| `02_feature_engineering.ipynb` | ML features + NLP + embeddings | ✅ |
| `03_pricing_model.ipynb` | GBM models + Optuna + SHAP | ✅ |
| `04_market_intelligence.ipynb` | Market intelligence + clustering | ✅ |
| `05_recommender.ipynb` | Recommendation system | ✅ |
| `streamlit_app.py` | Streamlit application | ✅ |

---

# Технологический стек

| Категория | Библиотеки |
|---|---|
| Data | pandas, numpy, geopandas, pyarrow |
| Visualization | matplotlib, seaborn, plotly, folium |
| ML | scikit-learn, catboost, lightgbm, xgboost |
| Tuning | optuna |
| Clustering | hdbscan, umap-learn |
| NLP | transformers, sentence-transformers, torch |
| Explainability | shap |
| Geo | geopy |
| App | streamlit |

---

# Ограничения проекта

- 43.95% listings без цены → selection bias
- Один snapshot → нет временного forecasting
- Amsterdam 30-night regulation влияет на интерпретацию occupancy
- `calendar.price` пустой в snapshot dataset
- Unavailability не равен реальному occupancy

---

# Главная цель проекта

Цель проекта — не только предсказание цены Airbnb.

Проект демонстрирует полный workflow реального ML-проекта:
- анализ данных,
- feature engineering,
- gradient boosting,
- explainable AI,
- NLP embeddings,
- clustering,
- recommendation systems,
- подготовка к deployment




# Airbnb Smart Pricing & Market Intelligence System

End-to-end ML project for the Airbnb market in Amsterdam: pricing prediction, market intelligence, host segmentation, and content-based recommendations.

The project is built as a modular ML pipeline — from raw Inside Airbnb data to feature engineering, gradient boosting models, explainable AI, clustering, recommendation systems, and deployment preparation.

**Dataset:** Inside Airbnb — Amsterdam  
**Snapshot:** 11 September 2025  
**Author:** Maksym Chunikhin

---

# Project Structure

```text
airbnb_project/
├── notebooks/
│   ├── 01_eda.ipynb
│   │   # Data understanding, cleaning, EDA, geo analysis
│   │
│   ├── 02_feature_engineering.ipynb
│   │   # 76 ML features, amenities parsing,
│   │   # NLP features, BERT sentiment,
│   │   # MiniLM embeddings, dataset preparation
│   │
│   ├── 03_pricing_model.ipynb
│   │   # CatBoost / LightGBM / XGBoost,
│   │   # Optuna tuning, ensemble,
│   │   # SHAP explainability
│   │
│   ├── 04_market_intelligence.ipynb
│   │   # Undervalued listings detection,
│   │   # fairness score,
│   │   # host segmentation,
│   │   # KMeans + HDBSCAN
│   │
│   └── 05_recommender.ipynb
│       # Content-based recommender,
│       # cosine similarity,
│       # embeddings + amenities + geo features
│
├── utils/
│   └── cleaning.py
│
├── data/
│   ├── raw/          # Inside Airbnb source files (not tracked)
│   └── processed/    # ML-ready parquet datasets (not tracked)
│
├── models/           # Trained models and artifacts (not tracked)
│
└── app/
    └── streamlit_app.py   # Streamlit dashboard 
```

---

# Project Overview

The project consists of five main stages:

## 1. Exploratory Data Analysis

Notebook: `01_eda.ipynb`

Main tasks:
- data understanding;
- cleaning and preprocessing;
- price analysis;
- neighborhood analysis;
- demand analysis;
- host analysis;
- spatial analysis;
- leakage detection.

---

## 2. Feature Engineering

Notebook: `02_feature_engineering.ipynb`

Main tasks:
- creation of ML features;
- amenities parsing;
- geo and demand features;
- NLP features from descriptions and reviews;
- multilingual BERT sentiment;
- MiniLM embeddings;
- train / validation / test preparation.

Output:
ML-ready datasets for pricing models and recommender systems.

---

## 3. Pricing ML Models

Notebook: `03_pricing_model.ipynb`

Main tasks:
- baseline regression models;
- CatBoost / LightGBM / XGBoost;
- Optuna hyperparameter tuning;
- ensemble blending;
- SHAP explainability;
- residual analysis;
- selection bias analysis.

Goal:
predict Airbnb listing prices using structured, geo, and NLP features.

---

## 4. Market Intelligence

Notebook: `04_market_intelligence.ipynb`

Main tasks:
- undervalued / overpriced listing detection;
- fairness score;
- revenue potential estimation;
- host segmentation;
- KMeans and HDBSCAN clustering.

Goal:
extract business insights from pricing residuals and host behavior.

---

## 5. Recommendation System

Notebook: `05_recommender.ipynb`

Main tasks:
- content-based recommendation pipeline;
- vector representation of listings;
- embeddings + amenities + geo features;
- cosine similarity + nearest neighbors.

Goal:
find similar Airbnb listings based on property characteristics.

---

# Current Status

| Notebook | Description | Status |
|---|---|:---:|
| `01_eda.ipynb` | EDA + cleaning + geo analysis | ✅ |
| `02_feature_engineering.ipynb` | ML features + NLP + embeddings | ✅ |
| `03_pricing_model.ipynb` | GBM models + Optuna + SHAP | ✅ |
| `04_market_intelligence.ipynb` | Market intelligence + clustering | ✅ |
| `05_recommender.ipynb` | Recommendation system | ✅ |
| `streamlit_app.py` | Streamlit application | ✅ |

---

# Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, numpy, geopandas, pyarrow |
| Visualization | matplotlib, seaborn, plotly, folium |
| ML | scikit-learn, catboost, lightgbm, xgboost |
| Tuning | optuna |
| Clustering | hdbscan, umap-learn |
| NLP | transformers, sentence-transformers, torch |
| Explainability | shap |
| Geo | geopy |
| App | streamlit |

---

# Honest Limitations

- 43.95% of listings have missing prices → selection bias
- Single snapshot → no temporal forecasting
- Amsterdam 30-night regulation affects occupancy interpretation
- `calendar.price` is empty in this dataset snapshot
- Unavailability does not necessarily mean real occupancy

---

# Main Goal

The goal of the project is not only to predict Airbnb prices.

The project demonstrates a full real-world ML workflow:
- data analysis,
- feature engineering,
- gradient boosting,
- explainable AI,
- NLP embeddings,
- clustering,
- recommendation systems,
- deployment preparation.