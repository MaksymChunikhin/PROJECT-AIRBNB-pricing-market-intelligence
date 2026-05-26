"""
Airbnb Smart Pricing & Market Intelligence — Streamlit App
Run:
    streamlit run app/streamlit_app.py
"""
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CITY_CENTER = (52.3676, 4.9041)

# Отображаемые названия категорий (данные в CSV не меняются)
CAT_LABELS = {"undervalued": "Выгодные", "fair": "Рыночные", "overpriced": "Переоценённые", "unknown": "Нет данных"}
CAT_COLORS = {"undervalued": "#2ecc71", "fair": "#3498db", "overpriced": "#e74c3c", "unknown": "#95a5a6"}

ROOM_TYPE_LABELS = {
    "Entire home/apt": "🏠 Всё жильё целиком",
    "Private room":    "🚪 Отдельная комната",
    "Hotel room":      "🏨 Номер в отеле",
    "Shared room":     "🛋️ Общая комната",
}
PROP_TYPE_LABELS = {
    "apartment": "🏢 Апартаменты / Квартира",
    "house":     "🏡 Дом / Коттедж",
    "hotel":     "🏨 Отель",
    "other":     "🔑 Другое",
}
ROOM_TYPE_INV = {v: k for k, v in ROOM_TYPE_LABELS.items()}
PROP_TYPE_INV  = {v: k for k, v in PROP_TYPE_LABELS.items()}

SEG_LABELS_RU = {
    "mainstream_hosts":       "Обычные хосты",
    "experienced_superhosts": "Опытные Superhost",
    "low_activity_hosts":     "Малоактивные хосты",
    "commercial_operators":   "Коммерческие операторы",
}

st.set_page_config(page_title="Airbnb Amsterdam Intelligence", page_icon="🏠", layout="wide")

# ---------- Loaders ----------
@st.cache_data(show_spinner="Загрузка listings...")
def load_listings():
    lf = pd.read_parquet(PROCESSED / "listings_features.parquet").reset_index(drop=True)
    lc = pd.read_csv(PROCESSED / "listings_classified.csv")
    return lf.merge(lc, on="id", how="left")

@st.cache_data(show_spinner="Загрузка hosts...")
def load_hosts():
    return pd.read_csv(PROCESSED / "hosts_segmented.csv")

@st.cache_data
def load_feature_cols():
    return pd.read_parquet(PROCESSED / "X_train.parquet").columns.tolist()

@st.cache_resource(show_spinner="Загрузка моделей...")
def load_models():
    with open(MODELS / "cat_cols.json") as f:
        cat_cols = json.load(f)
    with open(MODELS / "ensemble_weights.json") as f:
        ens = json.load(f)
    cat_model = cb.CatBoostRegressor(); cat_model.load_model(str(MODELS / "catboost_pricing.cbm"))
    lgb_model = lgb.Booster(model_file=str(MODELS / "lightgbm_pricing.txt"))
    xgb_model = xgb.Booster(); xgb_model.load_model(str(MODELS / "xgboost_pricing.json"))
    return {"cat": cat_model, "lgb": lgb_model, "xgb": xgb_model,
            "cat_cols": cat_cols, "weights": ens["weights"]}

@st.cache_resource(show_spinner="Загрузка SHAP...")
def load_shap():
    with open(MODELS / "shap_explainer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Загрузка recommender...")
def load_recommender():
    with open(MODELS / "recommender_nn.pkl", "rb") as f:
        art = pickle.load(f)
    X = np.load(PROCESSED / "X_combined.npy")
    return art, X

# ---------- Helpers ----------
def prepare_features(df, models):
    """Готовим фичи в формате моделей: str для CatBoost, category для LGB/XGB."""
    cat_cols = models["cat_cols"]
    X_cat_str = df.copy()
    for c in cat_cols:
        X_cat_str[c] = X_cat_str[c].fillna("unknown").astype(str)
    X_tree = X_cat_str.copy()
    for c in cat_cols:
        X_tree[c] = X_tree[c].astype("category")
    return X_cat_str, X_tree

def predict_ensemble(df, models):
    X_cat_str, X_tree = prepare_features(df, models)
    w = models["weights"]
    p = (
        w["cat"] * models["cat"].predict(X_cat_str)
        + w["lgb"] * models["lgb"].predict(X_tree)
        + w["xgb"] * models["xgb"].predict(xgb.DMatrix(X_tree, enable_categorical=True))
    )
    return np.expm1(p)

# ---------- Pages ----------
def page_dashboard(lf):
    st.title("🗺️ Аналитика рынка")
    st.caption("Интерактивный обзор рынка краткосрочной аренды в Амстердаме на основе данных Airbnb")

    c1, c2 = st.columns(2)
    nb_options = ["Все"] + sorted(lf["neighbourhood_cleansed"].dropna().unique().tolist())
    nb = c1.selectbox("Район", nb_options)
    rt_display = ["Все"] + [ROOM_TYPE_LABELS.get(v, v) for v in sorted(lf["room_type"].dropna().unique())]
    rt_label = c2.selectbox("Тип комнаты", rt_display)
    rt = ROOM_TYPE_INV.get(rt_label, rt_label)

    df = lf.copy()
    if nb != "Все":
        df = df[df["neighbourhood_cleansed"] == nb]
    if rt_label != "Все":
        df = df[df["room_type"] == rt]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings", f"{len(df):,}")
    c2.metric("Медианная цена", f"€{df['price'].median():.0f}" if len(df) else "–")
    c3.metric("Средний рейтинг", f"{df['review_scores_rating'].mean():.2f}" if len(df) else "–")
    c4.metric("Superhost", f"{(df['host_is_superhost'].eq('t').mean()*100):.1f}%" if len(df) else "–")

    st.subheader("Карта listings")
    st.caption("Цвет точки = категория относительно model-based benchmark: 🟢 Выгодные, 🔵 Рыночные, 🔴 Переоценённые")
    map_df = df.dropna(subset=["latitude", "longitude"])
    if len(map_df):
        m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="cartodbpositron")
        cluster = MarkerCluster(options={"maxClusterRadius": 40}).add_to(m)
        for _, r in map_df.iterrows():
            cat = r.get("category", "unknown")
            url = r.get("listing_url", "")
            link = f'<a href="{url}" target="_blank">Открыть на Airbnb ↗</a>' if url else ""
            folium.CircleMarker(
                [r["latitude"], r["longitude"]], radius=4,
                color=CAT_COLORS.get(cat, "#95a5a6"),
                fill=True, fill_opacity=0.75,
                popup=folium.Popup(
                    f"€{r['price']:.0f} | {r['room_type']}<br>"
                    f"{CAT_LABELS.get(cat, cat)}<br>{link}",
                    max_width=220,
                ),
            ).add_to(cluster)
        st_folium(m, height=480, width=None, returned_objects=[])
    else:
        st.info("Нет listings для отображения")

    st.subheader("Медианная цена по районам")
    nb_stats = (
        df.groupby("neighbourhood_cleansed")
        .agg(median_price=("price", "median"), listings=("id", "count"))
        .query("listings >= 10")
        .sort_values("median_price", ascending=False)
        .reset_index()
    )
    if len(nb_stats):
        fig = px.bar(nb_stats, x="median_price", y="neighbourhood_cleansed",
                     orientation="h", color="median_price", color_continuous_scale="Viridis",
                     labels={"median_price": "Median price, €", "neighbourhood_cleansed": ""})
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

def page_predictor(lf, feat_cols, models, explainer):
    st.title("💰 Прогноз цены")
    st.caption("Укажите параметры жилья, и AI-система рассчитает ожидаемую цену за ночь на основе рыночных данных Airbnb")

    c1, c2, c3 = st.columns(3)
    nb = c1.selectbox("Район", sorted(lf["neighbourhood_cleansed"].dropna().unique()))
    rt_label = c2.selectbox("Тип комнаты", [ROOM_TYPE_LABELS.get(v, v) for v in sorted(lf["room_type"].dropna().unique())])
    pt_label = c3.selectbox("Тип жилья", [PROP_TYPE_LABELS.get(v, v) for v in sorted(lf["property_type_grp"].dropna().unique())])

    rt = ROOM_TYPE_INV.get(rt_label, rt_label)
    pt = PROP_TYPE_INV.get(pt_label, pt_label)

    c1, c2, c3, c4 = st.columns(4)
    accommodates = c1.number_input("Гостей", 1, 16, 2)
    bedrooms = c2.number_input("Спален", 0, 10, 1)
    bathrooms = c3.number_input("Ванных", 0.0, 8.0, 1.0, step=0.5, format="%g")
    beds = c4.number_input("Кроватей", 1, 16, 1)

    c1, c2 = st.columns(2)
    superhost = c1.checkbox("Superhost", value=False)
    instant = c2.checkbox("Мгновенное бронирование", value=False)

    st.markdown("**Удобства:**")
    amen_cols = ["has_wifi", "has_kitchen", "has_ac", "has_washer", "has_dishwasher",
                 "has_parking", "has_workspace", "has_balcony", "has_self_checkin",
                 "has_pool", "has_gym", "has_elevator"]
    amen_labels = ["Wi-Fi", "Кухня", "Кондиционер", "Стиральная", "Посудомойка",
                   "Парковка", "Рабочее место", "Балкон", "Self check-in",
                   "Бассейн", "Спортзал", "Лифт"]
    amen_vals = {}
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(amen_cols, amen_labels)):
        amen_vals[col] = cols[i % 4].checkbox(label, value=col in {"has_wifi", "has_kitchen"})

    # Берём медианный template-row для района (или общий, если в районе мало listings)
    nb_rows = lf[lf["neighbourhood_cleansed"] == nb]
    template_src = nb_rows if len(nb_rows) >= 5 else lf

    row = {}
    for col in feat_cols:
        s = template_src[col].dropna() if col in template_src.columns else pd.Series(dtype=float)
        if not len(s):
            row[col] = 0
            continue
        row[col] = s.median() if pd.api.types.is_numeric_dtype(s) else s.mode().iloc[0]

    row["neighbourhood_cleansed"] = nb
    row["property_type_grp"] = pt
    row["accommodates"] = accommodates
    row["bedrooms"] = bedrooms
    row["bathrooms"] = bathrooms
    row["bathrooms_n"] = bathrooms
    row["beds"] = beds
    row["host_is_superhost"] = "t" if superhost else "f"
    row["instant_bookable"] = "t" if instant else "f"
    rt_ord = {"Entire home/apt": 3, "Private room": 2, "Hotel room": 1, "Shared room": 0}
    row["room_type_ord"] = rt_ord.get(rt, 2)
    for col in amen_cols:
        row[col] = int(amen_vals[col])
    row["amenities_count"] = sum(amen_vals.values()) + 3

    X_input = pd.DataFrame([row])[feat_cols]

    if st.button("Предсказать цену", type="primary"):
        pred = predict_ensemble(X_input, models)[0]
        nb_median = nb_rows["price"].median() if len(nb_rows) else lf["price"].median()

        c1, c2, c3 = st.columns(3)
        c1.metric("Предсказанная цена", f"€{pred:.0f}")
        c2.metric(f"Медиана района", f"€{nb_median:.0f}", delta=f"{pred - nb_median:+.0f} €")
        c3.metric("Город медиана", f"€{lf['price'].median():.0f}")

        st.subheader("Объяснение предсказания (SHAP, top-10 факторов)")
        try:
            X_cat_str, _ = prepare_features(X_input, models)
            sv = explainer.shap_values(X_cat_str)
            sv_row = sv[0] if sv.ndim > 1 else sv
            base = float(explainer.expected_value)

            FEAT_RU = {
                "bedrooms": "Количество спален",
                "accommodates": "Максимум гостей",
                "bathrooms": "Количество ванных",
                "bathrooms_n": "Ванных (число)",
                "beds": "Количество кроватей",
                "price": "Цена за ночь",
                "room_type_ord": "Формат жилья",
                "distance_to_center": "Расстояние до центра",
                "neighbourhood_cleansed": "Район",
                "property_type_grp": "Тип недвижимости",
                "host_is_superhost": "Superhost",
                "host_experience_years": "Опыт хоста",
                "host_listings_count": "Активных объектов у хоста",
                "host_total_listings_count": "Всего объектов у хоста",
                "host_response_rate": "Скорость ответа",
                "host_acceptance_rate": "Доля принятых бронирований",
                "instant_bookable": "Instant booking",
                "calendar_unavail_rate": "Уровень занятости",
                "unavailability_rate": "Общая занятость",
                "unavail_summer": "Занятость летом",
                "unavail_winter": "Занятость зимой",
                "unavail_other": "Занятость вне сезона",
                "unavail_weekend_diff": "Разница спроса: выходные vs будни",
                "reviews_per_month": "Отзывов в месяц",
                "number_of_reviews": "Количество отзывов",
                "number_of_reviews_ltm": "Отзывов за последние 12 месяцев",
                "review_scores_rating": "Общий рейтинг",
                "review_scores_cleanliness": "Чистота",
                "review_scores_location": "Локация",
                "review_scores_value": "Соотношение цена/качество",
                "review_scores_accuracy": "Точность описания",
                "review_scores_checkin": "Удобство заселения",
                "review_scores_communication": "Коммуникация с хостом",
                "reviews_sentiment_transformer": "Тональность отзывов",
                "amenities_count": "Количество удобств",
                "amenities_premium_score": "Уровень премиум-удобств",
                "has_wifi": "Wi-Fi",
                "has_kitchen": "Кухня",
                "has_ac": "Кондиционер",
                "has_washer": "Стиральная машина",
                "has_dishwasher": "Посудомоечная машина",
                "has_parking": "Парковка",
                "has_workspace": "Рабочее место",
                "has_balcony": "Балкон",
                "has_self_checkin": "Самостоятельное заселение",
                "has_pool": "Бассейн",
                "has_gym": "Спортзал",
                "has_elevator": "Лифт",
                "neighborhood_listing_density": "Плотность Airbnb в районе",
                "is_tourist_zone": "Туристический район",
                "geo_cluster": "Географическая зона",
                "minimum_nights": "Минимальный срок бронирования",
                "maximum_nights": "Максимальный срок бронирования",
                "avg_min_nights": "Средний минимальный срок аренды",
                "professional_host": "Профессиональный хост",
                "host_has_about": "Заполненный профиль хоста",
                "host_verifications_count": "Количество верификаций",
                "name_length": "Длина заголовка",
                "description_length": "Длина описания",
                "description_has_keywords": "Ключевые слова в описании",
                "calculated_host_listings_count": "Объектов у хоста",
                "calculated_host_listings_count_entire_homes": "Полных объектов у хоста",
                "calculated_host_listings_count_private_rooms": "Приватных комнат у хоста",
                "calculated_host_listings_count_shared_rooms": "Общих комнат у хоста",
                "bathrooms_shared": "Общая ванная комната",
                "latitude": "Широта",
                "longitude": "Долгота",
            }

            shap_df = pd.DataFrame({
                "feature": feat_cols,
                "value": X_cat_str.iloc[0].values,
                "shap": sv_row,
            })
            shap_df["abs"] = shap_df["shap"].abs()
            shap_df["feature_ru"] = shap_df["feature"].map(FEAT_RU).fillna(shap_df["feature"])
            top = shap_df.nlargest(10, "abs").sort_values("shap")

            fig = px.bar(top, x="shap", y="feature_ru", orientation="h",
                         color="shap", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                         labels={"shap": "SHAP (влияние на log-price)", "feature_ru": ""},
                         hover_data={"value": True, "feature": True})
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Base value (log-price): {base:.3f}. Положительные SHAP повышают цену, отрицательные — понижают.")
        except Exception as e:
            st.warning(f"SHAP недоступен: {e}")

def page_investment(lf):
    st.title("📈 Инвестиционные возможности")
    st.caption("Анализ ценового позиционирования listings: найдите объекты с потенциалом роста цены или те, что теряют спрос из-за завышенной стоимости")

    CAT_OPTIONS = {
        "🟢 Выгодные (цена ниже модели)":      "undervalued",
        "🔴 Переоценённые (цена выше модели)":  "overpriced",
        "⚪ Рыночные":                           "fair",
    }
    cat_label = st.radio("Категория", list(CAT_OPTIONS.keys()), horizontal=True)
    cat_val = CAT_OPTIONS[cat_label]

    pool = lf[lf["category"] == cat_val].copy()

    c1, c2, c3 = st.columns(3)
    nb_options = ["Все"] + sorted(pool["neighbourhood_cleansed"].dropna().unique().tolist())
    nb = c1.selectbox("Район", nb_options)
    min_rating = c2.slider("Мин. рейтинг", 0.0, 5.0, 4.0, 0.1)
    min_reviews = c3.slider("Мин. число отзывов", 0, 200, 0)

    f = pool.copy()
    if nb != "Все":
        f = f[f["neighbourhood_cleansed"] == nb]
    f = f[(f["review_scores_rating"].fillna(0) >= min_rating) & (f["number_of_reviews"] >= min_reviews)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Найдено объектов", f"{len(f):,}")
    gap = (f['predicted_price'] - f['price']).median()
    c2.metric("Медианный разрыв с AI", f"€{abs(gap):.0f}" if len(f) else "–")
    c3.metric("Медианный рейтинг", f"{f['review_scores_rating'].median():.2f}" if len(f) else "–")

    st.subheader("Карта")
    sample = f.dropna(subset=["latitude", "longitude"]).head(400)
    dot_color = CAT_COLORS.get(cat_val, "#95a5a6")
    if len(sample):
        m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="cartodbpositron")
        for _, r in sample.iterrows():
            url = r.get("listing_url", "")
            link = f'<a href="{url}" target="_blank">Открыть на Airbnb ↗</a>' if url else ""
            folium.CircleMarker(
                [r["latitude"], r["longitude"]], radius=5, color=dot_color,
                fill=True, fill_opacity=0.8,
                popup=folium.Popup(
                    f"€{r['price']:.0f} (модель: €{r['predicted_price']:.0f})<br>"
                    f"z: {r['residual_z']:.2f}<br>{link}",
                    max_width=240,
                ),
            ).add_to(m)
        st_folium(m, height=440, width=None, returned_objects=[])

    show_cols = ["id", "neighbourhood_cleansed", "room_type", "price", "predicted_price",
                 "residual_z", "review_scores_rating", "number_of_reviews", "fairness_score"]
    sort_asc = cat_val == "undervalued"
    tbl = f.nsmallest(50, "residual_z")[show_cols].round(2).copy() if sort_asc \
          else f.nlargest(50, "residual_z")[show_cols].round(2).copy()
    tbl["room_type"] = tbl["room_type"].map(ROOM_TYPE_LABELS).fillna(tbl["room_type"])
    tbl["price_gap"] = (tbl["predicted_price"] - tbl["price"]).round(0).astype(int)
    tbl = tbl.drop(columns=["residual_z"])
    # URL маппим до конвертации id в строку
    if "listing_url" in f.columns:
        url_map = f.set_index("id")["listing_url"].to_dict()
        tbl["listing_url"] = tbl["id"].map(url_map)
    else:
        tbl["listing_url"] = ""
    tbl["id"] = tbl["id"].astype(str)
    tbl = tbl[["id", "neighbourhood_cleansed", "room_type",
               "price", "predicted_price", "price_gap",
               "review_scores_rating", "number_of_reviews", "listing_url"]]
    tbl = tbl.rename(columns={
        "id":                     "ID объекта",
        "neighbourhood_cleansed": "Район",
        "room_type":              "Тип жилья",
        "price":                  "Цена, €",
        "predicted_price":        "Цена рассчитанная AI, €",
        "price_gap":              "Разрыв с AI, €",
        "review_scores_rating":   "Рейтинг",
        "number_of_reviews":      "Отзывов",
        "listing_url":            "Airbnb",
    })
    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Airbnb": st.column_config.LinkColumn(
                "Airbnb ↗",
                display_text="🔗 Открыть",
            )
        },
    )

def page_recommender(lf):
    st.title("🔍 Похожие объекты")
    st.caption("Кликните на объект на карте — система найдёт похожие listings")

    art, X = load_recommender()
    nn = art["nn"]
    id_to_idx = art["id_to_idx"]

    c1, c2 = st.columns(2)
    same_rt = c1.checkbox("Только тот же тип жилья", value=True)
    price_band = c2.slider("Ценовой коридор (±%)", 0, 100, 30) / 100

    # Карта всех listings — кликаем, получаем ID из tooltip
    st.subheader("Выберите объект на карте")
    map_df = lf.dropna(subset=["latitude", "longitude"])
    m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="cartodbpositron")
    cluster = MarkerCluster(options={"maxClusterRadius": 40}).add_to(m)
    for _, r in map_df.iterrows():
        cat = r.get("category", "unknown")
        url = r.get("listing_url", "")
        link = f'<a href="{url}" target="_blank">Открыть на Airbnb ↗</a>' if url else ""
        name = str(r.get("name", ""))[:50] or "—"
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=4,
            color=CAT_COLORS.get(cat, "#95a5a6"),
            fill=True, fill_opacity=0.7,
            tooltip=str(r["id"]),
            popup=folium.Popup(
                f"<b>{name}</b><br>€{r['price']:.0f} | {r['room_type']}<br>"
                f"{CAT_LABELS.get(cat, cat)}<br>{link}",
                max_width=260,
            ),
        ).add_to(cluster)

    map_data = st_folium(m, height=500, width=None,
                         returned_objects=["last_object_clicked_tooltip"])

    # Извлекаем ID кликнутого маркера
    clicked_id = None
    raw = (map_data or {}).get("last_object_clicked_tooltip")
    if raw:
        try:
            clicked_id = int(raw)
        except (ValueError, TypeError):
            pass

    if clicked_id is None:
        st.info("👆 Кликните на любую точку на карте, чтобы увидеть похожие объекты")
        return

    if clicked_id not in id_to_idx:
        st.warning("Объект не найден в индексе рекомендаций")
        return

    idx = id_to_idx[clicked_id]
    src = lf.iloc[idx]

    # Карточка источника
    st.divider()
    st.subheader("Выбранный объект")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Цена", f"€{src['price']:.0f}")
    c2.metric("Район", src["neighbourhood_cleansed"])
    c3.metric("Тип жилья", src["room_type"])
    c4.metric("Рейтинг", f"{src['review_scores_rating']:.2f}" if pd.notna(src['review_scores_rating']) else "–")
    src_url = src.get("listing_url", "")
    if src_url:
        st.markdown(f"[Открыть на Airbnb ↗]({src_url})")

    # Рекомендации
    dists, idxs = nn.kneighbors(X[idx:idx + 1], n_neighbors=min(200, len(lf)))
    recs = lf.iloc[idxs[0]].copy()
    recs["similarity"] = 1 - dists[0]
    recs = recs[recs["id"] != clicked_id]
    if same_rt:
        recs = recs[recs["room_type"] == src["room_type"]]
    if price_band:
        lo, hi = src["price"] * (1 - price_band), src["price"] * (1 + price_band)
        recs = recs[recs["price"].between(lo, hi)]

    top10 = recs.head(10)
    show_cols = ["id", "neighbourhood_cleansed", "room_type", "price",
                 "review_scores_rating", "number_of_reviews", "similarity"]
    tbl = top10[show_cols].round(3).copy()
    if "listing_url" in top10.columns:
        url_map = top10.set_index("id")["listing_url"].to_dict()
        tbl["listing_url"] = tbl["id"].map(url_map)
    else:
        tbl["listing_url"] = ""
    tbl["id"] = tbl["id"].astype(str)
    tbl["room_type"] = tbl["room_type"].map(ROOM_TYPE_LABELS).fillna(tbl["room_type"])
    tbl = tbl.rename(columns={
        "id":                     "ID объекта",
        "neighbourhood_cleansed": "Район",
        "room_type":              "Тип жилья",
        "price":                  "Цена, €",
        "review_scores_rating":   "Рейтинг",
        "number_of_reviews":      "Отзывов",
        "similarity":             "Схожесть",
        "listing_url":            "Airbnb",
    })
    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Airbnb": st.column_config.LinkColumn("Airbnb ↗", display_text="🔗 Открыть")
        },
    )

    # Карта источник + рекомендации
    st.subheader("Объект и рекомендации")
    top = top10.dropna(subset=["latitude", "longitude"])
    m2 = folium.Map(location=[src["latitude"], src["longitude"]], zoom_start=13, tiles="cartodbpositron")
    src_link = f'<a href="{src_url}" target="_blank">Открыть ↗</a>' if src_url else ""
    folium.Marker(
        [src["latitude"], src["longitude"]],
        popup=folium.Popup(f"<b>Источник</b><br>€{src['price']:.0f}<br>{src_link}", max_width=220),
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m2)
    for _, r in top.iterrows():
        url = r.get("listing_url", "")
        link = f'<a href="{url}" target="_blank">Открыть ↗</a>' if url else ""
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=6, color="#3498db",
            fill=True, fill_opacity=0.8,
            popup=folium.Popup(
                f"€{r['price']:.0f} | схожесть: {r['similarity']:.2f}<br>{link}",
                max_width=220,
            ),
        ).add_to(m2)
    st_folium(m2, height=420, width=None, returned_objects=[])

def page_hosts(lf, hosts):
    st.title("👤 Аналитика хостов")
    st.caption("Анализ хостов на рынке краткосрочной аренды Амстердама")

    st.subheader("Сегменты хостов Амстердама")
    seg_stats = hosts.groupby("final_label").agg(
        hosts=("host_id", "count"),
        median_listings=("host_listings_count", "median"),
        median_experience=("host_experience_years", "median"),
        median_price=("mean_price", "median"),
        median_occupancy=("median_unavail_rate", "median"),
        superhost_share=("is_superhost", "mean"),
    ).round(2).sort_values("hosts", ascending=False).reset_index()

    seg_stats["segment_ru"] = seg_stats["final_label"].map(SEG_LABELS_RU).fillna(seg_stats["final_label"])
    fig = px.pie(seg_stats, values="hosts", names="segment_ru",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 hole=0.4)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(height=380, showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    seg_display = seg_stats.rename(columns={
        "segment_ru":        "Сегмент",
        "hosts":             "Хостов",
        "median_listings":   "Объектов (медиана)",
        "median_experience": "Опыт, лет",
        "median_price":      "Средняя цена, €",
        "median_occupancy":  "Занятость",
    }).drop(columns=["final_label", "superhost_share"], errors="ignore")
    st.dataframe(seg_display, use_container_width=True, hide_index=True)

    st.subheader("Профиль хоста")

    # Поиск по ID или карта
    search_id = st.text_input("Вставьте ID объекта (или кликните на карте)", placeholder="Например: 27886")

    st.caption("Или выберите объект на карте:")
    map_df = lf.dropna(subset=["latitude", "longitude"])
    m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="cartodbpositron")
    cluster = MarkerCluster(options={"maxClusterRadius": 40}).add_to(m)
    for _, r in map_df.iterrows():
        cat = r.get("category", "unknown")
        name = str(r.get("name", ""))[:40] or "—"
        host_name_marker = str(r.get("host_name", "")) or "—"
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=4,
            color=CAT_COLORS.get(cat, "#95a5a6"),
            fill=True, fill_opacity=0.7,
            tooltip=str(r["id"]),
            popup=folium.Popup(
                f"<b>{name}</b><br>Хост: {host_name_marker}<br>"
                f"€{r['price']:.0f} | {ROOM_TYPE_LABELS.get(r['room_type'], r['room_type'])}<br>"
                f"{CAT_LABELS.get(cat, cat)}",
                max_width=260,
            ),
        ).add_to(cluster)

    map_data = st_folium(m, height=460, width=None,
                         returned_objects=["last_object_clicked_tooltip"])

    # Определяем listing_id: сначала текстовый ввод, потом клик на карте
    listing_id = None
    if search_id.strip():
        try:
            listing_id = int(search_id.strip())
        except ValueError:
            st.warning("Введите числовой ID объекта")
            return
    else:
        raw = (map_data or {}).get("last_object_clicked_tooltip")
        if raw:
            try:
                listing_id = int(raw)
            except (ValueError, TypeError):
                pass

    if listing_id is None:
        st.info("👆 Кликните на объект на карте или введите ID выше")
        return

    src = lf[lf["id"] == listing_id]
    if not len(src):
        st.warning("Объект не найден в базе")
        return

    src = src.iloc[0]
    host_id = src["host_id"]
    host_name = src.get("host_name", "") or ""
    st.caption(f"Хост: **{host_name}** (ID: {host_id})")

    h = hosts[hosts["host_id"] == host_id]
    if not len(h):
        st.info(f"Хост {host_name or host_id} есть в базе, но не попал в сегментацию — вероятно, недостаточно данных о ценах или активности.")
        return
    h = h.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Сегмент", SEG_LABELS_RU.get(h["final_label"], h["final_label"]))
    c2.metric("Объектов", int(h["host_listings_count"]))
    c3.metric("Опыт, лет", f"{h['host_experience_years']:.1f}")
    c4.metric("Superhost", "✅" if h["is_superhost"] else "—")

    c1, c2 = st.columns(2)
    c1.metric("Средняя цена портфолио", f"€{h['mean_price']:.0f}")
    c2.metric("Уровень занятости", f"{h['median_unavail_rate']:.2f}")

    seg = h["final_label"]
    radar_cols = ["host_listings_count", "host_experience_years", "mean_price",
                  "median_unavail_rate", "host_response_rate"]
    radar_labels = {
        "host_listings_count":   "Объектов",
        "host_experience_years": "Опыт, лет",
        "mean_price":            "Ср. цена",
        "median_unavail_rate":   "Занятость",
        "host_response_rate":    "Отклик",
    }
    seg_med = hosts[hosts["final_label"] == seg][radar_cols].median()
    host_vals = h[radar_cols]
    norm = (host_vals / seg_med.replace(0, np.nan)).fillna(1).clip(0, 3)

    radar_df = pd.DataFrame({
        "metric": [radar_labels.get(c, c) for c in norm.index],
        "value": norm.values,
    })
    seg_ru = SEG_LABELS_RU.get(seg, seg)
    fig = px.line_polar(radar_df, r="value", theta="metric", line_close=True,
                        title=f"Хост vs медиана сегмента «{seg_ru}» (1.0 = на уровне сегмента)")
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

    host_listings = lf[lf["host_id"] == host_id]
    if len(host_listings):
        st.subheader(f"Listings хоста ({len(host_listings)})")
        show_cols = ["id", "neighbourhood_cleansed", "room_type", "price",
                     "predicted_price", "category", "review_scores_rating"]
        avail = [c for c in show_cols if c in host_listings.columns]
        tbl = host_listings[avail].round(2).copy()
        if "id" in tbl.columns:
            tbl["id"] = tbl["id"].astype(str)
        if "room_type" in tbl.columns:
            tbl["room_type"] = tbl["room_type"].map(ROOM_TYPE_LABELS).fillna(tbl["room_type"])
        if "category" in tbl.columns:
            tbl["category"] = tbl["category"].map(CAT_LABELS).fillna(tbl["category"])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ---------- Main ----------
def main():
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] [role="radiogroup"] label,
        section[data-testid="stSidebar"] p {
            font-size: 1.25rem !important;
            line-height: 1.8 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🏠 Airbnb Amsterdam")
    st.sidebar.caption("AI-аналитика цен и рынка Airbnb")

    pages = {
        "🗺️ Аналитика рынка":            "dashboard",
        "💰 Прогноз цены":               "predictor",
        "📈 Инвестиционные возможности":  "investment",
        "🔍 Похожие объекты":            "recommender",
        "👤 Аналитика хостов":           "hosts",
    }
    choice = st.sidebar.radio("Страница", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Inside Airbnb Data • 11.09.2025  \nML & Market Intelligence Project by Maksym Chunikhin")

    lf = load_listings()
    page = pages[choice]

    if page == "dashboard":
        page_dashboard(lf)
    elif page == "predictor":
        page_predictor(lf, load_feature_cols(), load_models(), load_shap())
    elif page == "investment":
        page_investment(lf)
    elif page == "recommender":
        page_recommender(lf)
    elif page == "hosts":
        page_hosts(lf, load_hosts())

if __name__ == "__main__":
    main()
