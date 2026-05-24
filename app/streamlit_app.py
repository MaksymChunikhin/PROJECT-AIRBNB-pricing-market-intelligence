"""
Airbnb Smart Pricing & Market Intelligence — Streamlit App
🚧 In development — models and data pipelines are ready (notebooks 01–05),
   UI implementation is in progress.

Pages (planned):
  1. Market Dashboard      — map + KPI cards + filters
  2. Price Predictor       — form → ensemble prediction + SHAP waterfall
  3. Investment Finder     — undervalued listings + map
  4. Similar Listings      — content-based ANN recommender
  5. Host Insights         — segment profile + pricing comparison

Run:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Airbnb Amsterdam Intelligence",
    page_icon="🏠",
    layout="wide",
)

# --- Dev banner ---
st.warning("🚧 **Application in development.** ML models and data pipelines are complete (see notebooks 01–05). UI coming soon.", icon="🚧")

# --- Navigation ---
PAGES = {
    "🗺️ Market Dashboard":  "dashboard",
    "💰 Price Predictor":   "predictor",
    "📈 Investment Finder": "investment",
    "🔍 Similar Listings":  "recommender",
    "👤 Host Insights":     "hosts",
}

st.sidebar.title("Airbnb Amsterdam")
st.sidebar.caption("🚧 In development")
page = st.sidebar.radio("Page", list(PAGES.keys()))

# --- Routing ---
if PAGES[page] == "dashboard":
    st.title("🗺️ Market Dashboard")
    st.info("Folium map + KPI cards + neighbourhood filters")

elif PAGES[page] == "predictor":
    st.title("💰 Price Predictor")
    st.info("Input form → CatBoost/LGB/XGB ensemble prediction + SHAP waterfall")

elif PAGES[page] == "investment":
    st.title("📈 Investment Finder")
    st.info("Undervalued listings table + map (model from `04_market_intelligence`)")

elif PAGES[page] == "recommender":
    st.title("🔍 Similar Listings")
    st.info("Input listing_id → top-10 similar via ANN on MiniLM 384-d embeddings")

elif PAGES[page] == "hosts":
    st.title("👤 Host Insights")
    st.info("Host segment profile + pricing comparison vs district/cluster")
