# ============================================================
# DTI — Dynamic Discount Optimizer  |  Streamlit App
# ============================================================
# Converts the DTI Jupyter notebook into a production-ready
# single-file Streamlit web application.
#
# Features:
#   • Upload CSV or use built-in synthetic sample data
#   • Automated preprocessing & feature engineering
#   • Price / discount elasticity estimation
#   • ML model training  (Linear Regression, Random Forest, XGBoost)
#   • Optimal-discount recommendations
#   • Revenue-impact simulation
#   • Real-time Simulation Mode (slider → instant KPIs)
#   • Interactive Plotly charts + st.dataframe tables
# ============================================================

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Optional XGBoost — gracefully skip if not installed
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── App-wide constants ─────────────────────────────────────────────────────────
SEED     = 42
FEATURES = ["price", "units_sold", "interaction_score",
            "sentiment_score", "season_enc", "festival",
            "category_enc", "sales_value"]
TARGET   = "discount"

SEASON_MAP   = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
SEASON_NAMES = ["winter", "spring", "summer", "autumn"]
CATEGORIES   = ["electronics", "fashion", "grocery", "home_appliances",
                 "sports", "beauty", "books", "toys"]

np.random.seed(SEED)

# ===========================================================================
# ① PAGE CONFIG & GLOBAL STYLING
# ===========================================================================
st.set_page_config(
    page_title="DTI — Discount Optimizer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px 16px;
}
/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #1e1e2e; }
section[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
section[data-testid="stSidebar"] h2 { color: #cba6f7 !important; }
/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    border-radius: 14px;
    padding: 24px 32px;
    color: white;
    margin-bottom: 24px;
}
.app-header h1 { margin: 0; font-size: 2rem; }
.app-header p  { margin: 6px 0 0; opacity: .85; }
/* ── Section dividers ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 28px 0 8px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# ② SAMPLE-DATA GENERATOR
# ===========================================================================
@st.cache_data(show_spinner=False)
def generate_sample_data(n: int = 800) -> pd.DataFrame:
    """
    Creates a realistic synthetic retail dataset so the app works
    out-of-the-box with no file upload.

    Demand follows a log-linear price-sensitivity model:
        log(units_sold) = a - b*log(price) + c*log(1+discount) + noise
    """
    rng = np.random.default_rng(SEED)

    product_ids  = [f"P{str(i).zfill(4)}" for i in range(1, n + 1)]
    categories   = rng.choice(CATEGORIES, size=n)
    seasons      = rng.choice(SEASON_NAMES, size=n)
    festivals    = rng.choice([0, 1], size=n, p=[0.75, 0.25])

    prices       = rng.uniform(20, 500, size=n).round(2)
    discounts    = rng.uniform(0, 45, size=n).round(1)
    sentiment    = rng.uniform(-0.6, 0.9, size=n).round(3)
    interact     = rng.uniform(0.5, 10, size=n).round(2)

    # Price-sensitive demand
    log_demand   = (
        4.5
        - 0.55  * np.log1p(prices)
        + 0.45  * np.log1p(discounts)
        + 0.30  * sentiment
        + 0.15  * interact
        + 0.20  * festivals
        + rng.normal(0, 0.4, n)
    )
    units_sold   = np.clip(np.exp(log_demand).round(), 1, None)
    sales_value  = (prices * (1 - discounts / 100) * units_sold).round(2)

    df = pd.DataFrame({
        "product_id"        : product_ids,
        "category"          : categories,
        "season"            : seasons,
        "festival"          : festivals,
        "price"             : prices,
        "discount"          : discounts,
        "units_sold"        : units_sold,
        "sentiment_score"   : sentiment,
        "interaction_score" : interact,
        "sales_value"       : sales_value,
    })
    return df


# ===========================================================================
# ③ DATA VALIDATION & PREPROCESSING
# ===========================================================================
REQUIRED_COLS = {"price", "discount", "units_sold"}

def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Checks uploaded CSV has the minimum required columns."""
    missing = REQUIRED_COLS - set(df.columns.str.lower())
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, ""


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, type-casts, and engineers features from a raw dataframe.
    Works on both uploaded CSVs and the synthetic sample.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # ── Numeric coercion ────────────────────────────────────────────
    for col in ["price", "discount", "units_sold", "sales_value",
                "sentiment_score", "interaction_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "units_sold"])
    df["price"]      = df["price"].clip(lower=0.01)
    df["units_sold"] = df["units_sold"].clip(lower=0)
    df["discount"]   = df.get("discount", pd.Series(10.0, index=df.index))
    df["discount"]   = df["discount"].clip(0, 60).fillna(10)

    # ── Fill optional columns ───────────────────────────────────────
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    if "interaction_score" not in df.columns:
        df["interaction_score"] = 1.0
    if "sales_value" not in df.columns:
        df["sales_value"] = df["price"] * df["units_sold"]
    if "festival" not in df.columns:
        df["festival"] = 0
    if "season" not in df.columns:
        df["season"] = "summer"
    if "category" not in df.columns:
        df["category"] = "general"
    if "product_id" not in df.columns:
        df["product_id"] = [f"P{str(i).zfill(4)}" for i in range(1, len(df) + 1)]

    df["sentiment_score"]   = df["sentiment_score"].clip(-1, 1).fillna(0)
    df["interaction_score"] = df["interaction_score"].fillna(1)
    df["sales_value"]       = df["sales_value"].fillna(df["price"] * df["units_sold"])
    df["festival"]          = df["festival"].fillna(0).astype(int)

    # ── Encode categoricals ─────────────────────────────────────────
    df["season"]      = df["season"].str.lower().fillna("summer")
    df["season_enc"]  = df["season"].map(SEASON_MAP).fillna(2).astype(int)

    df["category"]    = df["category"].astype(str).str.lower().str.replace(" ", "_").fillna("general")
    le = LabelEncoder()
    df["category_enc"] = le.fit_transform(df["category"])

    df = df.drop_duplicates("product_id").reset_index(drop=True)
    return df


# ===========================================================================
# ④ ELASTICITY ESTIMATION
# ===========================================================================
def estimate_elasticity(df: pd.DataFrame) -> tuple[float, float]:
    """
    Log-log OLS regression to estimate price & discount elasticities.
    Returns (price_elasticity, discount_elasticity).
    """
    tmp = df[["price", "discount", "units_sold"]].copy()
    tmp["log_price"]    = np.log1p(tmp["price"])
    tmp["log_discount"] = np.log1p(tmp["discount"])
    tmp["log_units"]    = np.log1p(tmp["units_sold"])
    tmp = tmp.dropna()

    if len(tmp) < 10:
        return -0.55, 0.45   # sensible defaults

    m_p = LinearRegression().fit(tmp[["log_price"]],    tmp["log_units"])
    m_d = LinearRegression().fit(tmp[["log_discount"]], tmp["log_units"])
    return float(m_p.coef_[0]), float(m_d.coef_[0])


# ===========================================================================
# ⑤ MODEL TRAINING
# ===========================================================================
@st.cache_resource(show_spinner=False)
def train_models(df_hash: int, df: pd.DataFrame) -> dict:
    """
    Trains LR, RF, and (optionally) XGBoost on the preprocessed dataframe.
    `df_hash` is used purely for cache-busting when the data changes.
    Returns a dict with model objects, metrics, and the best model name.
    """
    model_df = df[FEATURES + [TARGET]].dropna()
    if len(model_df) < 30:
        st.warning("⚠️  Too few samples to train reliably (< 30 rows).")
        return {}

    X = model_df[FEATURES]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    results = {}

    # Linear Regression
    lr      = LinearRegression().fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    results["Linear Regression"] = {
        "model" : lr,
        "scaled": True,
        "RMSE"  : float(np.sqrt(mean_squared_error(y_test, lr_pred))),
        "MAE"   : float(mean_absolute_error(y_test, lr_pred)),
        "pred"  : lr_pred,
        "y_test": y_test.values,
    }

    # Random Forest
    rf      = RandomForestRegressor(
                  n_estimators=150, max_depth=8,
                  min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = {
        "model" : rf,
        "scaled": False,
        "RMSE"  : float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        "MAE"   : float(mean_absolute_error(y_test, rf_pred)),
        "pred"  : rf_pred,
        "y_test": y_test.values,
        "fi"    : dict(zip(FEATURES, rf.feature_importances_)),
    }

    # XGBoost (optional)
    if XGB_AVAILABLE:
        xgb_m = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0
        )
        xgb_m.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_m.predict(X_test)
        results["XGBoost"] = {
            "model" : xgb_m,
            "scaled": False,
            "RMSE"  : float(np.sqrt(mean_squared_error(y_test, xgb_pred))),
            "MAE"   : float(mean_absolute_error(y_test, xgb_pred)),
            "pred"  : xgb_pred,
            "y_test": y_test.values,
            "fi"    : dict(zip(FEATURES, xgb_m.feature_importances_)),
        }

    best_name = min(results, key=lambda k: results[k]["RMSE"])

    return {
        "results"   : results,
        "best_name" : best_name,
        "scaler"    : scaler,
    }


# ===========================================================================
# ⑥ RECOMMENDATION ENGINE
# ===========================================================================
def safe_norm(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise a 1-D numpy array to [0, 1].

    Accepts numpy arrays (not pandas Series) so there is zero risk of
    index-label misalignment during arithmetic.  A tiny epsilon in the
    denominator prevents division-by-zero when all values are identical.
    """
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def build_recommendations(df: pd.DataFrame, model_info: dict,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Scores every product by a weighted popularity function,
    predicts the optimal discount, and returns the top_n products.

    Root-cause fix
    --------------
    The original ValueError ("cannot reindex on an axis with duplicate
    labels") is triggered because ``df`` is a cleaned slice that may
    carry a non-contiguous or duplicated RangeIndex.  When pandas adds
    two Series it aligns them on their index labels first; duplicate
    labels make alignment ambiguous and raise the error.

    Fix strategy
    ------------
    1. ``reset_index(drop=True)`` immediately after the copy so df2 has
       a clean 0-based integer index with no duplicates.
    2. Extract every column used in the popularity formula as a raw
       ``numpy`` array via ``.to_numpy()``.  NumPy arithmetic is purely
       positional — it never looks at index labels — so misalignment is
       impossible by construction.
    3. ``safe_norm()`` now accepts and returns a numpy array, not a
       Series, for the same reason.
    """
    best_name = model_info["best_name"]
    best      = model_info["results"][best_name]
    model     = best["model"]
    scaler    = model_info["scaler"]

    # ── Build working copy with a guaranteed clean index ──────────────
    needed_cols = FEATURES + ["product_id", "category", "price", "season", "festival"]
    available   = [c for c in needed_cols if c in df.columns]
    df2 = (
        df[available]
        .dropna(subset=FEATURES)
        .copy()
        .reset_index(drop=True)          # ← eliminates duplicate / sparse labels
    )

    # Defensive guard: surface any remaining duplicates loudly rather
    # than silently producing wrong results downstream.
    if df2.index.duplicated().any():
        raise AssertionError(
            "build_recommendations: df2 still has duplicate index labels after "
            "reset_index — this should never happen; please file a bug report."
        )

    if len(df2) == 0:
        st.warning("⚠️  No products remain after dropping NaN feature rows.")
        return pd.DataFrame()

    # ── Popularity score — all arithmetic is on numpy arrays ──────────
    # Extract as numpy so pandas never attempts index alignment.
    units       = df2["units_sold"].to_numpy(dtype=float)
    interact    = df2["interaction_score"].to_numpy(dtype=float)
    sentiment   = df2["sentiment_score"].to_numpy(dtype=float)
    festival    = df2["festival"].to_numpy(dtype=float)
    sales_val   = df2["sales_value"].to_numpy(dtype=float) if "sales_value" in df2.columns \
                  else units * df2["price"].to_numpy(dtype=float)

    pop_score = (
        0.40 * safe_norm(units)
        + 0.25 * safe_norm(interact)
        + 0.20 * safe_norm(sentiment + 1)   # shift to [0, 2] before normalising
        + 0.10 * festival
        + 0.05 * safe_norm(sales_val)
    )

    # Assign back as a plain numpy array — no index clash possible
    df2["pop_score"] = pop_score

    # ── Model prediction ───────────────────────────────────────────────
    X = df2[FEATURES].to_numpy(dtype=float)   # numpy → model is always safe
    if best.get("scaled"):
        X = scaler.transform(X)

    df2["predicted_discount"]       = np.clip(model.predict(X), 0, 50)
    df2["recommended_discount_pct"] = (df2["predicted_discount"] / 5).round() * 5
    df2["effective_price"]          = (
        df2["price"] * (1 - df2["recommended_discount_pct"] / 100)
    ).round(2)

    # ── Return top-N ───────────────────────────────────────────────────
    keep = ["product_id", "category", "price", "effective_price",
            "recommended_discount_pct", "sentiment_score",
            "pop_score", "units_sold", "season", "festival"]
    keep = [c for c in keep if c in df2.columns]

    top = (
        df2.nlargest(top_n, "pop_score")[keep]
           .reset_index(drop=True)
    )
    top.index += 1   # 1-based display index
    return top


# ===========================================================================
# ⑦ REVENUE SIMULATION
# ===========================================================================
def simulate_revenue(rec_df: pd.DataFrame, full_df: pd.DataFrame,
                     disc_elast: float) -> pd.DataFrame:
    """
    Estimates revenue impact of moving from current → recommended discounts
    using the estimated discount elasticity.
    """
    merged = rec_df.merge(
        full_df[["product_id", "discount", "units_sold"]],
        on="product_id", how="left", suffixes=("", "_orig")
    )
    for col in ["discount", "units_sold"]:
        orig = f"{col}_orig"
        if orig in merged.columns:
            merged[col] = merged[orig].combine_first(merged[col])
            merged.drop(columns=[orig], inplace=True)

    merged["base_revenue"]   = merged["price"] * merged["units_sold"]
    merged["disc_delta_pct"] = (
        (merged["recommended_discount_pct"] - merged["discount"])
        / merged["discount"].clip(lower=1)
    ).clip(-0.5, 0.5)
    merged["demand_lift"]    = disc_elast * merged["disc_delta_pct"]
    merged["opt_units"]      = (merged["units_sold"] * (1 + merged["demand_lift"])).clip(lower=0)
    merged["opt_price"]      = merged["price"] * (1 - merged["recommended_discount_pct"] / 100)
    merged["opt_revenue"]    = merged["opt_price"] * merged["opt_units"]
    return merged


# ===========================================================================
# ⑧ SINGLE-PRODUCT OPTIMIZER  (sidebar tool)
# ===========================================================================
def optimal_discount_for_product(
    price: float, category: str, season: str,
    festival: int, sentiment: float,
    units_sold: float, interact: float,
    model_info: dict, df: pd.DataFrame
) -> dict:
    """
    Given user-specified product attributes, predict the optimal discount
    and estimate revenue at several candidate discount levels.
    """
    best_name = model_info["best_name"]
    best      = model_info["results"][best_name]
    model     = best["model"]
    scaler    = model_info["scaler"]

    le    = LabelEncoder().fit(df["category"].unique())
    try:
        cat_enc = int(le.transform([category])[0])
    except ValueError:
        cat_enc = 0

    seas_enc   = SEASON_MAP.get(season.lower(), 2)
    sales_val  = price * units_sold

    row = np.array([[price, units_sold, interact, sentiment,
                     seas_enc, festival, cat_enc, sales_val]])
    if best.get("scaled"):
        row = scaler.transform(row)

    pred_disc = float(np.clip(model.predict(row)[0], 0, 50))
    opt_disc  = round(pred_disc / 5) * 5

    # Revenue curve over discount range
    disc_range = np.arange(0, 55, 5)
    revenues   = []
    for d in disc_range:
        demand_adj = units_sold * (1 + 0.45 * (d - opt_disc) / max(opt_disc, 1))
        rev = price * (1 - d / 100) * max(demand_adj, 0)
        revenues.append(rev)

    return {
        "predicted_discount": pred_disc,
        "optimal_discount"  : opt_disc,
        "effective_price"   : round(price * (1 - opt_disc / 100), 2),
        "disc_range"        : disc_range.tolist(),
        "revenues"          : revenues,
    }


# ===========================================================================
# ⑨ MAIN APP
# ===========================================================================
def main():
    # ── Header banner ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🛍️ DTI — Dynamic Discount Optimizer</h1>
        <p>Optimize product discounts using price sensitivity, demand modelling & ML</p>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Data source
        st.markdown("### 📂 Data Source")
        use_sample = st.radio(
            "Choose dataset",
            ["Use sample data (built-in)", "Upload my own CSV"],
            index=0
        ) == "Use sample data (built-in)"

        uploaded_file = None
        if not use_sample:
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                help="Must contain at least: price, discount, units_sold"
            )

        st.markdown("---")
        st.markdown("### 🤖 Model Settings")
        top_n = st.slider("Top N recommendations", 5, 50, 15, step=5)

        st.markdown("---")
        st.markdown("### 🔍 Filters")
        season_filter = st.selectbox(
            "Season filter", ["All"] + SEASON_NAMES
        )
        festival_only = st.checkbox("Festival period only", value=False)
        min_sentiment = st.slider("Min sentiment score", -1.0, 1.0, -1.0, 0.05)

        st.markdown("---")
        st.markdown("### 🎮 Simulation Mode")
        st.caption("Instantly see the effect of a custom discount.")
        sim_price   = st.number_input("Product price ($)",   10.0, 2000.0, 150.0, step=10.0)
        sim_disc    = st.slider("Try this discount (%)", 0, 60, 20)
        sim_units   = st.number_input("Base units sold", 1, 50000, 500, step=50)
        sim_elast   = st.slider("Discount elasticity", 0.0, 2.0, 0.45, 0.05)

        st.markdown("---")
        st.markdown("### 🛒 Product Optimizer")
        opt_price    = st.number_input("Price ($)",         10.0, 5000.0, 200.0, step=10.0)
        opt_category = st.selectbox("Category", CATEGORIES)
        opt_season   = st.selectbox("Season", SEASON_NAMES)
        opt_festival = st.checkbox("Is festival period?", value=False)
        opt_sentiment= st.slider("Sentiment score",  -1.0, 1.0, 0.3, 0.05)
        opt_units    = st.number_input("Units sold", 1, 100000, 300, step=50)
        opt_interact = st.slider("Interaction score", 0.5, 10.0, 3.0, 0.5)
        run_opt      = st.button("🎯 Find Optimal Discount", use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # LOAD & VALIDATE DATA
    # ──────────────────────────────────────────────────────────────────────
    if use_sample:
        with st.spinner("Generating sample dataset …"):
            raw_df = generate_sample_data(800)
        st.info("ℹ️  Using built-in synthetic sample data (800 products).")
    else:
        if uploaded_file is None:
            st.warning("👈 Please upload a CSV file to get started.")
            st.stop()
        try:
            raw_df = pd.read_csv(uploaded_file)
            ok, msg = validate_dataframe(raw_df)
            if not ok:
                st.error(f"❌ Invalid CSV: {msg}")
                st.stop()
            st.success(f"✅ File uploaded — {len(raw_df):,} rows detected.")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            st.stop()

    # ──────────────────────────────────────────────────────────────────────
    # PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────
    with st.spinner("Preprocessing data …"):
        df = preprocess(raw_df)

    if len(df) < 20:
        st.error("❌ After cleaning, fewer than 20 rows remain. Please provide more data.")
        st.stop()

    # ──────────────────────────────────────────────────────────────────────
    # ELASTICITY
    # ──────────────────────────────────────────────────────────────────────
    price_elast, disc_elast = estimate_elasticity(df)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL TRAINING
    # ──────────────────────────────────────────────────────────────────────
    df_hash = hash(df.to_json())    # lightweight cache key
    with st.spinner("Training ML models … ⏳"):
        model_info = train_models(df_hash, df)

    if not model_info:
        st.stop()

    best_name = model_info["best_name"]
    results   = model_info["results"]

    # ──────────────────────────────────────────────────────────────────────
    # RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────────
    with st.spinner("Computing recommendations …"):
        rec_df = build_recommendations(df, model_info, top_n=top_n)

    # Apply sidebar filters
    filtered = rec_df.copy()
    if season_filter != "All":
        filtered = filtered[filtered["season"] == season_filter]
    if festival_only:
        filtered = filtered[filtered["festival"] == 1]
    filtered = filtered[filtered["sentiment_score"] >= min_sentiment]

    # ──────────────────────────────────────────────────────────────────────
    # ① KPI ROW
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Products",   f"{len(df):,}")
    k2.metric("Avg Price",        f"${df['price'].mean():.0f}")
    k3.metric("Price Elasticity", f"{price_elast:+.3f}")
    k4.metric("Disc. Elasticity", f"{disc_elast:+.3f}")
    k5.metric("Best Model",       best_name,
              delta=f"RMSE {results[best_name]['RMSE']:.2f}")

    # ──────────────────────────────────────────────────────────────────────
    # ② SIMULATION MODE
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎮 Simulation Mode</div>', unsafe_allow_html=True)

    demand_adj = sim_units * (1 + sim_elast * (sim_disc / max(sim_disc, 1)) * 0.1)
    sim_revenue = sim_price * (1 - sim_disc / 100) * demand_adj
    base_revenue = sim_price * sim_units

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Effective Price",   f"${sim_price * (1 - sim_disc/100):.2f}",
              delta=f"-{sim_disc}%")
    s2.metric("Adj. Demand",       f"{demand_adj:,.0f}",
              delta=f"{demand_adj - sim_units:+.0f} units")
    s3.metric("Simulated Revenue", f"${sim_revenue:,.0f}",
              delta=f"${sim_revenue - base_revenue:+,.0f}")
    s4.metric("Base Revenue",      f"${base_revenue:,.0f}")

    # Revenue-vs-discount curve
    disc_vals    = np.arange(0, 61, 2)
    rev_vals     = [
        sim_price * (1 - d / 100)
        * sim_units * (1 + sim_elast * (d / max(sim_disc, 1)) * 0.1)
        for d in disc_vals
    ]
    sim_fig = px.line(
        x=disc_vals, y=rev_vals,
        labels={"x": "Discount %", "y": "Revenue ($)"},
        title="Revenue vs Discount % (simulation)",
        color_discrete_sequence=["#6366f1"]
    )
    sim_fig.add_vline(x=sim_disc, line_dash="dash", line_color="#ec4899",
                      annotation_text=f"Current: {sim_disc}%")
    sim_fig.update_layout(margin=dict(t=40, b=20), height=320)
    st.plotly_chart(sim_fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ③ EDA CHARTS
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        seas_df = df.groupby("season")["units_sold"].sum().reset_index()
        seas_df["season"] = pd.Categorical(seas_df["season"], SEASON_NAMES, ordered=True)
        seas_df = seas_df.sort_values("season")
        fig_sea = px.bar(
            seas_df, x="season", y="units_sold",
            color="season", title="Units Sold by Season",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_sea.update_layout(showlegend=False, margin=dict(t=40, b=20), height=300)
        st.plotly_chart(fig_sea, use_container_width=True)

    with c2:
        fest_df = df.groupby("festival")["units_sold"].mean().reset_index()
        fest_df["period"] = fest_df["festival"].map({0: "Non-Festival", 1: "Festival"})
        fig_fest = px.pie(
            fest_df, names="period", values="units_sold",
            title="Avg Sales: Festival vs Normal",
            color_discrete_sequence=["#4ade80", "#f97316"]
        )
        fig_fest.update_layout(margin=dict(t=40, b=20), height=300)
        st.plotly_chart(fig_fest, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Price vs units sold
        samp = df.sample(min(500, len(df)), random_state=SEED)
        fig_pv = px.scatter(
            samp, x="price", y="units_sold",
            color="category", opacity=0.6,
            title="Price vs Units Sold",
            labels={"price": "Price ($)", "units_sold": "Units Sold"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pv.update_layout(margin=dict(t=40, b=20), height=320)
        st.plotly_chart(fig_pv, use_container_width=True)

    with c4:
        # Discount vs units sold
        fig_dv = px.scatter(
            samp, x="discount", y="units_sold",
            color="sentiment_score",
            color_continuous_scale="RdYlGn",
            title="Discount % vs Units Sold",
            labels={"discount": "Discount %", "units_sold": "Units Sold"},
            opacity=0.65
        )
        fig_dv.update_layout(margin=dict(t=40, b=20), height=320)
        st.plotly_chart(fig_dv, use_container_width=True)

    # Correlation heatmap
    corr_cols = [c for c in ["price", "discount", "units_sold",
                              "sentiment_score", "interaction_score",
                              "festival", "season_enc"] if c in df.columns]
    corr = df[corr_cols].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(margin=dict(t=40, b=20), height=350)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ④ ELASTICITY CHARTS
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Price & Discount Elasticity</div>',
                unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        lp = np.log1p(df["price"])
        lu = np.log1p(df["units_sold"])
        x_line = np.linspace(lp.min(), lp.max(), 100)
        m = LinearRegression().fit(lp.values.reshape(-1, 1), lu)
        y_line = m.predict(x_line.reshape(-1, 1))
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=lp, y=lu, mode="markers",
            marker=dict(size=4, color="#6366f1", opacity=0.4), name="Data"
        ))
        fig_pe.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="#ec4899", width=2),
            name=f"Elasticity = {price_elast:.3f}"
        ))
        fig_pe.update_layout(
            title="Log(Price) vs Log(Demand)",
            xaxis_title="Log(Price)", yaxis_title="Log(Units Sold)",
            margin=dict(t=40, b=20), height=320
        )
        st.plotly_chart(fig_pe, use_container_width=True)

    with e2:
        ld = np.log1p(df["discount"].clip(lower=0.01))
        lu = np.log1p(df["units_sold"])
        x_line2 = np.linspace(ld.min(), ld.max(), 100)
        m2 = LinearRegression().fit(ld.values.reshape(-1, 1), lu)
        y_line2 = m2.predict(x_line2.reshape(-1, 1))
        fig_de = go.Figure()
        fig_de.add_trace(go.Scatter(
            x=ld, y=lu, mode="markers",
            marker=dict(size=4, color="#f97316", opacity=0.4), name="Data"
        ))
        fig_de.add_trace(go.Scatter(
            x=x_line2, y=y_line2, mode="lines",
            line=dict(color="#22c55e", width=2),
            name=f"Elasticity = {disc_elast:.3f}"
        ))
        fig_de.update_layout(
            title="Log(Discount) vs Log(Demand)",
            xaxis_title="Log(Discount %)", yaxis_title="Log(Units Sold)",
            margin=dict(t=40, b=20), height=320
        )
        st.plotly_chart(fig_de, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ⑤ MODEL EVALUATION
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🤖 Model Evaluation</div>',
                unsafe_allow_html=True)

    metrics_df = pd.DataFrame(
        {n: {"RMSE": v["RMSE"], "MAE": v["MAE"]} for n, v in results.items()}
    ).T.reset_index().rename(columns={"index": "Model"})

    m1, m2 = st.columns(2)
    with m1:
        fig_met = px.bar(
            metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value"),
            x="Model", y="Value", color="Metric", barmode="group",
            title="RMSE & MAE by Model",
            color_discrete_sequence=["#6366f1", "#f97316"]
        )
        fig_met.update_layout(margin=dict(t=40, b=20), height=320)
        st.plotly_chart(fig_met, use_container_width=True)

    with m2:
        best_res = results[best_name]
        fig_pa = px.scatter(
            x=best_res["y_test"], y=best_res["pred"],
            labels={"x": "Actual Discount %", "y": "Predicted Discount %"},
            title=f"{best_name}: Predicted vs Actual",
            opacity=0.5,
            color_discrete_sequence=["#a855f7"]
        )
        mn = min(best_res["y_test"].min(), best_res["pred"].min())
        mx = max(best_res["y_test"].max(), best_res["pred"].max())
        fig_pa.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                         line=dict(color="red", dash="dash"))
        fig_pa.update_layout(margin=dict(t=40, b=20), height=320)
        st.plotly_chart(fig_pa, use_container_width=True)

    # Feature importances (RF or XGB only)
    fi_key = "XGBoost" if "XGBoost" in results else "Random Forest"
    if "fi" in results.get(fi_key, {}):
        fi_s = pd.Series(results[fi_key]["fi"]).sort_values()
        fig_fi = px.bar(
            x=fi_s.values, y=fi_s.index, orientation="h",
            title=f"Feature Importances — {fi_key}",
            labels={"x": "Importance", "y": "Feature"},
            color=fi_s.values,
            color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(margin=dict(t=40, b=20), height=320,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ⑥ RECOMMENDATIONS TABLE
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎁 Top Product Recommendations</div>',
                unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Products shown",     len(filtered))
    r2.metric("Avg sentiment",      f"{filtered['sentiment_score'].mean():.2f}")
    r3.metric("Avg optimal disc.",  f"{filtered['recommended_discount_pct'].mean():.1f}%")
    r4.metric("Avg effective price",f"${filtered['effective_price'].mean():.2f}")

    disp_cols = [c for c in ["product_id", "category", "price",
                              "effective_price", "recommended_discount_pct",
                              "sentiment_score", "pop_score", "units_sold",
                              "season", "festival"] if c in filtered.columns]
    fmt = {
        "price"                    : "${:.2f}",
        "effective_price"          : "${:.2f}",
        "recommended_discount_pct" : "{:.0f}%",
        "sentiment_score"          : "{:.3f}",
        "pop_score"                : "{:.4f}",
        "units_sold"               : "{:,.0f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in disp_cols}

    styled = (
        filtered[disp_cols]
        .style
        .format(fmt)
        .background_gradient(
            subset=["recommended_discount_pct"] if "recommended_discount_pct" in disp_cols else [],
            cmap="YlOrRd"
        )
        .background_gradient(
            subset=["sentiment_score"] if "sentiment_score" in disp_cols else [],
            cmap="RdYlGn"
        )
    )
    st.dataframe(styled, use_container_width=True, height=420)

    # Download button
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download recommendations CSV",
        data=csv_bytes,
        file_name="discount_recommendations.csv",
        mime="text/csv"
    )

    # ── Recommendation charts ──────────────────────────────────────────
    rc1, rc2 = st.columns(2)
    with rc1:
        top15 = filtered.head(15)
        fig_rb = px.bar(
            top15, x="product_id", y="recommended_discount_pct",
            color="sentiment_score", color_continuous_scale="RdYlGn",
            title="Recommended Discount by Product (top 15)",
            labels={"recommended_discount_pct": "Optimal Disc. %",
                    "product_id": "Product"}
        )
        fig_rb.update_layout(margin=dict(t=40, b=20), height=340,
                              xaxis_tickangle=-45)
        st.plotly_chart(fig_rb, use_container_width=True)

    with rc2:
        fig_bub = px.scatter(
            top15, x="price", y="sentiment_score",
            size="recommended_discount_pct", color="pop_score",
            hover_data=["product_id", "category"],
            title="Price vs Sentiment (bubble = discount)",
            color_continuous_scale="Viridis",
            labels={"price": "Price ($)", "sentiment_score": "Sentiment"}
        )
        fig_bub.update_layout(margin=dict(t=40, b=20), height=340)
        st.plotly_chart(fig_bub, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ⑦ REVENUE IMPACT
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💰 Revenue Impact Simulation</div>',
                unsafe_allow_html=True)

    rev_df = simulate_revenue(filtered, df, disc_elast)

    total_base = rev_df["base_revenue"].sum()
    total_opt  = rev_df["opt_revenue"].sum()
    lift_pct   = (total_opt - total_base) / max(total_base, 1) * 100
    demand_gain= (rev_df["opt_units"] - rev_df["units_sold"]).clip(lower=0) * rev_df["price"]
    disc_cost  = rev_df["opt_units"] * (rev_df["opt_price"] - rev_df["price"])

    ri1, ri2, ri3, ri4 = st.columns(4)
    ri1.metric("Baseline Revenue",   f"${total_base:,.0f}")
    ri2.metric("Optimised Revenue",  f"${total_opt:,.0f}",
               delta=f"{lift_pct:+.1f}%")
    ri3.metric("Demand Lift",        f"${demand_gain.sum():,.0f}")
    ri4.metric("Discount Cost",      f"${disc_cost.sum():,.0f}")

    # Waterfall
    bars       = [total_base, demand_gain.sum(), disc_cost.sum(), total_opt]
    categories = ["Baseline", "Demand Lift", "Price Reduction", "Optimised"]
    colors     = ["#6366f1", "#22c55e", "#f43f5e", "#f97316"]

    fig_wf = go.Figure(go.Bar(
        x=categories,
        y=[abs(b) for b in bars],
        marker_color=colors,
        text=[f"${abs(b)/1e3:.1f}K" for b in bars],
        textposition="outside"
    ))
    fig_wf.update_layout(
        title="Revenue Impact Waterfall",
        yaxis_title="Revenue ($)",
        margin=dict(t=50, b=20),
        height=360
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ⑧ SINGLE-PRODUCT OPTIMIZER
    # ──────────────────────────────────────────────────────────────────────
    if run_opt:
        st.markdown('<div class="section-title">🎯 Single-Product Optimization Result</div>',
                    unsafe_allow_html=True)
        with st.spinner("Computing optimal discount …"):
            opt_result = optimal_discount_for_product(
                price    = opt_price,
                category = opt_category,
                season   = opt_season,
                festival = int(opt_festival),
                sentiment= opt_sentiment,
                units_sold=opt_units,
                interact = opt_interact,
                model_info= model_info,
                df       = df,
            )

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Recommended Discount",  f"{opt_result['optimal_discount']:.0f}%")
        o2.metric("Effective Price",        f"${opt_result['effective_price']:.2f}")
        o3.metric("Original Price",         f"${opt_price:.2f}")
        o4.metric("Saving",                 f"${opt_price - opt_result['effective_price']:.2f}")

        fig_rev_curve = px.line(
            x=opt_result["disc_range"],
            y=opt_result["revenues"],
            labels={"x": "Discount %", "y": "Estimated Revenue ($)"},
            title="Revenue vs Discount % for this Product",
            color_discrete_sequence=["#6366f1"]
        )
        fig_rev_curve.add_vline(
            x=opt_result["optimal_discount"],
            line_dash="dash", line_color="#ec4899",
            annotation_text=f"Optimal: {opt_result['optimal_discount']:.0f}%"
        )
        fig_rev_curve.update_layout(margin=dict(t=50, b=20), height=340)
        st.plotly_chart(fig_rev_curve, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # ⑨ RAW DATA PREVIEW
    # ──────────────────────────────────────────────────────────────────────
    with st.expander("📋 View preprocessed dataset"):
        st.dataframe(df.head(200), use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🛍️ DTI Discount Optimizer • Built with Streamlit, scikit-learn & Plotly  "
        "• Model: " + best_name +
        f"  • Data: {len(df):,} products"
    )


# ===========================================================================
if __name__ == "__main__":
    main()
