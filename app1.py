# streamlit_app.py
# Lancer : streamlit run streamlit_app.py
# Prérequis : pip install streamlit pandas numpy matplotlib statsmodels scikit-learn openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error


st.set_page_config(
    page_title="Prépa examen — ARMA / ARIMA / SARIMAX / SARIMAX-X",
    layout="wide"
)

# -----------------------------
# Robust file reader (CSV/Excel/JSON/TXT)
# -----------------------------
def read_data_robust(file) -> pd.DataFrame:
    """
    Lecture robuste multi-formats :
    - CSV (séparateurs/encodages variés)
    - Excel (.xlsx/.xls) -> nécessite openpyxl
    - JSON
    - TXT/TSV
    Retourne un DataFrame.
    """
    name = file.name.lower()

    # Excel
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(file)
        except ImportError:
            st.error("Lecture Excel impossible : dépendance 'openpyxl' manquante.")
            st.info("Installe :  python -m pip install openpyxl")
            st.stop()
        df.columns = df.columns.astype(str).str.strip()
        return df

    # JSON
    if name.endswith(".json"):
        df = pd.read_json(file)
        df.columns = df.columns.astype(str).str.strip()
        return df

    # CSV / TXT (encodages + séparateurs)
    encodings = ["utf-8", "latin1", "cp1252"]
    seps = [None, ";", ",", "\t", "|"]

    last_error = None
    for enc in encodings:
        for sep in seps:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, engine="python", encoding=enc)

                # Cas : tout est dans une seule colonne type "DATE;VALUE"
                if df.shape[1] == 1 and any(x in df.columns[0] for x in [";", ",", "\t", "|"]):
                    continue

                df.columns = df.columns.astype(str).str.strip()
                return df
            except Exception as e:
                last_error = e

    raise ValueError("Impossible de lire le fichier (format/encodage/séparateur).") from last_error


# -----------------------------
# Helpers
# -----------------------------
def infer_date_col(df: pd.DataFrame):
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "datetime", "timestamp", "time", "month", "ds") or "date" in cl:
            return c
    best, best_ratio = None, 0.0
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio and ratio > 0.6:
            best, best_ratio = c, ratio
    return best

def infer_value_col(df: pd.DataFrame, date_col: str | None):
    candidates = [c for c in df.columns if c != date_col]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        return max(numeric, key=lambda c: float(df[c].var(skipna=True)))
    return None

def _agg_frame_by_date(tmp: pd.DataFrame, date_col: str, cols: list[str], agg: str) -> pd.DataFrame:
    if agg == "sum":
        return tmp.groupby(date_col, as_index=False)[cols].sum()
    if agg == "median":
        return tmp.groupby(date_col, as_index=False)[cols].median()
    if agg == "last":
        return tmp.groupby(date_col, as_index=False)[cols].last()
    return tmp.groupby(date_col, as_index=False)[cols].mean()

def build_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str, agg: str) -> pd.Series:
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    tmp = _agg_frame_by_date(tmp, date_col, [value_col], agg)
    s = tmp.set_index(date_col)[value_col].sort_index()

    if freq != "Auto":
        s = s.asfreq(freq)
        s = s.interpolate(limit_direction="both")

    s.name = value_col
    return s

def build_exog(df: pd.DataFrame, date_col: str, exog_cols: list[str], freq: str, agg: str) -> pd.DataFrame:
    tmp = df[[date_col] + exog_cols].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)

    for c in exog_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = _agg_frame_by_date(tmp, date_col, exog_cols, agg)
    X = tmp.set_index(date_col)[exog_cols].sort_index()

    if freq != "Auto":
        X = X.asfreq(freq)
        X = X.interpolate(limit_direction="both")

    return X

def adf_test(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    stat, pval, lags, nobs, *_ = adfuller(s, autolag="AIC")
    return {"adf_stat": stat, "p_value": pval, "lags": lags, "nobs": nobs}

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def plot_one_series(s: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    s.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def plot_train_test_pred(train: pd.Series, test: pd.Series, pred: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test", color="black")
    pred.plot(ax=ax, label="Prévision", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

def safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default


# -----------------------------
# UI
# -----------------------------
st.title("Prépa examen — ARMA / ARIMA / SARIMAX (+ exogènes SARIMAX)")
st.caption("Upload → série → STL/ADF/ACF/PACF → modèle → prévision → RMSE")

uploaded = st.file_uploader(
    "1) Charger un fichier (CSV / Excel / JSON / TXT)",
    type=["csv", "xlsx", "xls", "json", "txt"]
)

use_demo = st.checkbox("Utiliser un dataset exemple (généré)", value=uploaded is None)

if use_demo:
    rng = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    t = np.arange(len(rng))
    np.random.seed(42)
    y = 50 + 0.01*t + 2*np.sin(2*np.pi*t/7) + 5*np.sin(2*np.pi*t/365.25) + np.random.randn(len(rng))
    promo = (np.random.rand(len(rng)) < 0.12).astype(int)
    temp = 15 + 10*np.sin(2*np.pi*t/365.25) + 1.5*np.random.randn(len(rng))
    df = pd.DataFrame({"date": rng, "value": y, "promo": promo, "temp": temp})
else:
    if uploaded is None:
        st.stop()
    df = read_data_robust(uploaded)

df.columns = df.columns.astype(str).str.strip()

with st.expander("Aperçu du dataset", expanded=True):
    st.dataframe(df.head(25), use_container_width=True)
    st.caption(f"{df.shape[0]} lignes × {df.shape[1]} colonnes")

# Colonnes date / valeur
date_guess = infer_date_col(df)
value_guess = infer_value_col(df, date_guess)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.1, 1.1])
with c1:
    date_col = st.selectbox(
        "2) Colonne date",
        options=df.columns.tolist(),
        index=df.columns.get_loc(date_guess) if date_guess in df.columns else 0
    )
with c2:
    candidates = [c for c in df.columns.tolist() if c != date_col]
    if not candidates:
        st.error("Aucune colonne valeur disponible. Vérifie ton fichier.")
        st.stop()
    idx = candidates.index(value_guess) if value_guess in candidates else 0
    value_col = st.selectbox("3) Colonne valeur à prédire (y)", options=candidates, index=idx)
with c3:
    freq = st.selectbox("4) Fréquence (si régulière)", options=["Auto", "D", "W-SUN", "MS", "M"], index=0)
    st.caption("D=jour, W-SUN=hebdo, MS=mensuel-début, M=mensuel-fin")
with c4:
    agg = st.selectbox("Agrégation si doublons de dates", options=["mean", "sum", "median", "last"], index=0)

# Série y
series = build_series(df, date_col, value_col, freq, agg)

if len(series) < 30:
    st.error("Série trop courte après nettoyage (>= 30 points recommandés).")
    st.stop()

# Exogènes (numériques uniquement)
st.subheader("Variables exogènes (option SARIMAX)")
exog_candidates = [
    c for c in df.columns
    if c not in [date_col, value_col] and pd.api.types.is_numeric_dtype(df[c])
]
use_exog = st.checkbox("Utiliser des variables exogènes (SARIMAX-X)", value=False)
exog_cols = []
if use_exog:
    if len(exog_candidates) == 0:
        st.warning("Aucune colonne numérique disponible pour exogènes.")
        use_exog = False
    else:
        exog_cols = st.multiselect("Choisir les colonnes exogènes (X)", exog_candidates)

X = None
if use_exog and len(exog_cols) > 0:
    X = build_exog(df, date_col, exog_cols, freq, agg)
    common_idx = series.index.intersection(X.index)
    series = series.loc[common_idx]
    X = X.loc[common_idx]
    if len(series) < 30:
        st.error("Après alignement y/X, série trop courte.")
        st.stop()

# -----------------------------
# Visualisation
# -----------------------------
st.header("A) Série originale")
plot_one_series(series, "Série originale", value_col)

# -----------------------------
# STL decomposition
# -----------------------------
st.header("B) Décomposition STL")
default_period = 7
if freq in ("MS", "M"):
    default_period = 12
elif freq == "W-SUN":
    default_period = 52
elif freq == "D":
    default_period = 7
period = st.number_input("Période saisonnière (STL period)", min_value=2, max_value=365, value=int(default_period), step=1)

try:
    stl = STL(series, period=int(period), robust=True).fit()
    fig, ax = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(series.index, series.values); ax[0].set_title("Original")
    ax[1].plot(stl.trend.index, stl.trend.values); ax[1].set_title("Trend")
    ax[2].plot(stl.seasonal.index, stl.seasonal.values); ax[2].set_title("Seasonal")
    ax[3].plot(stl.resid.index, stl.resid.values); ax[3].set_title("Residual")
    for a in ax:
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.warning(f"STL impossible : {e}")

# -----------------------------
# Stationarity - ADF
# -----------------------------
st.header("C) Stationnarité (ADF)")
adf = adf_test(series)
st.write(f"ADF={adf['adf_stat']:.4f} | p-value={adf['p_value']:.4g} | lags={adf['lags']} | nobs={adf['nobs']}")
if adf["p_value"] < 0.05:
    st.success("✅ Série stationnaire (seuil 5%). ARMA possible.")
else:
    st.warning("⚠️ Série NON stationnaire (seuil 5%). ARIMA/SARIMAX recommandé (différenciation).")

# -----------------------------
# ACF/PACF
# -----------------------------
st.header("D) ACF / PACF (choix p, q)")
lags_user = st.number_input("Nombre de lags (max)", min_value=5, max_value=200, value=40, step=5)
x = series.dropna()
max_lags_pacf = max(5, min(int(lags_user), (len(x)//2) - 1))
max_lags_acf = max(5, min(int(lags_user), len(x) - 1))

c_acf, c_pacf = st.columns(2)
with c_acf:
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    plot_acf(x, lags=max_lags_acf, ax=ax1)
    ax1.set_title(f"ACF (lags={max_lags_acf})")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True)
with c_pacf:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    plot_pacf(x, lags=max_lags_pacf, ax=ax2, method="ywm")
    ax2.set_title(f"PACF (lags={max_lags_pacf})")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

st.info("💡 PACF aide à choisir **p** ; ACF aide à choisir **q**. Si la série n'est pas stationnaire, préfère ACF/PACF sur une série différenciée.")

# -----------------------------
# Modeling
# -----------------------------
st.header("E) Modélisation + RMSE + Graph")

test_pct = st.slider("Taille du test (%)", 10, 40, 20, 5)
n_test = max(5, int(len(series) * test_pct / 100))

train = series.iloc[:-n_test]
test = series.iloc[-n_test:]

X_train = X_test = None
if X is not None:
    X_train = X.iloc[:-n_test]
    X_test = X.iloc[-n_test:]

st.write(f"Train: {len(train)} points | Test: {len(test)} points")

model_type = st.radio("Choisir modèle", ["ARMA (p,q)", "ARIMA (p,d,q)", "SARIMAX (p,d,q)(P,D,Q,s)"], horizontal=True)

colA, colB, colC, colD = st.columns(4)
with colA: p = safe_int(st.text_input("p", "1"), 1)
with colB: q = safe_int(st.text_input("q", "1"), 1)
with colC: d = safe_int(st.text_input("d", "1"), 1)
with colD: s = safe_int(st.text_input("s (SARIMAX)", str(int(period))), int(period))

P = safe_int(st.text_input("P (SARIMAX)", "1"), 1)
D = safe_int(st.text_input("D (SARIMAX)", "1"), 1)
Q = safe_int(st.text_input("Q (SARIMAX)", "1"), 1)

horizon = st.number_input("Horizon futur (optionnel)", min_value=0, max_value=500, value=30, step=1)
st.caption("⚠️ Pour une prévision futur SARIMAX avec exogènes, il faut X_future.")

# Optional: upload exog future
X_future = None
if use_exog and len(exog_cols) > 0 and horizon > 0:
    st.subheader("Exogènes futures (optionnel)")
    fut = st.file_uploader("Fichier exogènes futures (CSV/Excel/JSON/TXT)", type=["csv", "xlsx", "xls", "json", "txt"], key="exog_future")
    if fut is not None:
        df_fut = read_data_robust(fut)
        df_fut.columns = df_fut.columns.astype(str).str.strip()
        missing = [c for c in ([date_col] + exog_cols) if c not in df_fut.columns]
        if missing:
            st.error(f"Fichier exogènes futures invalide. Colonnes manquantes: {missing}")
        else:
            X_future = build_exog(df_fut, date_col, exog_cols, freq, agg)

run = st.button("🚀 Entraîner + prédire")

if run:
    try:
        if model_type == "ARMA (p,q)":
            model = ARIMA(train, order=(p, 0, q), trend="n")
            res = model.fit()
            pred_test = res.get_forecast(steps=len(test)).predicted_mean
            title = f"ARMA({p},{q})"
        elif model_type == "ARIMA (p,d,q)":
            model = ARIMA(train, order=(p, d, q), trend="n")
            res = model.fit()
            pred_test = res.get_forecast(steps=len(test)).predicted_mean
            title = f"ARIMA({p},{d},{q})"
        else:
            model = SARIMAX(
                train,
                exog=X_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            pred_test = res.get_forecast(steps=len(test), exog=X_test).predicted_mean
            title = f"SARIMAX({p},{d},{q})({P},{D},{Q},{s})"
            if X_train is not None:
                title += f" + exog={exog_cols}"

        pred_test = pd.Series(pred_test.values, index=test.index, name="prediction")
        score = rmse(test.values, pred_test.values)

        st.success("✅ Modèle entraîné")
        st.write(f"**RMSE (sur test)** = {score:.4f}")

        with st.expander("Résumé du modèle (summary)", expanded=False):
            st.text(res.summary().as_text())

        plot_train_test_pred(train, test, pred_test, f"{title} — Prévision sur test", value_col)

        # Prévision future
        if horizon > 0:
            st.subheader("Prévision future")
            if model_type.startswith("SARIMAX") and X is not None and len(exog_cols) > 0:
                if X_future is None:
                    st.warning("Prévision future SARIMAX avec exogènes impossible sans X_future. "
                               "Charge un fichier d'exogènes futures ou mets horizon=0.")
                else:
                    model_full = SARIMAX(
                        series,
                        exog=X,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res_full = model_full.fit(disp=False)

                    Xf = X_future.sort_index()
                    if len(Xf) < horizon:
                        st.error(f"X_future trop court : {len(Xf)} lignes, horizon={horizon}.")
                    else:
                        Xf = Xf.iloc[:horizon]
                        fut_pred = res_full.get_forecast(steps=horizon, exog=Xf).predicted_mean
                        fut_pred = pd.Series(fut_pred.values, index=Xf.index, name="forecast")

                        fig, ax = plt.subplots(figsize=(11, 4))
                        series.plot(ax=ax, label="Observé")
                        fut_pred.plot(ax=ax, label="Prévision future", linestyle="--")
                        ax.set_title(f"{title} — Prévision future (horizon={horizon})")
                        ax.set_xlabel("Date")
                        ax.set_ylabel(value_col)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig, clear_figure=True)
            else:
                if model_type == "ARMA (p,q)":
                    model_full = ARIMA(series, order=(p, 0, q), trend="n").fit()
                    fut = model_full.get_forecast(steps=horizon).predicted_mean
                elif model_type == "ARIMA (p,d,q)":
                    model_full = ARIMA(series, order=(p, d, q), trend="n").fit()
                    fut = model_full.get_forecast(steps=horizon).predicted_mean
                else:
                    model_full = SARIMAX(
                        series,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    fut = model_full.get_forecast(steps=horizon).predicted_mean

                if series.index.freq is not None:
                    future_index = pd.date_range(series.index[-1] + series.index.freq, periods=horizon, freq=series.index.freq)
                else:
                    inferred = pd.infer_freq(series.index)
                    future_index = pd.date_range(series.index[-1], periods=horizon+1, freq=inferred)[1:] if inferred else pd.RangeIndex(1, horizon+1)

                fut_pred = pd.Series(fut.values, index=future_index, name="forecast")

                fig, ax = plt.subplots(figsize=(11, 4))
                series.plot(ax=ax, label="Observé")
                fut_pred.plot(ax=ax, label="Prévision future", linestyle="--")
                ax.set_title(f"{title} — Prévision future (horizon={horizon})")
                ax.set_xlabel("Date")
                ax.set_ylabel(value_col)
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig, clear_figure=True)

        st.subheader("Explications (à mettre dans le livrable)")
        st.markdown(f"""
- **ADF** : p-value = **{adf['p_value']:.4g}** (seuil 5%).  
  - p-value < 0.05 → stationnaire → ARMA possible  
  - sinon → ARIMA/SARIMAX (différenciation d, + saisonnalité)
- **ACF/PACF** : aide à choisir **q/p**.
- **SARIMAX-X (exog)** : ajoute des variables explicatives **X(t)** (ex : promo, météo, etc.).  
  ➜ Pour prévoir dans le futur, il faut **X_future** sur l’horizon.
- **RMSE** : mesure d’erreur sur le test (plus petit = meilleur).
        """)

    except Exception as e:
        st.error(f"Erreur pendant l'entraînement/prédiction : {e}")
        st.info("Astuce : réduis p/q, ajuste d, vérifie la fréquence, et pour SARIMAX-X assure-toi que X est aligné et sans NaN.")
