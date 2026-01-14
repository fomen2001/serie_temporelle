# streamlit_app.py
# Lancer : streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Projet Séries Temporelles — ARIMA / SARIMA",
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

    # CSV / TXT
    encodings = ["utf-8", "latin1", "cp1252"]
    seps = [None, ";", ",", "\t", "|"]

    last_error = None
    for enc in encodings:
        for sep in seps:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, engine="python", encoding=enc)

                # si 1 seule colonne du style "DATE;VALUE" -> mauvais split
                if df.shape[1] == 1 and any(x in df.columns[0] for x in [";", ",", "\t", "|"]):
                    continue

                df.columns = df.columns.astype(str).str.strip()
                return df
            except Exception as e:
                last_error = e

    raise ValueError("Impossible de lire le fichier (format/encodage/separateur).") from last_error


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

def build_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str, agg: str) -> pd.Series:
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    # Agrégation si dates dupliquées (panel data)
    if agg == "mean":
        tmp = tmp.groupby(date_col, as_index=False)[value_col].mean()
    elif agg == "sum":
        tmp = tmp.groupby(date_col, as_index=False)[value_col].sum()
    elif agg == "median":
        tmp = tmp.groupby(date_col, as_index=False)[value_col].median()
    elif agg == "last":
        tmp = tmp.groupby(date_col, as_index=False)[value_col].last()
    else:
        tmp = tmp.groupby(date_col, as_index=False)[value_col].mean()

    s = tmp.set_index(date_col)[value_col].sort_index()

    # Fréquence
    if freq != "Auto":
        s = s.asfreq(freq)
        s = s.interpolate(limit_direction="both")
    return s

def adf_test(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    stat, pval, lags, nobs, *_ = adfuller(s, autolag="AIC")
    return stat, pval, lags, nobs

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def plot_series(s: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    s.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def plot_forecast(history: pd.Series, forecast: pd.Series, title: str, ylabel: str, zoom_last: int | None = None):
    fig, ax = plt.subplots(figsize=(11, 4))

    if zoom_last is not None and zoom_last > 0 and len(history) > zoom_last:
        history_tail = history.iloc[-zoom_last:]
        history_tail.plot(ax=ax, label="Observé (zoom)")
    else:
        history.plot(ax=ax, label="Observé")

    forecast.plot(ax=ax, label="Prévision", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# UI
# -----------------------------
st.title("Projet — Séries Temporelles : ARIMA / SARIMA (Streamlit)")
st.caption("Upload → Visualisation → STL → Stationnarité (ADF) → Modèle → Prévision → Comparaison + RMSE")

uploaded = st.file_uploader(
    "1) Charger un fichier (CSV / Excel / JSON / TXT)",
    type=["csv", "xlsx", "xls", "json", "txt"]
)

use_demo = st.checkbox("Utiliser un dataset exemple (généré)", value=uploaded is None)

if use_demo:
    # Dataset demo (journalier avec tendance + saisonnalité hebdo + bruit)
    rng = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    t = np.arange(len(rng))
    np.random.seed(42)
    y = 10 + 0.002*t + 0.7*np.sin(2*np.pi*t/7) + 0.2*np.random.randn(len(rng))
    df = pd.DataFrame({"date": rng, "value": y})
else:
    if uploaded is None:
        st.stop()
    df = read_data_robust(uploaded)

df.columns = df.columns.astype(str).str.strip()

with st.expander("Aperçu du dataset", expanded=True):
    st.dataframe(df.head(25), use_container_width=True)
    st.caption(f"{df.shape[0]} lignes × {df.shape[1]} colonnes")

# Choix colonnes
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
    if len(candidates) == 0:
        st.error("Aucune colonne valeur disponible. Vérifie le fichier.")
        st.stop()
    idx = candidates.index(value_guess) if value_guess in candidates else 0
    value_col = st.selectbox("3) Colonne valeur à prédire", options=candidates, index=idx)

with c3:
    freq = st.selectbox("4) Fréquence (si régulière)", options=["Auto", "D", "W-SUN", "MS", "M"], index=0)
    st.caption("D=jour, W-SUN=hebdo, MS=mensuel-début, M=mensuel-fin")

with c4:
    agg = st.selectbox("Agrégation si doublons de dates", options=["mean", "sum", "median", "last"], index=0)
    st.caption("Panel data: plusieurs lignes par date")

# Construire la série
series = build_series(df, date_col, value_col, freq, agg)

if len(series) < 30:
    st.error("Série trop courte après nettoyage (>= 30 points recommandés).")
    st.stop()

# -----------------------------
# 2) Visualiser la série originale
# -----------------------------
st.header("2) Visualisation")
plot_series(series, "Série originale", value_col)

# -----------------------------
# 2b) STL decomposition
# -----------------------------
st.header("3) Décomposition STL (tendance / saisonnalité / résidu)")

# Choix période (important pour STL)
# Exemple : jour -> 7, mensuel -> 12, hebdo -> 52
default_period = 7
if freq in ("MS", "M"):
    default_period = 12
elif freq == "W-SUN":
    default_period = 52
elif freq == "D":
    default_period = 7

period = st.number_input("Période saisonnière pour STL (period)", min_value=2, max_value=365, value=int(default_period), step=1)

try:
    stl = STL(series, period=int(period), robust=True).fit()
    fig, ax = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(series.index, series.values)
    ax[0].set_title("Original")
    ax[1].plot(stl.trend.index, stl.trend.values)
    ax[1].set_title("Trend")
    ax[2].plot(stl.seasonal.index, stl.seasonal.values)
    ax[2].set_title("Seasonal")
    ax[3].plot(stl.resid.index, stl.resid.values)
    ax[3].set_title("Residual")
    for a in ax:
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.warning(f"STL impossible avec ces paramètres : {e}")

# -----------------------------
# 2c) Stationarity - ADF
# -----------------------------
st.header("4) Stationnarité (ADF)")
adf_stat, pval, lags, nobs = adf_test(series)
st.write(f"**ADF stat** = {adf_stat:.4f} | **p-value** = {pval:.4g} | lags={lags} | nobs={nobs}")

if pval < 0.05:
    st.success("✅ Série stationnaire (seuil 5%).")
else:
    st.warning("⚠️ Série NON stationnaire (seuil 5%). ARIMA/SARIMA recommandé (différenciation).")

# -----------------------------
# 5) Choix du modèle + entrainement
# -----------------------------
st.header("5) Modélisation & prévision")

model_choice = st.radio("Choisir un modèle", ["ARIMA", "SARIMA"], horizontal=True)

# Split train/test pour RMSE (évaluation)
test_pct = st.slider("Taille du test (%) pour calcul RMSE", 10, 40, 20, 5)
n_test = max(5, int(len(series) * test_pct / 100))
train = series.iloc[:-n_test]
test = series.iloc[-n_test:]

st.caption(f"Train: {len(train)} points | Test: {len(test)} points")

# Paramètres modèle
cA, cB, cC, cD = st.columns(4)
with cA:
    p = st.number_input("p", min_value=0, max_value=10, value=1, step=1)
with cB:
    d = st.number_input("d", min_value=0, max_value=3, value=1, step=1)
with cC:
    q = st.number_input("q", min_value=0, max_value=10, value=1, step=1)
with cD:
    horizon = st.number_input("Horizon de prévision (pas futurs)", min_value=1, max_value=500, value=30, step=1)

if model_choice == "SARIMA":
    st.subheader("Paramètres saisonniers (SARIMA)")
    cS1, cS2, cS3, cS4 = st.columns(4)
    with cS1:
        P = st.number_input("P", min_value=0, max_value=5, value=1, step=1)
    with cS2:
        D = st.number_input("D", min_value=0, max_value=2, value=1, step=1)
    with cS3:
        Q = st.number_input("Q", min_value=0, max_value=5, value=1, step=1)
    with cS4:
        s = st.number_input("s (période saisonnière)", min_value=2, max_value=365, value=int(period), step=1)

run = st.button("🚀 Entraîner + prédire")

if run:
    try:
        # ---- Fit on train, evaluate on test for RMSE
        if model_choice == "ARIMA":
            model = ARIMA(train, order=(int(p), int(d), int(q)), trend="n")
            res = model.fit()
            pred_test = res.get_forecast(steps=len(test)).predicted_mean
            title_model = f"ARIMA({int(p)},{int(d)},{int(q)})"
        else:
            model = SARIMAX(
                train,
                order=(int(p), int(d), int(q)),
                seasonal_order=(int(P), int(D), int(Q), int(s)),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            pred_test = res.get_forecast(steps=len(test)).predicted_mean
            title_model = f"SARIMA({int(p)},{int(d)},{int(q)})({int(P)},{int(D)},{int(Q)},{int(s)})"

        pred_test = pd.Series(pred_test.values, index=test.index, name="pred_test")
        score = rmse(test.values, pred_test.values)

        st.success("✅ Modèle entraîné")
        st.write(f"**RMSE (sur le test)** = {score:.4f}")

        with st.expander("Résumé du modèle (summary)", expanded=False):
            st.text(res.summary().as_text())

        # ---- Forecast future on full series
        if model_choice == "ARIMA":
            model_full = ARIMA(series, order=(int(p), int(d), int(q)), trend="n")
            res_full = model_full.fit()
        else:
            model_full = SARIMAX(
                series,
                order=(int(p), int(d), int(q)),
                seasonal_order=(int(P), int(D), int(Q), int(s)),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res_full = model_full.fit(disp=False)

        future_pred = res_full.get_forecast(steps=int(horizon)).predicted_mean

        # Construire index futur (si fréquence connue)
        if series.index.freq is not None:
            future_index = pd.date_range(series.index[-1] + series.index.freq, periods=int(horizon), freq=series.index.freq)
        else:
            # fallback : on essaie d'inférer
            inferred = pd.infer_freq(series.index)
            if inferred is None:
                # dernier recours : pas de freq, on met un index "step"
                future_index = pd.RangeIndex(start=1, stop=int(horizon)+1, step=1)
            else:
                future_index = pd.date_range(series.index[-1], periods=int(horizon)+1, freq=inferred)[1:]

        future_pred = pd.Series(future_pred.values, index=future_index, name="forecast")

        # Graph: comparaison test vs pred_test
        st.subheader("6) Comparaison Observé vs Prédit (sur le test)")
        plot_forecast(train, pred_test, f"{title_model} — Prédiction sur période test", value_col, zoom_last=None)

        fig_cmp, ax = plt.subplots(figsize=(11, 4))
        train.plot(ax=ax, label="Train")
        test.plot(ax=ax, label="Test", color="black")
        pred_test.plot(ax=ax, label="Prédit (test)", linestyle="--")
        ax.set_title("Superposition Observé vs Prédit (test)")
        ax.set_xlabel("Date")
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig_cmp, clear_figure=True)

        # Graph: original + future forecast
        st.subheader("7) Prévision sur horizon futur")
        plot_forecast(series, future_pred, f"{title_model} — Prévision future (horizon={int(horizon)})", value_col, zoom_last=200)

        st.subheader("Explications (à coller dans le PDF)")
        st.markdown(f"""
- **STL** : décomposition en **tendance**, **saisonnalité**, **résidu** pour analyser la structure de la série.
- **ADF** : p-value = **{pval:.4g}**  
  - p-value < 0.05 ⇒ série stationnaire  
  - p-value ≥ 0.05 ⇒ série non stationnaire ⇒ **différenciation (d)** souvent nécessaire (ARIMA/SARIMA)
- **Modèle choisi** : **{title_model}**
- **RMSE** : mesure d’erreur sur la période de test (plus petit = meilleur).
- **Horizon** : le modèle prévoit **{int(horizon)}** pas de temps dans le futur.
        """)
    except Exception as e:
        st.error(f"Erreur pendant l'entraînement / prévision : {e}")
        st.info("Astuce : réduis (p,q), vérifie la fréquence, ajuste (d) et la période saisonnière s.")
