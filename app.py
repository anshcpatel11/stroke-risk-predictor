"""
app.py — Stroke Risk Prediction Streamlit App
Trains the model at startup (cached) to avoid sklearn version mismatch with pickle.

Usage:
    streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="wide")

def engineer_features(df):
    df = df.copy()
    df["glucose_risk"] = pd.cut(
        df["avg_glucose_level"], bins=[0, 100, 125, 999],
        labels=["Normal", "Prediabetic", "Diabetic"]
    ).astype(str)
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 999],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    ).astype(str)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 40, 55, 65, 999],
        labels=["Under40", "40to55", "55to65", "Over65"]
    ).astype(str)
    df["comorbidity_score"]   = df["hypertension"].astype(int) + df["heart_disease"].astype(int)
    df["age_x_hypertension"]  = df["age"] * df["hypertension"]
    df["age_x_heart_disease"] = df["age"] * df["heart_disease"]
    return df

@st.cache_resource(show_spinner="Training model... (first load only, ~30 seconds)")
def load_model():
    for p in [Path("data/healthcare-dataset-stroke-data.csv"),
              Path("../data/healthcare-dataset-stroke-data.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        st.error("data/healthcare-dataset-stroke-data.csv not found.")
        st.stop()

    df = df[df["gender"] != "Other"].reset_index(drop=True)
    df = engineer_features(df)
    X  = df.drop(columns=["id", "stroke"])
    y  = df["stroke"]

    num_feats = ["age","avg_glucose_level","bmi","comorbidity_score","age_x_hypertension","age_x_heart_disease"]
    bin_feats = ["hypertension","heart_disease"]
    cat_feats = ["gender","ever_married","work_type","Residence_type","smoking_status","glucose_risk","bmi_category","age_group"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_feats),
        ("bin", Pipeline([("i", SimpleImputer(strategy="most_frequent"))]), bin_feats),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train_proc = preprocessor.fit_transform(X_train)

    y_arr = np.array(y_train)
    X_maj, X_min = X_train_proc[y_arr==0], X_train_proc[y_arr==1]
    target = len(X_maj) // 4
    X_min_over = resample(X_min, n_samples=target, replace=True, random_state=RANDOM_STATE)
    X_bal = np.vstack([X_maj, X_min_over])
    y_bal = np.concatenate([np.zeros(len(X_maj), dtype=int), np.ones(target, dtype=int)])
    idx   = np.random.RandomState(RANDOM_STATE).permutation(len(X_bal))
    X_bal, y_bal = X_bal[idx], y_bal[idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = StackingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ("rf",  RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                           class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05,
                                                   class_weight="balanced", random_state=RANDOM_STATE)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=cv, passthrough=False, n_jobs=-1
    )
    model.fit(X_bal, y_bal)

    X_test_proc = preprocessor.transform(X_test)
    probas = model.predict_proba(X_test_proc)[:, 1]
    ts = np.linspace(0.05, 0.95, 200)
    f1s = [f1_score(np.array(y_test), (probas >= t).astype(int), zero_division=0) for t in ts]
    return preprocessor, model, float(ts[np.argmax(f1s)])

def risk_tier(p):
    if p < 0.20:  return "Low",      "#27ae60", "✅"
    elif p < 0.50: return "Moderate", "#f39c12", "⚠️"
    else:          return "High",     "#e74c3c", "🚨"

def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=18, solid_capstyle="round")
    _, color, _ = risk_tier(prob)
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=18, solid_capstyle="round")
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.6*np.cos(angle), 0.6*np.sin(angle)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))
    ax.plot(0, 0, "o", color="#2c3e50", markersize=8)
    ax.text(0, -0.25, f"{prob:.1%}", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.4, 1.1); ax.axis("off")
    return fig

def draw_factors(row):
    factors = {
        "Age":           min(row["age"] / 100, 1.0),
        "Avg Glucose":   min((row["avg_glucose_level"] - 50) / 250, 1.0),
        "BMI":           min((row["bmi"] - 10) / 50, 1.0),
        "Hypertension":  float(row["hypertension"]),
        "Heart Disease": float(row["heart_disease"]),
        "Comorbidities": (row["hypertension"] + row["heart_disease"]) / 2,
    }
    vals = list(factors.values())
    colors = ["#e74c3c" if v > 0.6 else "#f39c12" if v > 0.3 else "#27ae60" for v in vals]
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    ax.barh(list(factors.keys()), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1); ax.set_xlabel("Relative Risk Level", fontsize=9)
    ax.set_title("Key Risk Factors", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

preprocessor, model, threshold = load_model()

st.title("🧠 Stroke Risk Predictor")
st.markdown("Enter patient information below to estimate stroke risk. This tool is for **educational and research purposes only** and is not a medical device.")
st.divider()

col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("Patient Information")
    c1, c2 = st.columns(2)
    with c1:
        age          = st.slider("Age", 1, 100, 55)
        gender       = st.selectbox("Gender", ["Male", "Female"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type    = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
    with c2:
        residence     = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking       = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])
        hypertension  = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
    st.markdown("**Clinical Measurements**")
    c3, c4 = st.columns(2)
    with c3:
        glucose = st.slider("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=0.5)
    with c4:
        bmi_val = st.slider("BMI", 10.0, 60.0, 25.0, step=0.1)
    predict_btn = st.button("Calculate Risk", type="primary", use_container_width=True)

with col_results:
    st.subheader("Risk Assessment")
    if predict_btn:
        row = {"age": age, "gender": gender, "hypertension": int(hypertension),
               "heart_disease": int(heart_disease), "ever_married": ever_married,
               "work_type": work_type, "Residence_type": residence,
               "avg_glucose_level": glucose, "bmi": bmi_val, "smoking_status": smoking}
        X_proc = preprocessor.transform(engineer_features(pd.DataFrame([row])))
        prob   = float(model.predict_proba(X_proc)[0, 1])
        tier, color, icon = risk_tier(prob)

        st.pyplot(draw_gauge(prob), use_container_width=False)
        st.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:{color}22;border:2px solid {color};'>"
            f"<span style='font-size:24px'>{icon}</span>"
            f"<span style='font-size:20px;font-weight:bold;color:{color}'> {tier} Risk</span></div>",
            unsafe_allow_html=True)
        st.markdown("")
        st.pyplot(draw_factors(row), use_container_width=False)

        with st.expander("What does this mean?"):
            msgs = {
                "Low":      "This patient profile shows **low relative stroke risk**. Maintaining a healthy lifestyle and routine checkups are recommended.",
                "Moderate": "This patient profile shows **moderate stroke risk**. Factors such as glucose levels, BMI, or cardiovascular history may warrant closer monitoring.",
                "High":     "This patient profile shows **elevated stroke risk**. Multiple high-risk factors are present. Closer clinical evaluation and preventive intervention are recommended.",
            }
            st.markdown(msgs[tier])
        st.caption("⚠️ For educational purposes only. Not validated for clinical use.")
    else:
        st.info("Fill in patient details and click **Calculate Risk** to see the assessment.")

st.divider()
st.caption("Model: Stacking Ensemble (LR + RF + HistGBM) | Dataset: UCI Stroke Prediction | Built by Ansh Patel — github.com/anshcpatel11")
