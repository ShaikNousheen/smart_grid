import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🔌 Smart Grid Load Forecasting Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("smart_grid_dataset.csv")
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def preprocess_data(df):
    df = df.copy()
    df = df.dropna()
    features = df.drop(columns=["Predicted Load (kW)"])
    target = df["Predicted Load (kW)"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def plot_anomalies(df):
    threshold = df["Predicted Load (kW)"].mean() + 2 * df["Predicted Load (kW)"].std()
    anomalies = df[df["Predicted Load (kW)"] > threshold]
    st.subheader("📊 Predicted Load with Anomalies")
    fig, ax = plt.subplots()
    df["Predicted Load (kW)"].plot(ax=ax, label="Predicted Load")
    ax.scatter(anomalies.index, anomalies["Predicted Load (kW)"], color='red', label='Anomalies')
    ax.legend()
    st.pyplot(fig)

def explain_model(model, X_train, feature_names):
    st.subheader("🧠 SHAP Feature Importance")
    explainer = shap.Explainer(model, X_train, feature_names=feature_names)
    shap_values = explainer(X_train)
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    st.pyplot(fig)

df = load_data()
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_data(df)
model = train_model(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Display Dashboard
st.subheader("📈 Current vs Predicted Load")
st.line_chart(df[["Predicted Load (kW)"]])

plot_anomalies(df)
explain_model(model, X_train_scaled, feature_names=X_train.columns)
