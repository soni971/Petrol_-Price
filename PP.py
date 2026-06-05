import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- BACKGROUND ----------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1601597111158-2fceff292cdc");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        [data-testid="stAppViewContainer"] {
            background: rgba(0,0,0,0.6);
        }

        h1, h2, h3 {
            color: #FFD700;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# ---------------- TITLE ----------------
st.title("⛽ Petrol Price Prediction Dashboard")

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("petrol_price_large_dataset1.csv")
except:
    st.error("❌ Dataset file not found!")
    st.stop()

# ---------------- PREPROCESS ----------------
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# ---------------- DATA ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------- MODEL ----------------
X = df[['Crude_Oil_Price','USD_INR','Demand_Index']]
y = df['Petrol_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", round(mae,2))
col2.metric("R² Score", round(r2,2))
col3.metric("Accuracy %", f"{accuracy:.2f}%")

# ---------------- PREDICTION ----------------
st.subheader("🔢 Enter Values")

crude = st.number_input("Crude Oil Price", value=80.0)
usd = st.number_input("USD/INR", value=75.0)
demand = st.number_input("Demand Index", value=100.0)

if st.button("🚀 Predict Petrol Price"):
    
    input_data = pd.DataFrame({
        'Crude_Oil_Price': [crude],
        'USD_INR': [usd],
        'Demand_Index': [demand]
    })

    prediction = model.predict(input_data)

    st.subheader("💰 Predicted Petrol Price")
    st.success(f"₹ {prediction[0]:.2f}")

    st.info(f"Model Accuracy: {accuracy:.2f}%")

# ---------------- VISUALIZATION ----------------
st.markdown("## 📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔥 Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(5,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("### 📉 Crude Oil vs Petrol Price")
    fig2, ax2 = plt.subplots(figsize=(5,4))
    sns.scatterplot(
        x=df["Crude_Oil_Price"],
        y=df["Petrol_Price"],
        ax=ax2
    )
    st.pyplot(fig2)

# ---------------- LINE GRAPH ----------------
st.markdown("## 📈 Actual vs Predicted")

fig3, ax3 = plt.subplots()
ax3.plot(y_test.values, label="Actual")
ax3.plot(y_pred, label="Predicted")

ax3.legend()
ax3.set_xlabel("Data Points")
ax3.set_ylabel("Price")

st.pyplot(fig3)

# ---------------- MAP ----------------
st.markdown("## 🗺 Petrol Price Map")

map_data = pd.DataFrame({
    "lat": [28.6139, 19.0760, 13.0827, 22.5726],
    "lon": [77.2090, 72.8777, 80.2707, 88.3639]
})

st.map(map_data)