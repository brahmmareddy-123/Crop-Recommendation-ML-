import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===============================
# Page Configuration
# ===============================
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌾 Smart Crop Recommendation System")
st.write("Enter soil and environmental details to get crop recommendation")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation_generated.csv")
    return df

df = load_data()

# ===============================
# Prepare Data
# ===============================
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===============================
# User Inputs
# ===============================
st.subheader("🌱 Enter Soil Details")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=40)
K = st.number_input("Potassium (K)", min_value=0, max_value=205, value=40)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# ===============================
# Prediction Button
# ===============================
if st.button("🌾 Recommend Crop"):
    
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    
    st.success(f"✅ Recommended Crop: {prediction[0]}")
    
    
# Model Evaluation
accuracy = model.score(X_test, y_test)
st.subheader("📈 Model Accuracy")
st.write(f"Accuracy: {round(accuracy*100,2)}%")
st.subheader("🌟 Feature Importance")

importances = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)
st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", ax=ax2)

st.pyplot(fig2)
