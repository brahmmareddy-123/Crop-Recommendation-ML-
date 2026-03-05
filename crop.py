# ===============================
# 1️⃣ Import Libraries
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 2️⃣ Load Dataset
# ===============================
df = pd.read_csv("Crop_recommendation_generated.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ===============================
# 3️⃣ Split Features & Target
# ===============================
X = df.drop("label", axis=1)
y = df["label"]

# ===============================
# 4️⃣ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5️⃣ Train Model (Random Forest)
# ===============================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 6️⃣ Model Evaluation
# ===============================
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7️⃣ Crop Recommendation Function
# ===============================
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

# ===============================
# 8️⃣ Test With Sample Input
# ===============================
recommended = recommend_crop(
    N=90,
    P=42,
    K=43,
    temperature=25,
    humidity=80,
    ph=6.5,
    rainfall=200
)

print("\nRecommended Crop:", recommended)