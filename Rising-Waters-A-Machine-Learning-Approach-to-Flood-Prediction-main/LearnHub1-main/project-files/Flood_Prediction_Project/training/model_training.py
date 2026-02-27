import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ----------------------------
# 1. Load Dataset
# ----------------------------

print("Loading dataset...")

dataset = pd.read_excel("../dataset/flood dataset.xlsx")

print("Dataset Loaded Successfully")
print(dataset.head())
print("\nDataset Shape:", dataset.shape)


# ----------------------------
# 2. Separate Features and Target
# ----------------------------

X = dataset.drop("flood", axis=1)
y = dataset["flood"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ----------------------------
# 3. Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ----------------------------
# 4. Feature Scaling
# ----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")


# ----------------------------
# 5. Model Training
# ----------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Model: {name}")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name


# ----------------------------
# 6. Save Best Model
# ----------------------------

print("\nBest Model Selected:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))

joblib.dump(best_model, "../flask_app/floods.save")
joblib.dump(scaler, "../flask_app/transform.save")

print("\nModel and Scaler Saved Successfully!")
