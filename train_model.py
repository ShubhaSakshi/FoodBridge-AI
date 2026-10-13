"""
Train FoodBridge's demonstration urgency classifier.

Important: This generates synthetic examples and labels using a transparent heuristic.
It proves the ML pipeline can train/load/predict, but it is NOT validated against real
donation outcomes and must not be used to decide whether food is safe.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

rng = np.random.default_rng(42)
n = 1200
food_type_code = rng.integers(0, 4, n)
quantity = rng.integers(1, 201, n)
age_hours = rng.integers(0, 25, n)
hours_to_deadline = rng.uniform(0, 24, n)

# Synthetic target: illustrative coordination priority only, not food safety.
urgency_score = (
    0.45 * np.clip(age_hours / 24, 0, 1)
    + 0.35 * (1 - np.clip(hours_to_deadline / 24, 0, 1))
    + 0.20 * np.clip(quantity / 200, 0, 1)
)
priority = np.where(urgency_score >= 0.62, "High",
           np.where(urgency_score >= 0.34, "Medium", "Low"))

data = pd.DataFrame({
    "food_type_code": food_type_code,
    "quantity": quantity,
    "age_hours": age_hours,
    "hours_to_deadline": hours_to_deadline,
    "priority": priority,
})
X = data.drop(columns="priority")
y = data["priority"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Holdout accuracy on synthetic data: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))
out = Path(__file__).with_name("foodbridge_priority_model.joblib")
joblib.dump(model, out)
print(f"Saved demo model to: {out}")
print("Reminder: synthetic-data demonstration only; not validated for real-world decisions.")
