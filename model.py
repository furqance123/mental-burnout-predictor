import pandas as pd

df = pd.read_csv(r"D:\BSCS COURSE\SEMESTER 6\AI lab\Proper project\mental_burnout_500_students.csv")

print(df.head())

df.drop("Timestamp", axis=1, inplace=True)

# Remove extra spaces
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

mapping = {
    "Never": 1,
    "Occasionally": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

# Apply mapping safely
for col in df.columns:
    df[col] = df[col].map(mapping)

# Remove missing values if any
df.dropna(inplace=True)

print(df.head())

df["Total_Score"] = df.sum(axis=1)

def burnout_label(score):
    if score <= 45:
        return "Low"
    elif score <= 70:
        return "Moderate"
    else:
        return "High"

df["Burnout_Level"] = df["Total_Score"].apply(burnout_label)

print(df["Burnout_Level"].value_counts())

X = df.drop(["Total_Score", "Burnout_Level"], axis=1)
y = df["Burnout_Level"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)

print("Decision Tree Accuracy:", tree_accuracy)

importance = tree_model.feature_importances_

features = X.columns

import matplotlib.pyplot as plt

df["Burnout_Level"].value_counts().plot(kind='bar')

plt.title("Burnout Level Distribution")
plt.xlabel("Burnout Level")
plt.ylabel("Number of Students")

plt.show()

plt.figure(figsize=(10,6))

plt.barh(features, importance)

plt.title("Feature Importance")

plt.xlabel("Importance Score")

plt.show()

import joblib

joblib.dump(model, "burnout_model.pkl")
joblib.dump(scaler, "scaler.pkl")