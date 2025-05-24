import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("chatgpt_dependency_dataset.csv")

# Separate features and target
X = df.drop("chatgpt_dependence", axis=1)
y = df["chatgpt_dependence"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define categorical and numerical columns
categorical_cols = ["reason_for_using_chatgpt", "department"]
numerical_cols = [
    "chatgpt_usage_frequency_per_week",
    "average_duration_per_session_minutes",
    "attempt_before_chatgpt",
    "confidence_in_solving_alone",
    "peer_usage_influence",
    "cgpa",
    "used_other_ai_tools",
    "chatgpt_preferred_over_google"
]

# Preprocessing pipeline: encode categorical, scale numerical
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# Model pipeline: preprocessing + classifier
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

# GridSearchCV for best parameters using F1-score (good for class imbalance)
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Train the model with grid search
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Evaluation on Test Data:")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the best model
with open("chatgpt_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\n✅ Model saved as 'chatgpt_model.pkl'")

# Feature importance extraction
feature_names = numerical_cols + categorical_cols
importances = best_model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n🔥 Feature Importances:")
print(importance_df)

# Optional: Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance in Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
