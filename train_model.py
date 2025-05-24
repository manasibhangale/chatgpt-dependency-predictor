import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Load the dataset
df = pd.read_csv("chatgpt_dependency_dataset.csv")

# Separate features and target
X = df.drop("chatgpt_dependence", axis=1)
y = df["chatgpt_dependence"]

# Define columns
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

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# Full pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train and save
model.fit(X, y)
with open("chatgpt_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'chatgpt_model.pkl'")