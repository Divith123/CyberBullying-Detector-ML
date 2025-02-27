import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load combined dataset
data = pd.read_csv("datasets/combined_dataset.csv")

# Drop rows with missing or invalid text
data = data.dropna(subset=["cleaned_text"])  # Drop rows where 'cleaned_text' is NaN
data = data[data["cleaned_text"].apply(lambda x: isinstance(x, str))]  # Keep only string values

# Split data
X = data["cleaned_text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_transformed, y_train)

# Evaluate model
y_pred = model.predict(X_test_transformed)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "models/cyberbullying_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")