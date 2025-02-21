import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1️⃣ Load Dataset
df = pd.read_csv("phishing_emails.csv")  # Ensure your dataset is named correctly

# 2️⃣ Preprocess Data
X = df["email_text"]  # Column containing email content
y = df["label"]  # 1 for phishing, 0 for safe

# 3️⃣ Convert Text to Numerical Features
vectorizer = TfidfVectorizer(max_features=5000)
X_transformed = vectorizer.fit_transform(X)

# 4️⃣ Split Data for Training
df_train, df_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 5️⃣ Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df_train, y_train)

# 6️⃣ Evaluate Model
y_pred = model.predict(df_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7️⃣ Save Model and Vectorizer
pickle.dump(model, open("phishing_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
