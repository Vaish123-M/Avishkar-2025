import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample Dataset (email content & labels)
data = [
    ("Your account needs verification, click here!", 1),
    ("Exclusive offer! Win a free iPhone!", 1),
    ("Hello, your meeting is scheduled at 3 PM.", 0),
    ("Bank alert! Update your password now.", 1),
    ("Hi, let's catch up for coffee tomorrow.", 0),
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["text", "label"])

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = np.array(df["label"])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
pickle.dump(model, open("phishing_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model training complete and saved!")
