# Email-Spam-Detection-using-Naive-Bayes-Classification

## AIM
To find Email Spam Detection using Naive Bayes Classification

## ALGORITHM

1. Load the labeled email dataset.

2. Convert text into numerical features using TF-IDF vectorization.

3. Split the dataset into training and testing sets.

4. Train the Multinomial Naive Bayes classifier.

5. Evaluate using Accuracy, Precision, Recall, F1-Score and visualize the results

## PROGRAM

```py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Sample dataset
data = {
    "text": [
        "Win money now",
        "Limited offer claim prize",
        "Hello how are you",
        "Let's meet tomorrow",
        "Congratulations you won lottery",
        "Project meeting at 10",
        "Free vacation offer",
        "Call me when free"
    ],
    "label": ["spam","spam","ham","ham","spam","ham","spam","ham"]
}

df = pd.DataFrame(data)

# Features and labels
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Name: BALAMURUGAN S")
print("Reg No: 212225240020")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualization
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
## OUTPUT

<img width="1391" height="288" alt="image" src="https://github.com/user-attachments/assets/c32ec106-a4b8-4e00-8747-66d566fa8c38" />

<img width="1372" height="556" alt="image" src="https://github.com/user-attachments/assets/15c77308-a447-480e-9191-a76aeefbb53c" />


## Result

The Naive Bayes classifier successfully classified spam and non-spam emails with high accuracy and 
clear confusion matrix visualization.
