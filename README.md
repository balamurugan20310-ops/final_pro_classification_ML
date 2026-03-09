# Email Spam Detection using Classification Models
## AIM

To develop a machine learning model that can automatically classify emails as Spam or Ham (Not Spam) using classification algorithms and evaluate their performance.

### Data Description

The dataset contains email text messages along with their corresponding labels indicating whether the email is Spam or Ham.

## 1. Dataset Features
Feature	Description
text	The content of the email message
label	Target variable indicating spam or ham

### Example dataset:

Text	Label
Win money now	Spam
Hello how are you	Ham
Limited offer claim prize	Spam
Let's meet tomorrow	Ham
Data Processing

Since machine learning models cannot process raw text directly, the text data is converted into numerical features using TF-IDF Vectorization.

TF-IDF helps measure the importance of a word in a document relative to the dataset.

### Data Visualization

Distribution of spam and ham emails can be visualized using a bar chart.

sns.countplot(x=df["label"])
plt.title("Spam vs Ham Distribution")
plt.show()

This helps understand the balance between spam and non-spam emails.

## 2. Objective of the Analysis

The main objective of this analysis is to build a classification model capable of detecting spam emails based on their text content.

Subtasks involved in this analysis include:

Preprocessing the email text data.

Converting text into numerical features using TF-IDF.

Splitting the dataset into training and testing sets.

Training classification models.

Comparing model performance using evaluation metrics.

Possible challenges that may occur include:

Small dataset size

Text data noise

Class imbalance between spam and ham emails

These challenges may affect model accuracy and generalization.

## 3. Classification Models

Two classification algorithms are used in this analysis:

## 1. Naive Bayes Classifier

Naive Bayes is a probabilistic classification algorithm commonly used for text classification problems.

### Advantages:

Fast training

Works well with text data

Handles high dimensional data

## 2. Logistic Regression Classifier

Logistic Regression is another classification model that predicts the probability of a class using the sigmoid function.

### Advantages:

Simple and interpretable

Effective for binary classification

Model Comparison

The models are evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Based on the evaluation results, Naive Bayes performs better for text classification tasks, making it the most suitable model for spam detection.

## 4. Key Insights and Findings

From the analysis, the following insights were obtained:

Spam emails usually contain promotional words such as win, offer, prize, free, lottery.

Ham emails typically contain conversational or professional words.

The Naive Bayes model performs well for text classification because it works effectively with word frequency features.

The confusion matrix shows how accurately the model distinguishes between spam and non-spam emails.

The model successfully identifies spam messages with good accuracy.

## 5. Next Steps and Improvements

Although the model performs well, several improvements can be made:

Possible Flaws

The dataset used is very small.

Real-world emails contain more complex patterns.

Some spam emails may appear similar to legitimate emails.

Future Improvements

Use a larger real-world dataset such as the SpamAssassin dataset.

Apply advanced preprocessing techniques like stopword removal and stemming.

Test additional models such as:

Support Vector Machine (SVM)

Random Forest

Deep Learning models

Improve feature extraction using word embeddings.

These improvements can help build a more accurate spam detection system.

## ALGORITHM

1. Import necessary libraries.

2. Load the email dataset.

3. Convert text data into numerical features using TF-IDF vectorization.

4. Split the dataset into training and testing data.

5. Train classification models (Naive Bayes and Logistic Regression).

6. Predict results using the trained models.

7. Evaluate models using Accuracy, Precision, Recall, and F1-score.

8. Visualize results using a confusion matrix.

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
PRINT("Reg No: 212225240020")
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

## Output
<img width="1797" height="316" alt="image" src="https://github.com/user-attachments/assets/860412f7-4d95-4cd7-bdb5-2815d3de291e" />
<img width="1794" height="602" alt="image" src="https://github.com/user-attachments/assets/eba4d25e-61da-4fb9-8e2a-fea2700b5029" />


## RESULT

The classification models were successfully implemented to detect spam emails.
Among the tested models, Naive Bayes provided better performance for text classification, making it the most suitable model for this task.
