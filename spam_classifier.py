# ============================================================
# PROJECT:SPAM SMS CLASSIFIER
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("  SPAM SMS CLASSIFIER - STARTING")
print("=" * 60)

# Load dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print("\n✓ Dataset loaded successfully!")
except FileNotFoundError:
    print("\n⚠ 'spam.csv' not found!")
    exit()

df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df = df.drop_duplicates(keep='first')
print(f"   Total messages: {len(df)}")
print(f"   Class distribution:\n{df['label'].value_counts()}")

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print("\n⏳ Preprocessing text (takes 30-60 seconds)...")
df['cleaned_message'] = df['message'].apply(preprocess_text)
print("✓ Preprocessing complete!")

# TF-IDF Features
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['cleaned_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95,
                        ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\n✓ TF-IDF features ready!")
print(f"   Training: {X_train_tfidf.shape[0]} samples, {X_train_tfidf.shape[1]} features")

# Train models
print("\n" + "=" * 60)
print("  TRAINING MODELS")
print("=" * 60)

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"\n✓ Naive Bayes Accuracy:  {acc_nb*100:.2f}%")

svm_model = LinearSVC(C=1.0, max_iter=1000, random_state=RANDOM_STATE, dual=False)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"✓ Linear SVM Accuracy:   {acc_svm*100:.2f}%")

# Detailed evaluation
print("\n" + "=" * 60)
print("  DETAILED EVALUATION - NAIVE BAYES")
print("=" * 60)

prec_nb = precision_score(y_test, y_pred_nb)
rec_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

print(f"   Accuracy:  {acc_nb*100:.2f}%")
print(f"   Precision: {prec_nb*100:.2f}%")
print(f"   Recall:    {rec_nb*100:.2f}%")
print(f"   F1-Score:  {f1_nb*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred_nb, target_names=['Ham', 'Spam'])}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'})
plt.title('Confusion Matrix - Naive Bayes', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# Test predictions
def predict_message(message, model, vectorizer):
    cleaned = preprocess_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(vec)[0][1]
    else:
        decision = model.decision_function(vec)[0]
        prob = 1 / (1 + np.exp(-decision))
    label = "🚨 SPAM" if pred == 1 else "✓ HAM"
    confidence = prob if pred == 1 else (1 - prob)
    return label, confidence

print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS")
print("=" * 60)

test_messages = [
    "Congratulations! You won a FREE iPhone 15! Click here to claim: http://claim.com",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT: Your account will be suspended. Verify immediately!",
    "Thanks for helping with the project yesterday!",
    "WIN $1000 CASH! Text WIN to 12345 now!",
    "Can you send me the notes from class?",
    "Limited offer! 50% off everything. Buy now!",
    "Hi Mom, I'll be home for dinner around 7.",
]

best_model = nb_model

for i, msg in enumerate(test_messages, 1):
    label, conf = predict_message(msg, best_model, tfidf)
    print(f"\n[{i}] {msg[:70]}{'...' if len(msg) > 70 else ''}")
    print(f"    → {label} (Confidence: {conf*100:.1f}%)")
    print("-" * 60)

# Save model
print("\n" + "=" * 60)
print("  SAVING MODEL")
print("=" * 60)

with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model saved: spam_classifier_model.pkl")

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Vectorizer saved: tfidf_vectorizer.pkl")

# Interactive testing
print("\n" + "=" * 60)
print("  TEST YOUR OWN MESSAGES")
print("=" * 60)
print("Type a message to classify (or 'quit' to exit):\n")

while True:
    try:
        user_msg = input("Enter message: ").strip()
    except EOFError:
        break
    
    if user_msg.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not user_msg:
        print("⚠ Please enter a message.\n")
        continue
    
    label, conf = predict_message(user_msg, best_model, tfidf)
    print(f"   → {label} (Confidence: {conf*100:.1f}%)\n")

print("\n" + "=" * 60)
print("  🎉 PROJECT COMPLETE!")
print("=" * 60)
print(f"\n📁 Files generated:")
print(f"   - spam_classifier_model.pkl")
print(f"   - tfidf_vectorizer.pkl")
print(f"   - confusion_matrix.png")
print("=" * 60)
