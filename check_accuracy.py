import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv('resume_dataset.csv')
texts = df['Resume_str'].fillna('')

encoder = LabelEncoder()
y = encoder.fit_transform(df['Category'])

# 1. Split raw text FIRST to prevent data leakage
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Fit TF-IDF only on training data, then transform test data
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 3. Compare models with balanced class weights for rare categories
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "Linear SVM": LinearSVC(class_weight='balanced', random_state=42),
}

best_name, best_f1, best_preds = None, -1, None

print(f"Total resumes: {len(df)} | Train: {len(X_train_text)} | Test: {len(X_test_text)}\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    macro_f1 = f1_score(y_test, preds, average='macro')
    weighted_f1 = f1_score(y_test, preds, average='weighted')
    print(f"{name:20s} | Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.2f} | Weighted F1: {weighted_f1:.2f}")

    if weighted_f1 > best_f1:
        best_name, best_f1, best_preds = name, weighted_f1, preds

print(f"\n--- Detailed Report for Best Model ({best_name}) ---")
print(classification_report(y_test, best_preds, target_names=encoder.classes_, zero_division=0))