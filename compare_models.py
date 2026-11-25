import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('resume_dataset.csv')
print("Total resumes:", len(df))

X = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(df['Resume_str'].fillna(''))
y = LabelEncoder().fit_transform(df['Category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Linear SVM": LinearSVC(),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    print(f"{name}: accuracy {accuracy_score(y_test, p)*100:.2f}% | macro F1 {f1_score(y_test, p, average='macro'):.2f}")       