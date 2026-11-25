import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Data
# In your final project, replace this with: df = pd.read_csv('UpdatedResumeDataSet.csv')
data = {
    'Resume': [
        # Data Science
        "Data scientist with experience in machine learning, python, and deep learning models.",
        "Expert in data analysis, SQL, Python, and creating predictive data pipelines.",
        "Machine learning engineer building AI models with TensorFlow and scikit-learn.",
        # Web Development
        "Web developer building responsive sites using React, HTML, CSS and Node.js.",
        "Frontend engineer with 5 years experience in JavaScript, UI/UX, and React JS.",
        "Full stack developer coding in JavaScript, React, and setting up backend APIs.",
        # HR
        "HR manager handling recruitment, employee relations, payroll, and culture.",
        "Human resources specialist focused on talent acquisition and corporate onboarding.",
        "Technical recruiter experienced in sourcing candidates and managing HR policies."
    ],
    'Category': ['Data Science', 'Data Science', 'Data Science', 
                 'Web Development', 'Web Development', 'Web Development', 
                 'HR', 'HR', 'HR']
}
df = pd.DataFrame(data)

print("--- TRAINING THE AI MODEL ---")

# 2. Convert Text to Numbers (Feature Extraction)
# The AI can't read text, so TF-IDF turns the resumes into a matrix of numbers based on word importance.
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Resume'])

# 3. Convert Categories to Numbers (Label Encoding)
# Turns 'Data Science' into 0, 'HR' into 1, etc.
encoder = LabelEncoder()
y = encoder.fit_transform(df['Category'])

# 4. Split the Data
# We train the AI on 80% of the data, and test it on 20% it has NEVER seen to check its accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
# Random Forest is highly effective for text classification.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train) # .fit() is where the actual "learning" happens

# 6. Test the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete! Accuracy on test data: {accuracy * 100}%\n")

# --- Test with a Brand New Resume ---
print("--- PREDICTING A NEW RESUME ---")
new_resume = ["I love building user interfaces with React and writing clean JavaScript code."]
print(f"Input text: '{new_resume[0]}'")

# We must transform the new text using the exact same vectorizer we trained earlier
new_resume_vector = vectorizer.transform(new_resume)

# Predict the category number, then convert it back to the text label
predicted_cat_num = model.predict(new_resume_vector)
predicted_category = encoder.inverse_transform(predicted_cat_num)

print(f"The AI predicts this resume belongs to: {predicted_category[0]}")