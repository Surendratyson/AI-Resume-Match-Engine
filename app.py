import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Enterprise ATS Dashboard", page_icon="📊", layout="wide")
st.title("📊 Enterprise ATS: Candidate Ranking Dashboard")
st.write("Upload a batch of resumes to automatically extract contact info, predict job categories, and rank candidates against the Job Description.")

@st.cache_resource
def load_match_models():
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    raw_skills = [
        # Languages
        "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "php", "go", "rust", "swift", "kotlin", "html", "css", "sql",
        # ML, AI & Data Science
        "machine learning", "deep learning", "nlp", "artificial intelligence", "ai", "computer vision", "data analysis",
        "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch", "open cv", "opencv", "matplotlib", "seaborn",
        "excel", "power bi", "tableau", "streamlit", "spark",
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "git", "github", "gitlab", "ci/cd", "jenkins", "linux", "bash",
        # Frameworks & Web
        "react", "angular", "vue", "node", "node.js", "express", "django", "flask", "fastapi", "spring boot", "bootstrap",
        # Databases
        "mongodb", "postgresql", "mysql", "nosql", "redis", "elasticsearch"
    ]

    skill_patterns = [{"label": "SKILL", "pattern": skill} for skill in raw_skills]
    ruler.add_patterns(skill_patterns) # type: ignore
    return sbert_model, nlp

@st.cache_resource
def load_classifier():
    try:
        df = pd.read_csv('resume_dataset.csv')
    except FileNotFoundError:
        st.error("Error: 'resume_dataset.csv' not found. Please ensure it is in the same folder as app.py.")
        return None, None, None

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['Resume_str'].fillna(''))

    encoder = LabelEncoder()
    y = encoder.fit_transform(df['Category'])

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X, y)

    return vectorizer, encoder, model

sbert_model, nlp = load_match_models()
class_vectorizer, class_encoder, class_model = load_classifier()

def extract_text_from_pdf(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            return ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.strip()
    except Exception:
        return ""

def extract_skills(text):
    doc = nlp(text.lower())
    return {ent.text for ent in doc.ents if ent.label_ == "SKILL"}

def extract_contact_info(text):
    """Uses Regex to find emails and phone numbers in raw resume text."""
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    email = email_match.group(0) if email_match else "Not Found"

    phone_match = re.search(
        r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{5}[-.\s]?\d{5})',
        text
    )
    phone = phone_match.group(0).strip() if phone_match else "Not Found"

    return email, phone

# --- User Interface ---
jd_text = st.text_area("Paste the Job Description here:", height=150)
uploaded_resumes = st.file_uploader("Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

if st.button("Run ATS Engine"):
    if uploaded_resumes and jd_text.strip():
        candidates_data = []
        jd_skills = extract_skills(jd_text)

        progress_bar = st.progress(0)
        total_resumes = len(uploaded_resumes)

        with st.spinner("Processing resumes and ranking candidates..."):
            # Encode the Job Description once instead of repeating it for every resume
            jd_embedding = sbert_model.encode([jd_text])[0]

            for i, resume_file in enumerate(uploaded_resumes):
                resume_text = extract_text_from_pdf(resume_file)

                if not resume_text:
                    candidates_data.append({
                        "Candidate File": resume_file.name,
                        "Total Match Score": 0.0,
                        "AI Context Match": 0.0,
                        "Keyword Match": 0.0,
                        "Predicted Role": "Unreadable PDF",
                        "Email": "Not Found",
                        "Phone": "Not Found",
                        "Missing Skills": ", ".join(sorted(jd_skills)).title() if jd_skills else "N/A"
                    })
                    progress_bar.progress((i + 1) / total_resumes)
                    continue

                # Extract Contact Info
                email, phone = extract_contact_info(resume_text)

                # Predict Job Category
                predicted_category = "Unknown"
                if class_model is not None and class_vectorizer is not None and class_encoder is not None:
                    resume_features = class_vectorizer.transform([resume_text])
                    cat_num = class_model.predict(resume_features)
                    predicted_category = class_encoder.inverse_transform(cat_num)[0].replace('-', ' ').title()

                # Skill Gap Analysis
                resume_skills = extract_skills(resume_text)
                missing_skills = sorted(jd_skills - resume_skills)

                # SBERT Context Score
                resume_embedding = sbert_model.encode([resume_text])[0]
                sbert_score = max(0.0, float(cosine_similarity([resume_embedding], [jd_embedding])[0][0]))

                # TF-IDF Keyword Score
                try:
                    match_vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = match_vectorizer.fit_transform([resume_text, jd_text])
                    tfidf_score = max(0.0, float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]))
                except ValueError:
                    tfidf_score = 0.0

                # Weighted Total Score (70% Context, 30% Keywords)
                total_score = (sbert_score * 0.7) + (tfidf_score * 0.3)

                candidates_data.append({
                    "Candidate File": resume_file.name,
                    "Total Match Score": round(total_score * 100, 2),
                    "AI Context Match": round(sbert_score * 100, 2),
                    "Keyword Match": round(tfidf_score * 100, 2),
                    "Predicted Role": predicted_category,
                    "Email": email,
                    "Phone": phone,
                    "Missing Skills": ", ".join(missing_skills).title() if missing_skills else "None!"
                })

                progress_bar.progress((i + 1) / total_resumes)

        # --- ATS Dashboard Display ---
        st.success(f"Successfully processed {total_resumes} candidate(s)!")
        st.subheader("🏆 Candidate Leaderboard")

        results_df = pd.DataFrame(candidates_data)
        results_df = results_df.sort_values(by="Total Match Score", ascending=False).reset_index(drop=True)

        styled_df = (
            results_df.style
            .background_gradient(subset=["Total Match Score"], cmap="Greens")
            .format({
                "Total Match Score": "{:.2f}%",
                "AI Context Match": "{:.2f}%",
                "Keyword Match": "{:.2f}%"
            })
        )

        st.dataframe(styled_df, use_container_width=True, height=400)

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download ATS Report (CSV)",
            data=csv,
            file_name="ats_candidate_rankings.csv",
            mime="text/csv"
        )
    else:
        st.error("Please upload at least one resume AND paste a job description!")