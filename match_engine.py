from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the SBERT model (this takes a few seconds the first time you run it)
print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_tfidf_match(resume_text, jd_text):
    """Calculates match score using TF-IDF (Strict Keyword Matching)."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(match_score * 100, 2)

def calculate_sbert_match(resume_text, jd_text):
    """Calculates match score using SBERT (Semantic/Meaning Matching)."""
    embeddings = model.encode([resume_text, jd_text])
    match_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(match_score * 100, 2)

# --- Test the Engine ---

# Notice how the Resume uses completely DIFFERENT words than the JD
sample_resume = "backend developer experienced in building apis using node and python sql database management"
sample_jd = "looking for a software engineer capable of creating web services with javascript and managing databases"

print("\n--- TEXT INPUTS ---")
print(f"Resume: {sample_resume}")
print(f"JD:     {sample_jd}")

print("\n--- MATCH SCORES ---")
tfidf_score = calculate_tfidf_match(sample_resume, sample_jd)
print(f"TF-IDF Match (Keyword Focus): {tfidf_score}%")

sbert_score = calculate_sbert_match(sample_resume, sample_jd)
print(f"SBERT Match (Meaning Focus):  {sbert_score}%")