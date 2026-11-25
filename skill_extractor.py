import spacy

# 1. Load the English language model we downloaded earlier
nlp = spacy.load("en_core_web_sm")

# 2. Access the EntityRuler to add custom rules to the NLP pipeline
# We add it *before* the 'ner' (Named Entity Recognition) component
ruler = nlp.add_pipe("entity_ruler", before="ner")

# 3. Define our custom dictionary of skills
# In a real project, you would load thousands of these from a CSV or JSON file.
skill_patterns = [
    {"label": "SKILL", "pattern": "python"},
    {"label": "SKILL", "pattern": "machine learning"},
    {"label": "SKILL", "pattern": "nlp"},
    {"label": "SKILL", "pattern": "data analysis"},
    {"label": "SKILL", "pattern": "sql"},
    {"label": "SKILL", "pattern": "java"},
    {"label": "SKILL", "pattern": "react"}
]

# 4. Inject the patterns into the ruler
ruler.add_patterns(skill_patterns)

def extract_skills(text):
    """
    Scans the text and returns a list of identified skills.
    """
    # Pass the text through our custom NLP pipeline
    doc = nlp(text)
    
    found_skills = []
    
    # Loop through all the entities the AI found
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            found_skills.append(ent.text)
            
    # Return unique skills only (removes duplicates)
    return list(set(found_skills))

# --- Test the Function ---
# Notice this text is already cleaned and lowercased (like the output of our last script)
sample_text = "john doe data scientist experienced python developer years experience machine learning data analysis nlp love building models java sql"

print("--- ANALYZING TEXT ---")
print(sample_text)

print("\n--- EXTRACTED SKILLS ---")
skills = extract_skills(sample_text)
print(skills)