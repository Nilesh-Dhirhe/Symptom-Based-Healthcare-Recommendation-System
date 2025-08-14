import pandas as pd
from embeddings.bio_wordvec_loader import load_biowordvec
from recommender.recommend import recommend_treatments

if __name__ == "__main__":
    # Paths
    BIOWORDVEC_PATH = "path/to/biowordvec.bin"  # Update to your BioWordVec file
    SYMPTOM_DATA = "data/sample_symptoms.csv"
    MAPPING_PATH = "data/snomed_ct_mappings.csv"

    # Load model
    model = load_biowordvec(BIOWORDVEC_PATH)

    # Load known symptoms
    symptom_df = pd.read_csv(SYMPTOM_DATA)
    known_symptoms = symptom_df['symptom'].tolist()

    # Query example
    user_symptom = "cough"
    recommendations = recommend_treatments(user_symptom, model, MAPPING_PATH, known_symptoms)
    print("Recommendations:")
    for rec in recommendations:
        print(rec)
