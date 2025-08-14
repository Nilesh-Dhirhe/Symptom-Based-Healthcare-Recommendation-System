from embeddings.symptom_embedder import embed_symptoms
from recommender.ontology_mapper import load_snomed_mappings, map_to_specialist
from recommender.annoy_search import SymptomANN

def recommend_treatments(user_symptom, embedding_model, mapping_path, known_symptoms):
    symptom_embeddings = embed_symptoms(known_symptoms, embedding_model)

    ann = SymptomANN(vector_size=len(next(iter(symptom_embeddings.values()))), metric='angular')
    for symptom, vec in symptom_embeddings.items():
        ann.add_item(symptom, vec)
    ann.build()

    if user_symptom not in symptom_embeddings:
        return f"Symptom '{user_symptom}' not recognized."

    similar_symptoms = ann.query(symptom_embeddings[user_symptom], top_k=3)
    mapping_df = load_snomed_mappings(mapping_path)

    results = []
    for s in similar_symptoms:
        snomed_id, specialist = map_to_specialist(s, mapping_df)
        results.append({"symptom": s, "snomed_id": snomed_id, "specialist": specialist})
    return results
