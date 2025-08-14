import numpy as np

def embed_symptoms(symptom_list, model):
    """
    Generate embeddings for a list of symptoms using BioWordVec.
    """
    embeddings = {}
    for symptom in symptom_list:
        tokens = symptom.lower().split()
        vectors = []
        for token in tokens:
            if token in model:
                vectors.append(model[token])
        if vectors:
            embeddings[symptom] = np.mean(vectors, axis=0)
    return embeddings
