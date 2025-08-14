import pytest
import numpy as np
from embeddings.symptom_embedder import embed_symptoms

class DummyModel:
    def __contains__(self, key):
        return True
    def __getitem__(self, key):
        return np.array([1.0, 2.0, 3.0])

def test_embed_symptoms():
    model = DummyModel()
    symptoms = ["fever", "headache"]
    embeddings = embed_symptoms(symptoms, model)
    assert "fever" in embeddings
    assert embeddings["fever"].shape == (3,)
