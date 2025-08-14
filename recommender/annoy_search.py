from annoy import AnnoyIndex
import numpy as np

class SymptomANN:
    def __init__(self, vector_size, metric='angular'):
        self.index = AnnoyIndex(vector_size, metric)
        self.id_map = {}
        self.counter = 0

    def add_item(self, symptom, vector):
        self.index.add_item(self.counter, vector)
        self.id_map[self.counter] = symptom
        self.counter += 1

    def build(self, n_trees=10):
        self.index.build(n_trees)

    def query(self, vector, top_k=3):
        ids = self.index.get_nns_by_vector(vector, top_k)
        return [self.id_map[i] for i in ids]
