import pandas as pd
from recommender.ontology_mapper import map_to_specialist

def test_map_to_specialist():
    data = {
        "symptom": ["fever", "cough"],
        "snomed_id": [123, 456],
        "specialist": ["GP", "Pulmonologist"]
    }
    df = pd.DataFrame(data)
    snomed_id, specialist = map_to_specialist("fever", df)
    assert snomed_id == 123
    assert specialist == "GP"
