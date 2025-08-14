import pandas as pd

def load_snomed_mappings(path):
    """
    Load SNOMED CT mappings from CSV.
    """
    df = pd.read_csv(path)
    return df

def map_to_specialist(symptom, mapping_df):
    """
    Map symptom to SNOMED CT ID and specialist.
    """
    row = mapping_df[mapping_df['symptom'].str.lower() == symptom.lower()]
    if not row.empty:
        return row.iloc[0]['snomed_id'], row.iloc[0]['specialist']
    return None, None
