import gensim
import os

def load_biowordvec(path: str):
    """
    Load BioWordVec embeddings from a .bin or .txt file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"BioWordVec file not found at: {path}")
    print(f"[INFO] Loading BioWordVec embeddings from {path}...")
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print("[INFO] BioWordVec loaded successfully.")
    return model
