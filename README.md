# Symptom-Based Healthcare Recommendation System

## Overview
An AI-powered system that suggests relevant treatments and specialists based on input symptoms.  
It uses **BioWordVec embeddings**, **SNOMED CT ontology mapping**, and **Annoy approximate nearest neighbor search** to find semantically similar symptoms.

## Features
- Symptom vectorization with **BioWordVec**
- Ontology mapping with **SNOMED CT**
- Fast symptom search using **Annoy**
- Configurable distance metrics (angular, Manhattan)

## Installation
```bash
git clone https://github.com/yourusername/symptom-recommender.git
cd symptom-recommender
pip install -r requirements.txt
```

## Usage 
python main.py 

## File Structure 
symptom_recommender/
  data/
  embeddings/ 
  recommender/
  tests/ 

## Testing 
pytest tests/
