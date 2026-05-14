from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid

app = FastAPI()
@app.get("/")
def root():
    return {"message": "API funcionando"}