from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid

app = FastAPI()
@app.get("/")
def root():
    return {"message": "API funcionando"}


model = SentenceTransformer("all-MiniLM-L6-v2")

class Document:
  def __init__(self, text: str, metadata: dict[str, str]):
    self.text = text
    self.metadata = metadata

class SearchResult:
  def __init__(self, score: float, document: Document):
    self.score = score
    self.document = document

class FilteredVectorStore:
  def __init__(self, embedding_model: SentenceTransformer):
    self.embedding_model = embedding_model
    self.documents = []
    self.embeddings = []

  def add_documents(self, documents: list[Document]):
    self.documents.extend(documents)
    texts = [doc.text for doc in documents]
    new_embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
    self.embeddings.extend(new_embeddings)

  def search(self, query: str,top_k: int = 5,metadata_filter: dict[str, str] | None = None) -> list[SearchResult]:
    query_embedding = self.embedding_model.encode([query],convert_to_numpy=True)

    filtered_docs = []
    filtered_embeddings = []

    for doc, emb in zip(self.documents, self.embeddings):

      include_doc = True

      # Si existe filtro por metadata
      if metadata_filter is not None:

        # Por cada filtro, si valor no coincide, no se incluye
        for key, value, in metadata_filter.items():
          if doc.metadata.get(key) != value:
            include_doc = False
            break

      # Filtrar los documentos según sus metadatos
      if include_doc:
        filtered_docs.append(doc)
        filtered_embeddings.append(emb)

    if len(filtered_docs) == 0:
      return []

    # Cos calculo
    similarities = cosine_similarity(query_embedding,np.array(filtered_embeddings))[0]

    # Ordenados indice y resultados
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
      results.append(
          SearchResult(
              score=float(similarities[idx]),
              document=filtered_docs[idx]
          )
      )

    return results
  

