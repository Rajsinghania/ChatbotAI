import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

df = pd.read_csv("Query_Data.csv")

#preprocess
df["Question"] = df["Question"].str.lower()
df["Answer"] = df["Answer"].str.lower()


# Load a pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for questions
questions = df["Question"].tolist()
question_embeddings = embedding_model.encode(questions)

# Build a FAISS index for semantic search
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Save the index and embeddings for later use
faiss.write_index(index, "graphy_qa_index.faiss")
df.to_csv("graphy_qa_with_embeddings.csv", index=False)
