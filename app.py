from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import pandas as pd
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the embedding model, CSV data, and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Same model used for embeddings!
df = pd.read_csv("data/graphy_qa_with_embeddings.csv")
index = faiss.read_index("data/graphy_qa_index.faiss")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json["message"]
    
    # Step 1: Convert user query to embedding
    query_embedding = embedding_model.encode([user_query])
    
    # Step 2: Search the FAISS index for closest matches
    k = 3  # Retrieve top 3 matches (you can adjust this)
    distances, indices = index.search(query_embedding, k)
    
    # Step 3: Get the most relevant answer(s)
    responses = []
    for idx in indices[0]:
        if idx >= 0:  # FAISS returns -1 if no match is found
            answer = df.iloc[idx]["Answer"]
            responses.append(answer)
    
    # Return the best answer (or all top answers)
    return jsonify({"response": responses[0] if responses else "I don’t know. Please contact support@graphy.com."})

if __name__ == "__main__":
    app.run(debug=True)