# ChatbotAI
AI-Powered Support Chatbot with Flask, FAISS, and Sentence Transformers

This is a simple AI-powered chatbot built using Flask, SentenceTransformers, FAISS, and Postman for backend testing. The chatbot is designed to handle user queries by finding the most relevant answers from a pre-trained dataset.

# Features
Natural Language Understanding: The chatbot uses the all-MiniLM-L6-v2 model from SentenceTransformers to understand user queries.

Fast Search: Powered by the FAISS library, it quickly retrieves the most relevant answers from a database of pre-embedded responses.

Interactive Web Interface: Users can interact with the chatbot through a simple HTML and JavaScript-based UI.

API Integration: Supports REST API calls for easy integration with other platforms.

Scalable Backend: Built with Flask, making it lightweight and easy to deploy.

# Technologies Used
Backend: Flask (Python)
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
Search Engine: FAISS (Facebook AI Similarity Search)
Frontend: HTML, CSS, JavaScript
Testing: Postman

# How It Works
The user inputs a question through the web interface.
The backend converts the question into an embedding vector using the SentenceTransformer model.
FAISS retrieves the closest matching answers from the pre-indexed dataset.
The chatbot responds with the most relevant answer.
