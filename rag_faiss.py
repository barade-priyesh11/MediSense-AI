import cohere
import numpy as np
import faiss
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# Fetch Cohere API key from environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

class RAGSystem:
    """
    A RAG (Retrieval-Augmented Generation) system using Cohere for embeddings
    and FAISS for efficient vector similarity search.
    """
    def __init__(self, document_list: List[str]):
        """
        Initializes the RAG system with a list of documents.

        Args:
            document_list: A list of text strings representing the documents.
        """
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in environment variables. Please set it in your .env file.")
        self.co = cohere.Client(COHERE_API_KEY)
        self.documents = document_list
        self.embeddings = self._generate_embeddings(self.documents)
        self.index = self._build_faiss_index(self.embeddings)

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using the Cohere API.
        
        Args:
            texts: A list of text strings to embed.
        
        Returns:
            A numpy array of the generated embeddings.
        """
        print("Generating embeddings with Cohere...")
        response = self.co.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type='search_document'
        )
        return np.array(response.embeddings).astype('float32')

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Builds a FAISS index from the generated embeddings for fast search.

        Args:
            embeddings: A numpy array of document embeddings.

        Returns:
            A FAISS index object.
        """
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        # Using IndexFlatL2 for a simple Euclidean distance-based index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print("FAISS index built.")
        return index

    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Searches the vector store for the most relevant documents.

        Args:
            query: The search query string.
            k: The number of top documents to retrieve.

        Returns:
            A list of strings, each representing a relevant document.
        """
        # Generate embedding for the query
        query_embedding = self.co.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query'
        ).embeddings[0]
        
        # Reshape the query for FAISS search
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Perform the search
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the corresponding documents from the original list
        relevant_documents = [self.documents[i] for i in indices[0]]
        return relevant_documents

if __name__ == "__main__":
    # Example usage for testing
    health_data = [
        "Common cold symptoms include a stuffy or runny nose, sore throat, cough, and mild fever. It's caused by a virus and typically resolves in 7-10 days.",
        "Influenza (the flu) is a contagious respiratory illness caused by influenza viruses. Symptoms are more severe than a cold and can include high fever, body aches, fatigue, and chills.",
        "COVID-19 symptoms can range from mild to severe, and may include fever, cough, fatigue, and loss of taste or smell. Serious cases can lead to pneumonia and respiratory failure.",
        "Allergies are an immune system response to foreign substances. Symptoms often include sneezing, itching, hives, and a runny nose. Common allergens are pollen, dust mites, and pet dander.",
        "A balanced diet is essential for good health. It should include a mix of fruits, vegetables, lean proteins, and whole grains to provide all necessary nutrients."
    ]
    
    # Initialize the RAG system
    rag_system = RAGSystem(health_data)

    # Test the search function
    query = "symptoms of a cold"
    results = rag_system.search(query, k=2)
    print("\nFound documents:")
    for doc in results:
        print(f"- {doc}")