import os
import qdrant_client
import uuid
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus

# --- Part 0: Configuration ---
# Load the GOOGLE_API_KEY from the .env file
load_dotenv()

# --- Mock Data: Simulating different context sources ---
# In a real system, this data would be fetched from Git, Jira, Confluence, etc.
mock_data = {
    "code": [
        {
            "id": "code_001",
            "source": "rag_executor/gemini_rag.py",
            "content": "def get_embedding(text, model='text-embedding-3-small'):\n   text = text.replace('\\n', ' ')\n   return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']"
        },
        {
            "id": "code_002",
            "source": "vector_db/embed_and_store.py",
            "content": "def store_chunks(chunks, collection_name='code_collection'):\n    client = chromadb.Client()\n    collection = client.get_or_create_collection(name=collection_name)\n    for chunk in chunks:\n        collection.add(ids=[chunk['id']], embeddings=[chunk['embedding']], metadatas=[{'source': chunk['source']}])"
        }
    ],
    "docs": [
        {
            "id": "doc_001",
            "source": "architecture.md",
            "content": "The system uses a RAG architecture. It takes a developer prompt, generates embeddings, retrieves context from a vector DB, builds a new prompt, and sends it to an LLM like GPT-4o."
        }
    ],
    "tasks": [
        {
            "id": "task_001",
            "source": "Jira-123",
            "content": "Refactor the authentication module to use OAuth2 instead of basic auth. The new implementation should be backward compatible for at least one release cycle."
        }
    ]
}

# --- Part 1: Embedding Model Initialization (Mukesh's Responsibility) ---
# Initialize one embedding model for the whole application for efficiency.
# Using BGE-Large as specified in the architecture document.
print("Initializing embedding model BAAI/bge-large-en-v1.5... (This may take a moment on first run)")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model initialized.")


# --- Part 2: Context Collector & Vector Retrieval with Qdrant (Chirag's Responsibility) ---
class VectorContextStore:
    """A wrapper for a Qdrant to handle context storage and retrieval."""
    def __init__(self, collection_name="developer_assistant_db", embedding_function=None):
        if not embedding_function:
            raise ValueError("An embedding function must be provided to the VectorContextStore.")
            
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.client = qdrant_client.QdrantClient(":memory:")
        
        try:
            vector_size = self.embedding_function.client.get_sentence_embedding_dimension()
        except AttributeError:
            vector_size = 1024
            print(f"Could not dynamically determine vector size, falling back to {vector_size}.")

        # Use the modern, safer way to create a collection
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Qdrant collection '{self.collection_name}' created.")

    def preprocess_and_store_data(self, data):
        """Embeds and stores data from various sources in batches for efficiency."""
        print("Preprocessing and storing data in vector DB...")
        all_contents = []
        all_payloads = []
        
        for source_type, items in data.items():
            for item in items:
                all_contents.append(item["content"])
                all_payloads.append({
                    "content": item["content"],
                    "source": item["source"], 
                    "type": source_type
                })

        if all_contents:
            embeddings = self.embedding_function.embed_documents(all_contents)
            
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
                    for embedding, payload in zip(embeddings, all_payloads)
                ]
            )
            
            if operation_info.status != UpdateStatus.COMPLETED:
                print("Warning: Data storage may not be fully complete.")
            else:
                print("Data storage complete.")

    def retrieve_relevant_context(self, query_text, n_results=5):
        """Performs a semantic search to find relevant context."""
        print(f"Retrieving top {n_results} relevant contexts for the query...")
        query_embedding = self.embedding_function.embed_query(query_text)
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=n_results
        )
        
        contexts = []
        for hit in search_results:
            context = {
                "content": hit.payload["content"],
                "source": hit.payload["source"],
                "type": hit.payload["type"],
                "distance": 1 - hit.score  # Convert similarity score to distance
            }
            contexts.append(context)
        return contexts

# --- Part 3: Prompt Builder, LLM, and IDE Integration (Survil's Responsibility) ---
def build_prompt_with_context(user_prompt, contexts):
    """Builds a structured prompt for the LLM using a template."""
    print("Building structured prompt with retrieved context...")
    context_str = ""
    if not contexts:
        context_str = "No relevant context was found in the knowledge base."
    else:
        for ctx in sorted(contexts, key=lambda x: x['distance']):
            context_str += f"- Context Type: {ctx['type']}\n"
            context_str += f"- Source: {ctx['source']}\n"
            context_str += f"- Content: {ctx['content']}\n\n"

    prompt_template = f"""
You are a RAG-based Developer Assistant. Your goal is to provide clean, tested, and commented code based on the developer's intent and the relevant context provided.

Developer Intent:
---
{user_prompt}
---

Relevant Context Retrieved from my Knowledge Base:
---
{context_str}
---

Instructions:
Based on the developer's intent and the provided context, please provide a comprehensive answer. If the request is for code, return clean, well-documented, and tested code ready to be applied. If no relevant context is found, state that and answer based on your general knowledge.
"""
    return prompt_template

def call_llm(prompt, model="gemini-1.5-flash-latest"):
    """Calls the Gemini LLM with the final prompt using an API Key."""
    print(f"Sending final prompt to the LLM ({model})...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "FATAL ERROR: GOOGLE_API_KEY is not set in the environment. Please add it to your .env file."
            
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"An unexpected error occurred when calling the LLM: {e}"

# --- Main Execution Flow ---
if __name__ == "__main__":
    # Check for API key at startup for a clear error message.
    if not os.getenv("GOOGLE_API_KEY"):
        print("FATAL ERROR: The GOOGLE_API_KEY is not configured.")
        print("Please create a .env file and add your Google AI Studio API key to it.")
    else:
        context_store = VectorContextStore(embedding_function=embedding_model)
        context_store.preprocess_and_store_data(mock_data)
        
        user_query = input("\nAsk your developer assistant a question: ")

        retrieved_contexts = context_store.retrieve_relevant_context(user_query, n_results=3)
        final_prompt = build_prompt_with_context(user_query, retrieved_contexts)
        llm_response = call_llm(final_prompt)
        
        print("\n--- Assistant's Response ---\n")
        print(llm_response)
        print("\n--------------------------\n") 