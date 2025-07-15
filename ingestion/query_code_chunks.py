import sys
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

COLLECTION_NAME = 'code_chunks'

if __name__ == '__main__':
    # Accept Qdrant DB path as optional argument
    if len(sys.argv) > 1:
        qdrant_path = sys.argv[1]
    else:
        qdrant_path = ':memory:'

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    qdrant_client = QdrantClient(qdrant_path)

    print(f"Connected to Qdrant at: {qdrant_path}")
    print(f"Searching collection: {COLLECTION_NAME}")

    while True:
        query = input("\nEnter your code search query (or 'exit' to quit): ")
        if query.strip().lower() == 'exit':
            break
        query_embedding = embedding_model.embed_query(query)
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5
        )
        if not results:
            print("No results found.")
            continue
        print(f"\nTop {len(results)} results:")
        for i, hit in enumerate(results, 1):
            payload = hit.payload
            print(f"\nResult {i}:")
            print(f"File: {payload.get('file')}")
            print(f"Lines: {payload.get('start_line')} - {payload.get('end_line')}")
            print(f"Score: {hit.score:.4f}")
            print("Code chunk:")
            print(payload.get('content'))
            print("-"*40) 