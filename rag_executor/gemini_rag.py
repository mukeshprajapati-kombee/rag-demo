import argparse
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    parser = argparse.ArgumentParser(description="RAG: Retrieve from Chroma and answer with Gemini.")
    parser.add_argument('--project-name', required=True, help='Project name (Chroma namespace)')
    parser.add_argument('--persist-dir', default='chroma_db', help='Chroma DB directory')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top chunks to retrieve')
    args = parser.parse_args()

    # Check credentials
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        print("Please set your GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS environment variable.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = Chroma(
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
        collection_name=args.project_name
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print(f"Project: {args.project_name}")
    print("Type your question (or 'exit' to quit):")
    while True:
        query = input('> ')
        if query.lower() == 'exit':
            break
        results = vectorstore.similarity_search(query, k=args.top_k)
        if not results:
            print("No relevant chunks found.")
            continue
        # Format context for Gemini
        context = "\n\n".join([
            f"File: {doc.metadata.get('path', 'N/A')}\nType: {doc.metadata.get('type', 'N/A')}\nTags: {doc.metadata.get('tags', 'N/A')}\nContent:\n{doc.page_content[:1000]}"
            for doc in results
        ])
        prompt = f"Answer the following question using the provided code/documentation context.\n\nQuestion: {query}\n\nContext:\n{context}\n\nIf you reference code, mention the file path."
        answer = llm.invoke(prompt)
        print(f"\nGemini Answer:\n{answer}\n")
        print("Referenced files:")
        for doc in results:
            print(f"- {doc.metadata.get('path', 'N/A')}")
        print()

if __name__ == "__main__":
    main() 