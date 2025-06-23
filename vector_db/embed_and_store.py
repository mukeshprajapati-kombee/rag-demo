import os
import argparse
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import sys

def embed_and_store(chunks_json, project_name, persist_dir):
    # Load chunks
    with open(chunks_json, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Prepare documents and metadatas
    texts = [chunk['content'] for chunk in chunks]
    metadatas = [
        {
            'project': chunk['project'],
            'path': chunk['path'],
            'type': chunk['type'],
            'tags': ','.join(chunk['tags']) if isinstance(chunk['tags'], list) else str(chunk['tags'])
        } for chunk in chunks
    ]

    # Use HuggingFace bge-large embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # Store in Chroma (namespace = project name)
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir,
        collection_name=project_name
    )
    vectorstore.persist()
    print(f"Embedded and stored {len(texts)} chunks in Chroma namespace '{project_name}' at '{persist_dir}'")

def main():
    parser = argparse.ArgumentParser(description="Embed and store chunks in Chroma DB using HuggingFace embeddings.")
    parser.add_argument('--chunks-json', help='Path to the JSON file with chunks (single project mode)')
    parser.add_argument('--project-name', help='Project name (used as Chroma namespace, single project mode)')
    parser.add_argument('--persist-dir', default='chroma_db', help='Chroma DB directory')
    parser.add_argument('--multi-project-config', help='Path to a JSON file with a list of projects (multi-project mode)')
    args = parser.parse_args()

    if args.multi_project_config:
        # Multi-project mode
        try:
            with open(args.multi_project_config, 'r', encoding='utf-8') as f:
                projects = json.load(f)
        except Exception as e:
            print(f"Error reading multi-project config: {e}")
            sys.exit(1)
        for project in projects:
            chunks_json = project['chunks_json']
            project_name = project['project_name']
            persist_dir = project.get('persist_dir', 'chroma_db')
            print(f"Embedding project: {project_name} from {chunks_json}")
            embed_and_store(chunks_json, project_name, persist_dir)
    else:
        # Single project mode (backward compatible)
        if not (args.chunks_json and args.project_name):
            print("Error: Must specify --chunks-json and --project-name (or use --multi-project-config)")
            sys.exit(1)
        embed_and_store(args.chunks_json, args.project_name, args.persist_dir)

if __name__ == "__main__":
    main() 