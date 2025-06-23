import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json
import sys
import heapq

def get_project_names(args):
    if args.multi_project_config:
        try:
            with open(args.multi_project_config, 'r', encoding='utf-8') as f:
                projects = json.load(f)
            return [p['project_name'] for p in projects]
        except Exception as e:
            print(f"Error reading multi-project config: {e}")
            sys.exit(1)
    elif args.project_names:
        return [name.strip() for name in args.project_names.split(',') if name.strip()]
    elif args.project_name:
        return [args.project_name]
    else:
        print("Error: Must specify --project-name, --project-names, or --multi-project-config")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Query Chroma DB for relevant chunks.")
    parser.add_argument('--project-name', help='Project name (Chroma namespace, single project mode)')
    parser.add_argument('--project-names', help='Comma-separated list of project names (multi-project mode)')
    parser.add_argument('--multi-project-config', help='Path to a JSON file with a list of projects (multi-project mode)')
    parser.add_argument('--persist-dir', default='chroma_db', help='Chroma DB directory')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top results to show')
    args = parser.parse_args()

    project_names = get_project_names(args)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstores = {
        name: Chroma(
            persist_directory=args.persist_dir,
            embedding_function=embeddings,
            collection_name=name
        ) for name in project_names
    }

    print(f"Projects: {', '.join(project_names)}")
    print("Type your question (or 'exit' to quit):")
    while True:
        query = input('> ')
        if query.lower() == 'exit':
            break
        all_results = []
        for pname, vstore in vectorstores.items():
            results = vstore.similarity_search(query, k=args.top_k)
            for doc in results:
                all_results.append((doc, pname))
        # Sort all results by score (if available), else just take top-k
        # Chroma returns docs, not scores, so we just take the first k from all_results
        # (If you want to use scores, use similarity_search_with_score)
        top_results = all_results[:args.top_k]
        print(f"\nTop {args.top_k} relevant chunks across projects:")
        for i, (doc, pname) in enumerate(top_results, 1):
            meta = doc.metadata
            print(f"[{i}] Project: {pname}")
            print(f"    File: {meta.get('path', 'N/A')}")
            print(f"    Type: {meta.get('type', 'N/A')}")
            print(f"    Tags: {meta.get('tags', 'N/A')}")
            snippet = doc.page_content[:300].replace('\n', ' ')
            print(f"    Content: {snippet}...\n")

if __name__ == "__main__":
    main() 