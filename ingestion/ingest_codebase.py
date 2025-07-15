import os
import sys
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# File extensions to include
INCLUDE_EXTENSIONS = {'.vue', '.js', '.jsx', '.ts', '.tsx', '.php', '.blade.php'}
EXCLUDE_DIRS = {'node_modules', 'vendor'}

def should_include(filename):
    for ext in INCLUDE_EXTENSIONS:
        if filename.endswith(ext):
            return True
    return False

def should_exclude_dir(dirname):
    return dirname in EXCLUDE_DIRS or dirname.startswith('.')

def chunk_file(filepath, chunk_size=20):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = ''.join(lines[i:i+chunk_size])
        chunks.append({
            'content': chunk,
            'file': filepath,
            'start_line': i+1,
            'end_line': min(i+chunk_size, len(lines))
        })
    return chunks

def ingest_codebase(root_dir, qdrant_client, embedding_model, collection_name='code_chunks'):
    all_chunks = []
    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to skip excluded directories
        dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
        for fname in filenames:
            if should_include(fname):
                fpath = os.path.join(dirpath, fname)
                matching_files.append(fpath)
                all_chunks.extend(chunk_file(fpath))
    print(f"Found {len(matching_files)} matching files:")
    for f in matching_files:
        print(f"  {f}")
    if all_chunks:
        print(f"Embedding {len(all_chunks)} code chunks...")
        embeddings = embedding_model.embed_documents([c['content'] for c in all_chunks])
        if qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.delete_collection(collection_name=collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=chunk
                ) for emb, chunk in zip(embeddings, all_chunks)
            ]
        )
        print(f"Ingested {len(all_chunks)} code chunks into Qdrant collection '{collection_name}'.")
    else:
        print("No matching files found.")

if __name__ == '__main__':
    # Accept project root as a command-line argument, default to current directory
    if len(sys.argv) > 1:
        PROJECT_ROOT = sys.argv[1]
    else:
        PROJECT_ROOT = '.'

    QDRANT_PATH = './qdrant_db'  # Always use persistent DB in rag_demo root

    print(f"Scanning for code files in: {PROJECT_ROOT}")
    print(f"Storing vectors in persistent Qdrant DB at: {QDRANT_PATH}")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    qdrant_client = QdrantClient(QDRANT_PATH)

    ingest_codebase(PROJECT_ROOT, qdrant_client, embedding_model) 