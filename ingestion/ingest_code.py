import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter
)
import sys

# File type to language mapping for better chunking
EXT_TYPE_MAP = {
    '.py': {'type': 'python', 'language': Language.PYTHON},
    '.js': {'type': 'javascript', 'language': Language.JS},
    '.ts': {'type': 'typescript', 'language': Language.TS},
    '.jsx': {'type': 'react', 'language': Language.JS},
    '.tsx': {'type': 'react', 'language': Language.TS},
    '.php': {'type': 'php', 'language': Language.PHP},
    '.md': {'type': 'markdown', 'language': None},
    '.json': {'type': 'json', 'language': None},
}

# Directories to ignore during traversal
IGNORE_DIRS = {'node_modules', 'venv', '.git', '__pycache__', '.next', 'build', 'dist'}

# Default chunking configuration
DEFAULT_CHUNK_CONFIG = {
    'code': {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'length_function': len,
    },
    'markdown': {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'length_function': len,
        'headers_to_split_on': [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
        ]
    }
}

def infer_tags(file_path):
    # Simple tag inference based on file path
    tags = []
    parts = Path(file_path).parts
    for part in parts:
        if part not in ("src", "pages", "components", "utils", "lib"):
            tags.append(part.lower())
    return tags

def split_code(content: str, language, config: Dict[str, Any]) -> List[str]:
    """Split code into chunks using language-aware splitting if possible, else generic."""
    if language is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            length_function=config['length_function']
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            length_function=config['length_function']
        )
    return splitter.split_text(content)

def split_markdown(content: str, config: Dict[str, Any]) -> List[str]:
    """Split markdown content using header-aware splitting."""
    # First split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config['headers_to_split_on']
    )
    header_splits = header_splitter.split_text(content)
    
    # Then split by size if needed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        length_function=config['length_function']
    )
    
    chunks = []
    for split in header_splits:
        chunks.extend(text_splitter.split_text(split.page_content))
    return chunks

def chunk_file(file_path: str, content: str, ext: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single file into chunks with metadata."""
    file_info = EXT_TYPE_MAP[ext]
    chunks = []
    
    # Split content based on file type
    if file_info['type'] == 'markdown':
        split_contents = split_markdown(content, config['markdown'])
    else:
        split_contents = split_code(content, file_info['language'], config['code'])
    
    # Create chunk objects with metadata
    for i, chunk_content in enumerate(split_contents):
        chunk = {
            'content': chunk_content,
            'chunk_index': i,
            'total_chunks': len(split_contents),
            'start_line': None,  # TODO: Track line numbers
            'end_line': None,    # TODO: Track line numbers
        }
        chunks.append(chunk)
    
    return chunks

def chunk_files(project_root: str, project_name: str, chunk_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Process all files in the project into chunks."""
    if chunk_config is None:
        chunk_config = DEFAULT_CHUNK_CONFIG
        
    chunks = []
    for root, dirs, files in os.walk(project_root):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1]
            file_path = os.path.join(root, file)
            
            if ext in EXT_TYPE_MAP:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"  !! Error reading {file_path}: {e}")
                    continue
                
                # Get file chunks with metadata
                file_chunks = chunk_file(file_path, content, ext, chunk_config)
                
                # Add file and project metadata to each chunk
                for chunk in file_chunks:
                    chunk.update({
                        "project": project_name,
                        "path": os.path.relpath(file_path, project_root),
                        "type": EXT_TYPE_MAP[ext]['type'],
                        "language": EXT_TYPE_MAP[ext]['language'],
                        "tags": infer_tags(file_path),
                    })
                    chunks.append(chunk)
            else:
                print(f"  -> Skipped (unsupported extension: {ext})")
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Ingest code and markdown files from one or more projects.")
    parser.add_argument('--project-root', help='Path to the project root directory (single project mode)')
    parser.add_argument('--project-name', help='Project name for metadata (single project mode)')
    parser.add_argument('--output', default=None, help='Output JSON file (optional, single project mode)')
    parser.add_argument('--chunk-size', type=int, help='Override default chunk size')
    parser.add_argument('--chunk-overlap', type=int, help='Override default chunk overlap')
    parser.add_argument('--multi-project-config', help='Path to a JSON file with a list of projects (multi-project mode)')
    args = parser.parse_args()

    # Override chunk config if specified
    chunk_config = DEFAULT_CHUNK_CONFIG.copy()
    if args.chunk_size:
        chunk_config['code']['chunk_size'] = args.chunk_size
        chunk_config['markdown']['chunk_size'] = args.chunk_size
    if args.chunk_overlap:
        chunk_config['code']['chunk_overlap'] = args.chunk_overlap
        chunk_config['markdown']['chunk_overlap'] = args.chunk_overlap

    if args.multi_project_config:
        # Multi-project mode
        try:
            with open(args.multi_project_config, 'r', encoding='utf-8') as f:
                projects = json.load(f)
        except Exception as e:
            print(f"Error reading multi-project config: {e}")
            sys.exit(1)
        for project in projects:
            project_root = project['project_root']
            project_name = project['project_name']
            output_path = project.get('output', f"data/{project_name}-chunks.json")
            print(f"Ingesting project: {project_name} at {project_root}")
            chunks = chunk_files(project_root, project_name, chunk_config)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"  Wrote {len(chunks)} chunks to {output_path}")
    else:
        # Single project mode (backward compatible)
        if not (args.project_root and args.project_name):
            print("Error: Must specify --project-root and --project-name (or use --multi-project-config)")
            sys.exit(1)
        chunks = chunk_files(args.project_root, args.project_name, chunk_config)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"Wrote {len(chunks)} chunks to {args.output}")
        else:
            print(json.dumps(chunks, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main() 