"""
Document loading and processing for the knowledge base.

This module handles loading text documents from a directory,
parsing metadata headers, and chunking documents for embedding.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_metadata(content):
    """
    Parse metadata header from document content.
    
    Documents should start with a YAML-like metadata block:
    ---
    Author: Name
    Date: YYYY-MM-DD
    Topic: Topic Name
    Summary: Brief summary
    ---
    
    Args:
        content: Raw document content with metadata header
        
    Returns:
        tuple: (metadata dict, content without header)
        
    Example:
        >>> content = "---\\nAuthor: John\\n---\\nText"
        >>> metadata, text = parse_metadata(content)
        >>> metadata['author']
        'John'
    """
    metadata = {}
    
    # Check if content starts with metadata delimiter
    if content.startswith('---'):
        parts = content.split('---', 2)
        
        if len(parts) >= 3:
            header = parts[1].strip()
            
            # Parse each line in the header
            for line in header.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Convert to lowercase for consistency
                    metadata[key.strip().lower()] = value.strip()
            
            # Return metadata and content without header
            return metadata, parts[2].strip()
    
    # No metadata header found - return empty dict
    return metadata, content


def load_knowledge_base(directory="knowledge_base"):
    """
    Load all text documents from the knowledge base directory.
    
    This function:
    1. Finds all .txt files in the directory
    2. Loads each file using TextLoader
    3. Parses metadata headers
    4. Creates Document objects with content and metadata
    
    Args:
        directory: Path to directory containing .txt files
        
    Returns:
        list: List of Document objects with content and metadata
        
    Example:
        >>> docs = load_knowledge_base("knowledge_base")
        >>> len(docs)
        4
        >>> docs[0].metadata['author']
        'George Villa'
    """
    documents = []
    doc_path = Path(directory)
    
    # Check if directory exists
    if not doc_path.exists():
        print(f"❌ Directory '{directory}' not found")
        return documents
    
    # Get all .txt files
    txt_files = list(doc_path.glob("*.txt"))
    print(f"📂 Found {len(txt_files)} documents in {directory}/")
    
    # Load each file
    for file in txt_files:
        try:
            loader = TextLoader(str(file), encoding='utf-8')
            docs = loader.load()
            
            for doc in docs:
                # Parse metadata from content
                metadata, content = parse_metadata(doc.page_content)
                
                # Add file information to metadata
                metadata['source'] = file.name
                metadata['filepath'] = str(file)
                
                # Create new document with parsed content and metadata
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
                
                author = metadata.get('author', 'Unknown')
                topic = metadata.get('topic', 'Unknown')
                print(f"  ✅ Loaded: {file.name}")
                print(f"      Author: {author}")
                print(f"      Topic: {topic}")
        
        except Exception as e:
            print(f"  ❌ Error loading {file.name}: {e}")
    
    print(f"\n📚 Successfully loaded {len(documents)} documents")
    return documents


def get_document_stats(documents):
    """
    Get statistics about loaded documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        dict: Statistics including total docs, authors, topics, avg length
        
    Example:
        >>> docs = load_knowledge_base()
        >>> stats = get_document_stats(docs)
        >>> stats['total_documents']
        4
    """
    if not documents:
        return {
            "total_documents": 0,
            "unique_authors": 0,
            "authors": [],
            "topics": [],
            "avg_length": 0,
            "total_words": 0
        }
    
    authors = set()
    topics = set()
    total_length = 0
    total_words = 0
    
    for doc in documents:
        author = doc.metadata.get('author', 'Unknown')
        topic = doc.metadata.get('topic', 'Unknown')
        authors.add(author)
        topics.add(topic)
        total_length += len(doc.page_content)
        total_words += len(doc.page_content.split())
    
    return {
        "total_documents": len(documents),
        "unique_authors": len(authors),
        "authors": sorted(list(authors)),
        "topics": sorted(list(topics)),
        "avg_length": total_length // len(documents),
        "total_words": total_words,
        "avg_words_per_doc": total_words // len(documents)
    }


def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks while preserving metadata.
    
    Chunking is important because:
    1. Large documents don't fit in model context windows
    2. Smaller chunks provide more precise retrieval
    3. Overlapping chunks prevent losing context at boundaries
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        list: List of Document chunks with preserved metadata
        
    Example:
        >>> docs = load_knowledge_base()
        >>> chunks = create_chunks(docs, chunk_size=500)
        >>> len(chunks) > len(docs)
        True
    """
    if not documents:
        print("⚠️  No documents to chunk")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print(f"\n✂️  Chunking documents...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Overlap: {chunk_overlap} characters")
    
    chunks = text_splitter.split_documents(documents)
    
    avg_chunk_size = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    
    print(f"\n✅ Split {len(documents)} documents into {len(chunks)} chunks")
    print(f"   Average chunk size: {avg_chunk_size} characters")
    
    return chunks


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("TESTING DOCUMENT LOADER")
    print("=" * 60)
    
    # Load documents
    docs = load_knowledge_base()
    
    # Show statistics
    print("\n" + "=" * 60)
    print("DOCUMENT STATISTICS")
    print("=" * 60)
    stats = get_document_stats(docs)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create chunks
    print("\n" + "=" * 60)
    print("CREATING CHUNKS")
    print("=" * 60)
    chunks = create_chunks(docs)
    
    # Show sample chunk
    if chunks:
        print("\n" + "=" * 60)
        print("SAMPLE CHUNK")
        print("=" * 60)
        sample = chunks[0]
        print(f"Source: {sample.metadata.get('source')}")
        print(f"Author: {sample.metadata.get('author')}")
        print(f"Topic: {sample.metadata.get('topic')}")
        print(f"Content: {sample.page_content[:200]}...")
    
    print("\n✅ Loader test complete!")
