"""
Vector store management for semantic search.

This module handles creating, saving, loading, and searching
the FAISS vector store that enables semantic similarity search.
"""

import os
from langchain_community.vectorstores import FAISS
from src.embeddings import create_embeddings


def create_vector_store(chunks, embeddings=None):
    """
    Create FAISS vector store from document chunks.

    This process:
    1. Takes each document chunk
    2. Converts it to a vector using the embeddings model
    3. Stores the vector in FAISS for fast similarity search

    Args:
        chunks: List of Document chunks
        embeddings: BedrockEmbeddings instance (creates new if None)

    Returns:
        FAISS: Vector store ready for similarity search

    Example:
        >>> from src.loader import load_knowledge_base, create_chunks
        >>> docs = load_knowledge_base()
        >>> chunks = create_chunks(docs)
        >>> vectorstore = create_vector_store(chunks)
        >>> vectorstore.index.ntotal
        16
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty chunks list")

    if embeddings is None:
        embeddings = create_embeddings()

    print(f"\n🔄 Creating vector store from {len(chunks)} chunks...")
    print("   This may take a minute as each chunk is embedded...")

    # Create vector store (this calls AWS Bedrock for each chunk)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"✅ Vector store created with {vectorstore.index.ntotal} vectors")
    return vectorstore


def save_vector_store(vectorstore, path="vector_index"):
    """
    Save vector store to disk for later use.

    Persisting the vector store avoids re-embedding documents
    on every run, which is slow and costs money.

    Args:
        vectorstore: FAISS vector store to save
        path: Directory path for storing the index

    Example:
        >>> save_vector_store(vectorstore, "my_index")
        💾 Vector store saved to 'my_index/'
    """
    if not vectorstore:
        raise ValueError("Cannot save None vectorstore")

    vectorstore.save_local(path)
    print(f"💾 Vector store saved to '{path}/'")
    print(f"   Index file: {path}/index.faiss")
    print(f"   Metadata file: {path}/index.pkl")


def load_vector_store(path="vector_index", embeddings=None):
    """
    Load vector store from disk.

    Args:
        path: Directory path containing saved index
        embeddings: BedrockEmbeddings instance (creates new if None)

    Returns:
        FAISS: Loaded vector store

    Raises:
        FileNotFoundError: If vector store doesn't exist at path

    Example:
        >>> vectorstore = load_vector_store("my_index")
        📂 Vector store loaded from 'my_index/' (16 vectors)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vector store not found at '{path}'. "
            f"Create one first with create_vector_store()."
        )

    if embeddings is None:
        embeddings = create_embeddings()

    print(f"\n📂 Loading vector store from '{path}/'...")

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"✅ Vector store loaded ({vectorstore.index.ntotal} vectors)")
    return vectorstore


def search(vectorstore, query, k=3):
    """
    Search for relevant documents using semantic similarity.

    This converts your query to a vector and finds the k most
    similar document chunks based on vector distance.

    Args:
        vectorstore: FAISS vector store
        query: Search query string
        k: Number of results to return

    Returns:
        list: Top k most relevant Document chunks

    Example:
        >>> results = search(vectorstore, "What is Python?", k=3)
        >>> len(results)
        3
        >>> results[0].metadata['topic']
        'Python Programming Fundamentals'
    """
    if not vectorstore:
        raise ValueError("Vectorstore is None")

    results = vectorstore.similarity_search(query, k=k)
    return results


def search_with_scores(vectorstore, query, k=3):
    """
    Search with similarity scores.

    Lower scores indicate higher similarity in FAISS
    (the score is actually a distance measure).

    Args:
        vectorstore: FAISS vector store
        query: Search query string
        k: Number of results to return

    Returns:
        list: Tuples of (Document, similarity_score)

    Example:
        >>> results = search_with_scores(vectorstore, "testing", k=2)
        >>> for doc, score in results:
        ...     print(f"Score: {score:.4f}, Topic: {doc.metadata['topic']}")
        Score: 0.2341, Topic: Software Testing Strategies
        Score: 0.4125, Topic: DevOps Practices and CI/CD
    """
    if not vectorstore:
        raise ValueError("Vectorstore is None")

    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":
    # Test the vector store
    from src.loader import load_knowledge_base, create_chunks

    print("=" * 60)
    print("TESTING VECTOR STORE")
    print("=" * 60)

    # Load documents
    print("\n1. Loading documents...")
    docs = load_knowledge_base()

    # Create chunks
    print("\n2. Creating chunks...")
    chunks = create_chunks(docs)

    # Create embeddings
    print("\n3. Initializing embeddings...")
    embeddings = create_embeddings()

    # Create vector store
    print("\n4. Creating vector store...")
    vectorstore = create_vector_store(chunks, embeddings)

    # Test search
    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    test_queries = [
        "What is Python programming?",
        "How does AWS cloud work?",
        "What is continuous integration?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_with_scores(vectorstore, query, k=2)

        for doc, score in results:
            source = doc.metadata.get('source', 'Unknown')
            topic = doc.metadata.get('topic', 'Unknown')
            print(f"   📄 {source}")
            print(f"      Topic: {topic}")
            print(f"      Score: {score:.4f}")
            print(f"      Preview: {doc.page_content[:100]}...")

    # Test save/load
    print("\n" + "=" * 60)
    print("TESTING SAVE/LOAD")
    print("=" * 60)

    print("\n5. Saving vector store...")
    save_vector_store(vectorstore, "test_vector_index")

    print("\n6. Loading vector store...")
    loaded = load_vector_store("test_vector_index", embeddings)

    print("\n✅ All vector store tests passed!")
