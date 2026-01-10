"""
RAG (Retrieval-Augmented Generation) pipeline.

This module implements the complete RAG system that:
1. Loads and processes documents
2. Creates searchable vector store
3. Retrieves relevant context for questions
4. Generates answers with source citations
"""

from langchain_aws import ChatBedrock
from src.loader import load_knowledge_base, create_chunks, get_document_stats
from src.embeddings import create_embeddings
from src.vectorstore import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    search
)
import boto3
import os
from dotenv import load_dotenv


class KnowledgeBaseRAG:
    """
    Complete RAG system for knowledge base question answering.

    This class encapsulates the entire RAG pipeline:
    - Document loading and processing
    - Vector store creation and management
    - Question answering with source citations

    Example:
        >>> kb = KnowledgeBaseRAG()
        >>> kb.load()
        >>> kb.process()
        >>> kb.index()
        >>> result = kb.ask("What is Python?")
        >>> print(result['answer'])
        >>> print(result['sources'])
    """

    def __init__(self):
        """Initialize the RAG system."""
        load_dotenv()

        self.documents = []
        self.chunks = []
        self.vectorstore = None
        self.embeddings = None
        self.llm = None

        print("🤖 Knowledge Base RAG System initialized")

    def _init_llm(self):
        """Initialize the language model for answer generation."""
        if self.llm is None:
            if os.getenv("AWS_PROFILE"):
                os.environ["AWS_PROFILE"] = os.getenv("AWS_PROFILE")

            client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1"
            )

            self.llm = ChatBedrock(
                model_id="us.amazon.nova-lite-v1:0",
                client=client,
                model_kwargs={
                    "max_tokens_to_sample": 1500,
                    "temperature": 0.7
                }
            )
            print("✅ Language model initialized (Nova Lite)")

    def load(self, directory="knowledge_base"):
        """
        Load documents from directory.

        Args:
            directory: Path to directory containing documents

        Returns:
            int: Number of documents loaded
        """
        print(f"\n{'='*60}")
        print(f"STEP 1: LOADING DOCUMENTS")
        print(f"{'='*60}")

        self.documents = load_knowledge_base(directory)

        if self.documents:
            stats = get_document_stats(self.documents)
            print(f"\n📊 Knowledge Base Statistics:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Total words: {stats['total_words']:,}")
            print(f"   Average words/doc: {stats['avg_words_per_doc']}")
            print(f"   Authors: {', '.join(stats['authors'])}")
            print(f"   Topics covered: {len(stats['topics'])}")

        return len(self.documents)

    def process(self, chunk_size=500, chunk_overlap=50):
        """
        Process loaded documents into chunks.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between chunks

        Returns:
            int: Number of chunks created
        """
        if not self.documents:
            print("❌ No documents to process. Load documents first.")
            return 0

        print(f"\n{'='*60}")
        print(f"STEP 2: PROCESSING DOCUMENTS")
        print(f"{'='*60}")

        self.chunks = create_chunks(
            self.documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return len(self.chunks)

    def index(self):
        """
        Create vector store index from processed chunks.

        Returns:
            int: Number of vectors in the index
        """
        if not self.chunks:
            print("❌ No chunks to index. Process documents first.")
            return 0

        print(f"\n{'='*60}")
        print(f"STEP 3: CREATING VECTOR INDEX")
        print(f"{'='*60}")

        self.embeddings = create_embeddings()
        self.vectorstore = create_vector_store(self.chunks, self.embeddings)

        return self.vectorstore.index.ntotal if self.vectorstore else 0

    def save(self, path="vector_index"):
        """
        Save the vector store to disk.

        Args:
            path: Directory path for saving
        """
        if self.vectorstore:
            print(f"\n{'='*60}")
            print(f"SAVING VECTOR STORE")
            print(f"{'='*60}")
            save_vector_store(self.vectorstore, path)
        else:
            print("❌ No vector store to save. Create index first.")

    def load_index(self, path="vector_index"):
        """
        Load an existing vector store from disk.

        Args:
            path: Directory path containing saved index
        """
        print(f"\n{'='*60}")
        print(f"LOADING EXISTING INDEX")
        print(f"{'='*60}")

        self.embeddings = create_embeddings()
        self.vectorstore = load_vector_store(path, self.embeddings)

    def ask(self, question, k=3, show_context=False):
        """
        Ask a question and get an answer with source citations.

        Args:
            question: Question to answer
            k: Number of document chunks to retrieve
            show_context: If True, print retrieved context

        Returns:
            dict: Contains 'question', 'answer', 'sources', 'num_chunks_used'
        """
        # Ensure LLM is initialized
        self._init_llm()

        # Check if system is ready
        if not self.vectorstore:
            return {
                "question": question,
                "answer": "❌ Knowledge base not initialized. Load documents and create index first.",
                "sources": [],
                "num_chunks_used": 0
            }

        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        # Step 1: Retrieve relevant chunks
        print(f"\n🔍 Searching for relevant information (top {k} chunks)...")
        relevant_docs = search(self.vectorstore, question, k=k)

        if not relevant_docs:
            return {
                "question": question,
                "answer": "No relevant information found in the knowledge base for this question.",
                "sources": [],
                "num_chunks_used": 0
            }

        print(f"✅ Found {len(relevant_docs)} relevant chunks")

        # Show sources
        print(f"\n📚 Sources:")
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            author = doc.metadata.get('author', 'Unknown')
            topic = doc.metadata.get('topic', 'Unknown')
            print(f"   {i}. {source}")
            print(f"      Author: {author}")
            print(f"      Topic: {topic}")

        # Step 2: Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            author = doc.metadata.get('author', 'Unknown')
            context_parts.append(
                f"[Source {i}: {source} by {author}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        if show_context:
            print(f"\n📄 Context being sent to LLM:")
            print("-" * 60)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 60)

        # Step 3: Create prompt
        prompt = f"""Based on the following context from a knowledge base, please answer the question.
Only use information from the context provided. If the context doesn't contain enough information
to fully answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, comprehensive answer based on the context above. Include specific details
and cite which sources you're drawing from when relevant."""

        # Step 4: Generate answer
        print(f"\n🤖 Generating answer...")
        response = self.llm.invoke(prompt)
        answer = response.content

        # Step 5: Extract sources with metadata
        sources = []
        for doc in relevant_docs:
            source_info = {
                "file": doc.metadata.get('source', 'Unknown'),
                "author": doc.metadata.get('author', 'Unknown'),
                "topic": doc.metadata.get('topic', 'Unknown')
            }
            # Avoid duplicate sources
            if source_info not in sources:
                sources.append(source_info)

        # Print result
        print(f"\n💬 Answer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        print(f"\n📖 Based on {len(sources)} source(s):")
        for source in sources:
            print(f"   • {source['file']} - {source['topic']}")

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_chunks_used": len(relevant_docs)
        }

    def interactive(self):
        """Start an interactive Q&A session."""
        print(f"\n{'='*60}")
        print("🎯 INTERACTIVE MODE")
        print("='*60}")
        print("Ask questions about your knowledge base.")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print(f"{'='*60}")

        while True:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break

            self.ask(question)


def build_knowledge_base(load_existing=False, save_index=True):
    """
    Build or load the complete knowledge base.

    Args:
        load_existing: If True, load existing index instead of rebuilding
        save_index: If True, save the index after building

    Returns:
        KnowledgeBaseRAG: Ready-to-use knowledge base system
    """
    kb = KnowledgeBaseRAG()

    if load_existing and os.path.exists("vector_index"):
        # Load existing index
        print("\n📂 Loading existing vector index...")
        kb.load_index()
    else:
        # Build from scratch
        print("\n🏗️  Building knowledge base from scratch...")
        kb.load()
        kb.process()
        kb.index()

        if save_index:
            kb.save()

    return kb


if __name__ == "__main__":
    import sys

    print("🚀 Knowledge Base RAG System")
    print("=" * 60)

    # Check command line arguments
    load_existing = "--load" in sys.argv
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    # Build the knowledge base
    kb = build_knowledge_base(load_existing=load_existing)

    if interactive:
        # Start interactive mode
        kb.interactive()
    else:
        # Run demo questions
        print(f"\n{'='*60}")
        print("DEMO: Testing with sample questions")
        print(f"{'='*60}")

        demo_questions = [
            "What is Python and why is it popular?",
            "How does AWS Bedrock help with AI applications?",
            "What is continuous integration in DevOps?",
            "Why is software testing important?"
        ]

        for question in demo_questions:
            result = kb.ask(question)
            input("\nPress Enter to continue to next question...")

        print(f"\n{'='*60}")
        print("✅ Demo complete!")
        print("\nUsage tips:")
        print("  python main.py              # Build and run demo")
        print("  python main.py --load       # Load existing index")
        print("  python main.py --interactive # Interactive Q&A")
        print(f"{'='*60}")
