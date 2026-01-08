# Knowledge Base RAG System

![Tests](https://github.com/gvill0576/knowledge-base-rag/workflows/Test/badge.svg)

A Retrieval-Augmented Generation (RAG) system that provides AI-powered question answering with source citations from a personal knowledge base.

## Features

- 📚 Multi-document knowledge base with metadata
- 🔍 Semantic search using FAISS vector store
- 🤖 AI-powered answers with AWS Bedrock
- 📝 Source citation for all responses
- ✅ Comprehensive test coverage
- 🔄 Automated CI/CD pipeline

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set AWS credentials
export AWS_PROFILE=bootcamp

# Run the system
python main.py

# Interactive mode
python main.py --interactive
```

## Project Structure
```
knowledge-base-rag/
├── src/
│   ├── loader.py          # Document loading and chunking
│   ├── embeddings.py      # AWS Bedrock embeddings
│   ├── vectorstore.py     # FAISS vector store
│   └── rag.py            # Complete RAG pipeline
├── tests/
│   ├── test_loader.py
│   ├── test_vectorstore.py
│   └── test_rag.py
├── knowledge_base/        # Your documents
├── .github/workflows/     # CI/CD
└── main.py               # Entry point
```

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Author

Built by [Your Name] as part of Code Platoon's AI Cloud & DevOps Bootcamp.

## License

Educational project - MIT License
