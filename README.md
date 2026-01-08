# Knowledge Base RAG System

![Tests](https://github.com/gvill0576/knowledge-base-rag/workflows/Test/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Retrieval-Augmented Generation (RAG) system that provides AI-powered question answering with source citations from a personal knowledge base.

## Overview

This project demonstrates a production-ready RAG system built with LangChain and AWS Bedrock. It loads documents, creates semantic embeddings, stores them in a vector database, and uses AI to answer questions while citing sources.

**Key Features:**
- 📚 Multi-document knowledge base with metadata
- 🔍 Semantic search using FAISS vector store  
- 🤖 AI-powered answers with AWS Bedrock Nova
- 📝 Automatic source citation
- ✅ Comprehensive test coverage (95%+)
- 🔄 Automated CI/CD pipeline
- 💾 Persistent vector index

## Quick Start
```bash
# Clone repository
git clone https://github.com/gvill0576/knowledge-base-rag.git
cd knowledge-base-rag

# Install dependencies
pip install -r requirements.txt

# Set AWS credentials
export AWS_PROFILE=default

# Run the system (first time - builds index)
python3 main.py

# Run with existing index (much faster)
python3 main.py --load

# Interactive Q&A mode
python3 main.py --load --interactive
```

## Usage Examples

### Demo Mode
```bash
python3 main.py
```
Runs predefined questions against your knowledge base.

### Interactive Mode
```bash
python3 main.py --load --interactive
```
Ask your own questions in an interactive session.

### Load Existing Index
```bash
python3 main.py --load
```
Skips re-embedding (saves time and API calls).

## Project Structure
```
knowledge-base-rag/
├── src/
│   ├── loader.py          # Document loading and chunking
│   ├── embeddings.py      # AWS Bedrock embeddings
│   ├── vectorstore.py     # FAISS vector store
│   └── rag.py            # Complete RAG pipeline
├── tests/
│   ├── test_loader.py     # Loader tests
│   ├── test_vectorstore.py # Vector store tests
│   └── test_rag.py        # RAG integration tests
├── knowledge_base/        # Your documents (.txt files)
├── vector_index/          # Persisted FAISS index
├── .github/workflows/     # CI/CD automation
├── main.py               # Entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Knowledge Base

The knowledge base contains documents on:
- Python Programming Fundamentals
- AWS Cloud Computing Fundamentals  
- DevOps Practices and CI/CD
- Software Testing Strategies

Each document includes:
- Author metadata
- Topic and summary
- 300+ words of content
- Structured sections

## How It Works

1. **Load** - Reads documents from `knowledge_base/` directory
2. **Parse** - Extracts metadata (author, topic, date, summary)
3. **Chunk** - Splits into 500-character pieces with 50-char overlap
4. **Embed** - Converts chunks to 1536-dimensional vectors (Titan Embed)
5. **Index** - Stores in FAISS for fast similarity search
6. **Query** - Finds relevant chunks for questions
7. **Generate** - Uses Nova Lite to create answers
8. **Cite** - Returns answer with source attributions

## Testing
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python3 -m pytest tests/test_loader.py -v

# View coverage report
open htmlcov/index.html
```

## CI/CD Pipeline

Every push and pull request automatically:
- ✅ Lints code with flake8
- ✅ Runs all tests with pytest
- ✅ Checks test coverage
- ✅ Verifies knowledge base requirements
- ✅ Validates project structure

## Technologies

- **LangChain** - RAG framework
- **AWS Bedrock** - AI models (Titan Embed, Nova Lite)
- **FAISS** - Vector similarity search
- **pytest** - Testing framework
- **GitHub Actions** - CI/CD automation

## Requirements

- Python 3.9+
- AWS Account with Bedrock access
- Enabled models: Titan Embed Text v2, Nova Lite

## Development
```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and add tests
# ...

# Run tests
python3 -m pytest tests/ -v

# Lint code
python3 -m flake8 src/ tests/ main.py

# Commit and push
git add .
git commit -m "Description of changes"
git push origin feature/my-feature

# Create pull request
gh pr create
```

## Author

Built by **George Villanueva** as part of Code Platoon's AI Cloud & DevOps Bootcamp.

Demonstrates:
- Full-stack RAG implementation
- AWS cloud integration
- Professional Git workflow
- Test-driven development
- CI/CD best practices

## License

MIT License - Educational project

## Acknowledgments

- Code Platoon for the bootcamp curriculum
- Anthropic for LangChain framework
- AWS for Bedrock AI services
