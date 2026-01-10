"""
Tests for document loader functionality.

These tests verify document loading, metadata parsing,
and chunking without requiring AWS credentials.
"""

import pytest
import os
import tempfile
from src.loader import (
    parse_metadata,
    load_knowledge_base,
    create_chunks,
    get_document_stats
)


@pytest.fixture
def sample_doc_content():
    """Sample document with metadata header for testing."""
    return """---
Author: Test Author
Date: 2025-01-08
Topic: Testing
Summary: A test document for unit tests.
---

This is the actual content of the test document.
It has multiple sentences and paragraphs for testing.

This is another paragraph with more content to ensure
the document has sufficient length for chunking tests.

Additional content here to make the document longer
so we can properly test chunking with overlaps.
"""


@pytest.fixture
def sample_doc_no_metadata():
    """Sample document without metadata header."""
    return """This is a document without any metadata header.
It should still be processed correctly, just without
extracted metadata fields."""


@pytest.fixture
def temp_knowledge_base(sample_doc_content, sample_doc_no_metadata):
    """Create temporary knowledge base directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test document with metadata
        doc_path = os.path.join(tmpdir, "test_doc.txt")
        with open(doc_path, "w") as f:
            f.write(sample_doc_content)

        # Create second test document
        doc2_path = os.path.join(tmpdir, "test_doc2.txt")
        with open(doc2_path, "w") as f:
            f.write(sample_doc_content.replace("Test Author", "Second Author"))

        # Create document without metadata
        doc3_path = os.path.join(tmpdir, "no_metadata.txt")
        with open(doc3_path, "w") as f:
            f.write(sample_doc_no_metadata)

        yield tmpdir


class TestMetadataParsing:
    """Tests for metadata header parsing."""

    def test_parse_metadata_extracts_all_fields(self, sample_doc_content):
        """Test that all metadata fields are correctly extracted."""
        metadata, content = parse_metadata(sample_doc_content)

        assert metadata['author'] == 'Test Author'
        assert metadata['date'] == '2025-01-08'
        assert metadata['topic'] == 'Testing'
        assert metadata['summary'] == 'A test document for unit tests.'

    def test_parse_metadata_removes_header_from_content(self, sample_doc_content):
        """Test that metadata header is removed from content."""
        metadata, content = parse_metadata(sample_doc_content)

        assert 'actual content' in content
        assert '---' not in content
        assert 'Author:' not in content

    def test_parse_metadata_handles_no_header(self):
        """Test parsing document without metadata header."""
        content = "Just plain content without any header at all."
        metadata, result = parse_metadata(content)

        assert metadata == {}
        assert result == content

    def test_parse_metadata_handles_empty_content(self):
        """Test parsing empty content."""
        metadata, result = parse_metadata("")

        assert metadata == {}
        assert result == ""


class TestDocumentLoading:
    """Tests for document loading functionality."""

    def test_load_knowledge_base_finds_all_documents(self, temp_knowledge_base):
        """Test that all documents in directory are found."""
        documents = load_knowledge_base(temp_knowledge_base)

        assert len(documents) == 3

    def test_load_knowledge_base_parses_metadata(self, temp_knowledge_base):
        """Test that loaded documents have correct metadata."""
        documents = load_knowledge_base(temp_knowledge_base)

        authors = [doc.metadata.get('author') for doc in documents]
        assert 'Test Author' in authors
        assert 'Second Author' in authors

    def test_load_knowledge_base_handles_missing_directory(self):
        """Test behavior when directory doesn't exist."""
        documents = load_knowledge_base("nonexistent_directory")

        assert documents == []

    def test_load_knowledge_base_includes_source(self, temp_knowledge_base):
        """Test that source filename is included in metadata."""
        documents = load_knowledge_base(temp_knowledge_base)

        assert all('source' in doc.metadata for doc in documents)
        assert all('filepath' in doc.metadata for doc in documents)

    def test_load_knowledge_base_handles_no_metadata_doc(self, temp_knowledge_base):
        """Test loading document without metadata header."""
        documents = load_knowledge_base(temp_knowledge_base)

        # Should still load the document
        assert len(documents) == 3

        # Find the document without metadata
        no_meta_docs = [d for d in documents if d.metadata.get('author') is None]
        assert len(no_meta_docs) == 1


class TestDocumentStatistics:
    """Tests for document statistics calculation."""

    def test_get_document_stats_counts_documents(self, temp_knowledge_base):
        """Test that document count is correct."""
        documents = load_knowledge_base(temp_knowledge_base)
        stats = get_document_stats(documents)

        assert stats['total_documents'] == 3

    def test_get_document_stats_counts_unique_authors(self, temp_knowledge_base):
        """Test that unique author count is correct."""
        documents = load_knowledge_base(temp_knowledge_base)
        stats = get_document_stats(documents)

        # Two documents with authors, one without
        assert stats['unique_authors'] == 3  # Including 'Unknown'
        assert 'Test Author' in stats['authors']
        assert 'Second Author' in stats['authors']

    def test_get_document_stats_handles_empty_list(self):
        """Test statistics with empty document list."""
        stats = get_document_stats([])

        assert stats['total_documents'] == 0
        assert stats['unique_authors'] == 0
        assert stats['authors'] == []
        assert stats['avg_length'] == 0

    def test_get_document_stats_calculates_averages(self, temp_knowledge_base):
        """Test that average calculations are correct."""
        documents = load_knowledge_base(temp_knowledge_base)
        stats = get_document_stats(documents)

        assert stats['avg_length'] > 0
        assert stats['total_words'] > 0
        assert stats['avg_words_per_doc'] > 0


class TestDocumentChunking:
    """Tests for document chunking functionality."""

    def test_create_chunks_splits_documents(self, temp_knowledge_base):
        """Test that documents are split into multiple chunks."""
        documents = load_knowledge_base(temp_knowledge_base)
        chunks = create_chunks(documents, chunk_size=100, chunk_overlap=20)

        # Should create more chunks than documents
        assert len(chunks) > len(documents)

    def test_create_chunks_preserves_metadata(self, temp_knowledge_base):
        """Test that chunks retain source metadata."""
        documents = load_knowledge_base(temp_knowledge_base)
        chunks = create_chunks(documents, chunk_size=100, chunk_overlap=20)

        # All chunks should have source metadata
        assert all('source' in chunk.metadata for chunk in chunks)
        assert all('author' in chunk.metadata or chunk.metadata.get('author') is None
                   for chunk in chunks)

    def test_create_chunks_respects_size_limit(self, temp_knowledge_base):
        """Test that chunks don't significantly exceed size limit."""
        documents = load_knowledge_base(temp_knowledge_base)
        chunk_size = 100
        chunks = create_chunks(documents, chunk_size=chunk_size, chunk_overlap=10)

        # Allow some buffer for word boundaries (50% over is acceptable)
        max_acceptable = chunk_size * 1.5
        for chunk in chunks:
            assert len(chunk.page_content) <= max_acceptable

    def test_create_chunks_handles_empty_list(self):
        """Test chunking with empty document list."""
        chunks = create_chunks([])

        assert chunks == []

    def test_create_chunks_with_small_documents(self, temp_knowledge_base):
        """Test chunking when documents are smaller than chunk size."""
        documents = load_knowledge_base(temp_knowledge_base)
        # Use very large chunk size
        chunks = create_chunks(documents, chunk_size=10000, chunk_overlap=50)

        # Should still create chunks (one per document at minimum)
        assert len(chunks) >= len(documents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
