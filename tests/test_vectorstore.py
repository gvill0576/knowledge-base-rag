"""
Tests for vector store functionality.

These tests use mocking to avoid requiring AWS credentials
and to make tests run fast without actually embedding text.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.vectorstore import search, search_with_scores


class TestSearch:
    """Tests for search functionality."""
    
    def test_search_returns_correct_number(self):
        """Test that search returns requested number of results."""
        # Create mock vectorstore
        mock_vectorstore = Mock()
        
        # Create mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "Test content 1"
        mock_doc1.metadata = {"source": "test1.txt", "author": "Test"}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Test content 2"
        mock_doc2.metadata = {"source": "test2.txt", "author": "Test"}
        
        # Configure mock to return our documents
        mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]
        
        # Test search
        results = search(mock_vectorstore, "test query", k=2)
        
        assert len(results) == 2
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=2)
    
    def test_search_returns_documents_with_metadata(self):
        """Test that search returns Document objects with metadata."""
        mock_vectorstore = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.txt", "author": "Test Author"}
        mock_vectorstore.similarity_search.return_value = [mock_doc]
        
        results = search(mock_vectorstore, "test query", k=1)
        
        assert len(results) == 1
        assert hasattr(results[0], 'page_content')
        assert hasattr(results[0], 'metadata')
        assert results[0].metadata['author'] == 'Test Author'
    
    def test_search_raises_on_none_vectorstore(self):
        """Test that search raises error with None vectorstore."""
        with pytest.raises(ValueError, match="Vectorstore is None"):
            search(None, "test query")
    
    def test_search_handles_empty_results(self):
        """Test behavior when no results found."""
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = []
        
        results = search(mock_vectorstore, "nonexistent topic", k=3)
        
        assert results == []


class TestSearchWithScores:
    """Tests for search with scores functionality."""
    
    def test_search_with_scores_returns_tuples(self):
        """Test that search_with_scores returns (doc, score) tuples."""
        mock_vectorstore = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.txt"}
        mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        
        results = search_with_scores(mock_vectorstore, "test query", k=1)
        
        assert len(results) == 1
        assert len(results[0]) == 2  # Tuple of (doc, score)
        doc, score = results[0]
        assert score == 0.5
    
    def test_search_with_scores_includes_metadata(self):
        """Test that results include document metadata."""
        mock_vectorstore = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {
            "source": "test.txt",
            "author": "Test Author",
            "topic": "Testing"
        }
        mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.3)]
        
        results = search_with_scores(mock_vectorstore, "test query", k=1)
        
        doc, score = results[0]
        assert doc.metadata['author'] == 'Test Author'
        assert doc.metadata['topic'] == 'Testing'
    
    def test_search_with_scores_returns_multiple_results(self):
        """Test that multiple results are returned correctly."""
        mock_vectorstore = Mock()
        
        # Create multiple mock documents
        docs = []
        for i in range(3):
            doc = Mock()
            doc.page_content = f"Content {i}"
            doc.metadata = {"source": f"test{i}.txt"}
            docs.append((doc, 0.1 * i))
        
        mock_vectorstore.similarity_search_with_score.return_value = docs
        
        results = search_with_scores(mock_vectorstore, "test", k=3)
        
        assert len(results) == 3
        # Check scores are included
        scores = [score for _, score in results]
        assert scores == [0.0, 0.1, 0.2]
    
    def test_search_with_scores_raises_on_none_vectorstore(self):
        """Test that search raises error with None vectorstore."""
        with pytest.raises(ValueError, match="Vectorstore is None"):
            search_with_scores(None, "test query")


@patch('src.vectorstore.create_embeddings')
@patch('src.vectorstore.FAISS')
class TestVectorStoreCreation:
    """Tests for vector store creation (mocked to avoid AWS calls)."""
    
    def test_create_vector_store_uses_embeddings(self, mock_faiss, mock_create_embeddings):
        """Test that create_vector_store uses embeddings model."""
        from src.vectorstore import create_vector_store
        
        # Create mock chunks
        mock_chunk = Mock()
        mock_chunk.page_content = "Test"
        mock_chunk.metadata = {"source": "test.txt"}
        chunks = [mock_chunk]
        
        # Mock embeddings
        mock_embeddings = Mock()
        mock_create_embeddings.return_value = mock_embeddings
        
        # Mock FAISS
        mock_vectorstore = Mock()
        mock_vectorstore.index.ntotal = 1
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        # Create vector store
        result = create_vector_store(chunks)
        
        # Verify embeddings were created
        mock_create_embeddings.assert_called_once()
        # Verify FAISS was called with chunks and embeddings
        mock_faiss.from_documents.assert_called_once_with(chunks, mock_embeddings)
    
    def test_create_vector_store_raises_on_empty_chunks(self, mock_faiss, mock_create_embeddings):
        """Test that create_vector_store raises error with empty chunks."""
        from src.vectorstore import create_vector_store
        
        with pytest.raises(ValueError, match="Cannot create vector store from empty chunks"):
            create_vector_store([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
