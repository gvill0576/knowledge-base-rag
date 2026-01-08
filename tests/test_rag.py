"""
Tests for RAG pipeline functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


@patch('src.rag.load_dotenv')
@patch('src.rag.boto3.client')
@patch('src.rag.ChatBedrock')
class TestKnowledgeBaseRAG:
    """Tests for KnowledgeBaseRAG class."""
    
    def test_init(self, mock_chat, mock_boto, mock_dotenv):
        """Test initialization."""
        from src.rag import KnowledgeBaseRAG
        
        kb = KnowledgeBaseRAG()
        
        assert kb.documents == []
        assert kb.chunks == []
        assert kb.vectorstore is None
    
    def test_ask_without_index(self, mock_chat, mock_boto, mock_dotenv):
        """Test asking question before index is created."""
        from src.rag import KnowledgeBaseRAG
        
        kb = KnowledgeBaseRAG()
        result = kb.ask("test question")
        
        assert "not initialized" in result["answer"].lower()
        assert result["sources"] == []
        assert result["num_chunks_used"] == 0
    
    @patch('src.rag.load_knowledge_base')
    def test_load_documents(self, mock_load, mock_chat, mock_boto, mock_dotenv):
        """Test document loading."""
        from src.rag import KnowledgeBaseRAG
        
        # Mock document
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"author": "Test", "topic": "Testing"}
        mock_load.return_value = [mock_doc]
        
        kb = KnowledgeBaseRAG()
        num_docs = kb.load("test_dir")
        
        assert num_docs == 1
        assert len(kb.documents) == 1
    
    @patch('src.rag.create_chunks')
    def test_process_without_documents(self, mock_chunks, mock_chat, mock_boto, mock_dotenv):
        """Test processing before loading documents."""
        from src.rag import KnowledgeBaseRAG
        
        kb = KnowledgeBaseRAG()
        result = kb.process()
        
        assert result == 0
        mock_chunks.assert_not_called()
    
    @patch('src.rag.search')
    def test_ask_with_no_results(self, mock_search, mock_chat, mock_boto, mock_dotenv):
        """Test asking when search returns no results."""
        from src.rag import KnowledgeBaseRAG
        
        kb = KnowledgeBaseRAG()
        kb.vectorstore = Mock()  # Fake vectorstore
        mock_search.return_value = []  # No results
        
        result = kb.ask("test question")
        
        assert "No relevant information" in result["answer"]
        assert result["sources"] == []
    
    @patch('src.rag.search')
    def test_ask_returns_sources(self, mock_search, mock_chat, mock_boto, mock_dotenv):
        """Test that answers include source citations."""
        from src.rag import KnowledgeBaseRAG
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Test answer based on the documents"
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance
        
        # Mock search results
        mock_doc = Mock()
        mock_doc.page_content = "Test content about Python"
        mock_doc.metadata = {
            "source": "test.txt",
            "author": "Test Author",
            "topic": "Testing Python"
        }
        mock_search.return_value = [mock_doc]
        
        kb = KnowledgeBaseRAG()
        kb.vectorstore = Mock()
        kb.llm = mock_chat_instance
        
        result = kb.ask("test question")
        
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file"] == "test.txt"
        assert result["sources"][0]["author"] == "Test Author"
        assert result["num_chunks_used"] == 1
    
    @patch('src.rag.search')
    def test_ask_removes_duplicate_sources(self, mock_search, mock_chat, mock_boto, mock_dotenv):
        """Test that duplicate sources are removed."""
        from src.rag import KnowledgeBaseRAG
        
        # Mock LLM
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance
        
        # Mock search results with duplicate sources
        mock_doc1 = Mock()
        mock_doc1.page_content = "Content 1"
        mock_doc1.metadata = {"source": "test.txt", "author": "Test", "topic": "Testing"}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Content 2"
        mock_doc2.metadata = {"source": "test.txt", "author": "Test", "topic": "Testing"}
        
        mock_search.return_value = [mock_doc1, mock_doc2]
        
        kb = KnowledgeBaseRAG()
        kb.vectorstore = Mock()
        kb.llm = mock_chat_instance
        
        result = kb.ask("test question")
        
        # Should only have 1 source even though 2 chunks from same source
        assert len(result["sources"]) == 1
        assert result["num_chunks_used"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
