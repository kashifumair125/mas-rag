"""
tests/test_retriever.py — Unit tests for the RAG pipeline
Run with: pytest tests/ -v

These tests use mocking so they run in CI without a real ChromaDB or LLM.
This is what "proper engineering" looks like — and what CI pipelines test.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document


# ── Test: chunking logic ──────────────────────────────────────────────────────
class TestChunking:
    def test_chunk_size_respected(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        long_text = "Financial regulations. " * 100  # ~2200 chars
        doc = Document(page_content=long_text, metadata={"source": "test"})
        chunks = splitter.split_documents([doc])
        
        for chunk in chunks:
            # Allow small overflow at sentence boundaries, but roughly enforced
            assert len(chunk.page_content) <= 600, f"Chunk too large: {len(chunk.page_content)}"
    
    def test_overlap_creates_continuity(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        
        text = "A" * 80 + " BOUNDARY " + "B" * 80
        doc = Document(page_content=text, metadata={})
        chunks = splitter.split_documents([doc])
        
        # With overlap, boundary text should appear in more than one chunk
        assert len(chunks) >= 2
    
    def test_metadata_preserved_after_chunking(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        
        doc = Document(
            page_content="Test content. " * 50,
            metadata={"source_file": "mas_circular.pdf", "page": 3}
        )
        chunks = splitter.split_documents([doc])
        
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "mas_circular.pdf"
            assert chunk.metadata["page"] == 3


# ── Test: answer formatting ───────────────────────────────────────────────────
class TestAnswerFormatting:
    def test_query_returns_expected_keys(self):
        """query() must always return 'answer' and 'sources'."""
        with patch("src.retriever.build_qa_chain") as mock_chain_builder:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "result": "The capital requirement is SGD 250,000.",
                "source_documents": [
                    Document(
                        page_content="A Major Payment Institution must maintain...",
                        metadata={"source_file": "PSN01.pdf", "page": 4}
                    )
                ]
            }
            mock_chain_builder.return_value = mock_chain
            
            from src.retriever import query
            result = query("What is the capital requirement?", qa_chain=mock_chain)
            
            assert "answer" in result
            assert "sources" in result
            assert isinstance(result["sources"], list)
    
    def test_sources_contain_required_fields(self):
        """Each source dict must have content, file, and page."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Test answer",
            "source_documents": [
                Document(
                    page_content="Some regulatory text here",
                    metadata={"source_file": "circular.pdf", "page": 2}
                )
            ]
        }
        
        from src.retriever import query
        result = query("test question", qa_chain=mock_chain)
        
        source = result["sources"][0]
        assert "content" in source
        assert "file" in source
        assert "page" in source
    
    def test_empty_source_documents_handled(self):
        """System should not crash if no documents are retrieved."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "I cannot find this in the documents.",
            "source_documents": []
        }
        
        from src.retriever import query
        result = query("obscure question", qa_chain=mock_chain)
        
        assert result["answer"] == "I cannot find this in the documents."
        assert result["sources"] == []


# ── Test: ingest pipeline ─────────────────────────────────────────────────────
class TestIngest:
    def test_load_pdfs_returns_empty_for_empty_dir(self, tmp_path):
        from src.ingest import load_pdfs
        result = load_pdfs(tmp_path)
        assert result == []
    
    def test_chunk_documents_produces_output(self):
        from src.ingest import chunk_documents
        docs = [
            Document(page_content="This is test content. " * 50, metadata={"source": "test.pdf"})
        ]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1  # long doc should be split


# ── Test: OpenMetadata agent ──────────────────────────────────────────────────
class TestOpenMetadataAgent:
    def test_sample_documents_returned_on_connection_error(self):
        """App should not crash if OpenMetadata is offline."""
        from src.openmetadata_agent import fetch_openmetadata_tables
        
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            docs = fetch_openmetadata_tables()
            
            # Should return sample docs instead of crashing
            assert len(docs) > 0
    
    def test_sample_documents_have_required_metadata(self):
        from src.openmetadata_agent import _get_sample_documents
        docs = _get_sample_documents()
        
        for doc in docs:
            assert "table_name" in doc.metadata
            assert len(doc.page_content) > 0
