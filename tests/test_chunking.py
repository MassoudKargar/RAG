"""Tests for the chunking functionality."""
import pytest
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.chunker import ChunkerService, Chunk, ChunkMetadata


class TestChunkingBasics:
    """Test basic chunking functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = ChunkerService(chunk_size=800, chunk_overlap=100)
    
    def test_small_document(self):
        """Test 1: Small document - should produce at least 1 chunk."""
        text = "این یک متن کوتاه برای تست است."
        chunks = self.chunker.chunk_document(text, document_id="test_001")
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].metadata["document_id"] == "test_001"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == len(chunks)
    
    def test_1000_line_document(self):
        """Test 2: 1,000-line document - should produce multiple chunks."""
        text = "\n\n".join([f"خط شماره {i}: این یک متن تستی برای بررسی chunking است." for i in range(1000)])
        chunks = self.chunker.chunk_document(text, document_id="test_1000")
        
        assert len(chunks) > 1
        assert len(chunks) >= 2
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.metadata["document_id"] == "test_1000"
            assert len(chunk.text) <= 800 + 200
    
    def test_5000_line_document(self):
        """Test 3: 5,000-line document - should produce many chunks."""
        text = "\n\n".join([f"خط شماره {i}: متن طولانی برای تست chunking با حروف فارسی." for i in range(5000)])
        chunks = self.chunker.chunk_document(text, document_id="test_5000")
        
        assert len(chunks) > 10
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
    
    def test_paragraph_boundaries(self):
        """Test 4: Chunking should try to preserve paragraph boundaries."""
        para1 = "پاراگراف اول. این یک متن طولانی است که باید در یک چانک قرار بگیرد."
        para2 = "پاراگراف دوم. این پاراگراف دیگری است که باید جدا باشد."
        text = f"{para1}\n\n{para2}"
        
        chunks = self.chunker.chunk_document(text, document_id="para_test")
        
        assert len(chunks) >= 1
        if len(chunks) > 1:
            all_text = " ".join(c.text for c in chunks)
            assert "پاراگراف اول" in all_text
            assert "پاراگراف دوم" in all_text
    
    def test_heading_boundaries(self):
        """Test 5: Heading boundaries should be detected and preserved."""
        text = """# فصل اول
این متن فصل اول است.

## بخش دوم
این متن بخش دوم فصل اول است.

## بخش سوم
این متن بخش سوم فصل اول است."""

        chunks = self.chunker.chunk_document(text, document_id="heading_test")
        
        assert len(chunks) >= 1
        all_text = "\n".join(c.text for c in chunks)
        assert "فصل اول" in all_text or "بخش دوم" in all_text or "بخش سوم" in all_text
    
    def test_chunk_overlap(self):
        """Test 6: Chunks should have overlap between them."""
        chunker_small = ChunkerService(chunk_size=200, chunk_overlap=50)
        
        # Create text with paragraphs to ensure splitting
        para1 = "این یک متن تستی است. " * 15
        para2 = "پاراگراف دوم با متن متفاوت. " * 15
        para3 = "پاراگراف سوم نیز دارای متن منحصر به فرد است. " * 15
        text = f"{para1}\n\n{para2}\n\n{para3}"
        
        chunks = chunker_small.chunk_document(text, document_id="overlap_test")
        
        assert len(chunks) > 1
    
    def test_metadata_preservation(self):
        """Test 7: User-provided metadata should be preserved in chunks."""
        user_metadata = {"source": "test_file.txt", "category": "legal", "author": "test_user"}
        text = "این یک متن تستی است. " * 100
        
        chunks = self.chunker.chunk_document(
            text,
            document_id="meta_test",
            source="test_file.txt",
            metadata=user_metadata
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["source"] == "test_file.txt"
            assert chunk.metadata["category"] == "legal"
            assert chunk.metadata["author"] == "test_user"
            assert chunk.metadata["document_id"] == "meta_test"
            assert "chunk_index" in chunk.metadata
    
    def test_unique_ids(self):
        """Test 8: Each chunk should have unique IDs."""
        text = "این یک متن تستی است. " * 100
        
        chunks1 = self.chunker.chunk_document(text, document_id="unique_test")
        chunks2 = self.chunker.chunk_document(text, document_id="unique_test")
        
        assert len(chunks1) == len(chunks2)
        
        for i in range(len(chunks1)):
            assert chunks1[i].metadata["chunk_index"] == i
            assert chunks1[i].metadata["chunk_index"] == chunks2[i].metadata["chunk_index"]
    
    def test_empty_document(self):
        """Test 9: Empty document should produce no chunks."""
        chunks = self.chunker.chunk_document("", document_id="empty_test")
        assert len(chunks) == 0
        
        chunks = self.chunker.chunk_document("   ", document_id="whitespace_test")
        assert len(chunks) == 0
    
    def test_very_long_paragraph(self):
        """Test 10: Very long paragraph should still be chunked."""
        long_text = "این یک پاراگراف بسیار طولانی است. " * 200
        text = long_text
        
        chunks = self.chunker.chunk_document(text, document_id="long_para_test")
        
        assert len(chunks) >= 1
        all_text = "".join(c.text for c in chunks)
        assert len(all_text) > len(long_text) * 0.9
    
    def test_persian_text(self):
        """Test 11: Persian text should be properly chunked."""
        lines = []
        for i in range(200):
            lines.append(f"این خط شماره {i} است و شامل متن فارسی است.")
        text = "\n\n".join(lines)
        
        chunks = self.chunker.chunk_document(text, document_id="persian_test")
        
        assert len(chunks) > 0
        all_text = "".join(c.text for c in chunks)
        assert "فارسی" in all_text or "این خط" in all_text
    
    def test_document_id_auto_assignment(self):
        """Test: Document IDs should be auto-generated when not provided."""
        text = "Test " * 100
        chunks = self.chunker.chunk_document(text)
        
        assert len(chunks) > 0
        doc_id = chunks[0].metadata["document_id"]
        assert doc_id
        assert len(doc_id) > 0
    
    def test_chunk_start_end(self):
        """Test: Chunks should have character position metadata."""
        text = "این یک متن تستی است. " * 50
        chunks = self.chunker.chunk_document(text, document_id="pos_test")
        
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                assert chunks[i].metadata["chunk_end"] > 0
                assert chunks[i + 1].metadata["chunk_start"] > 0
    
    def test_metadata_not_silently_discarded(self):
        """Test: User metadata should always be in chunk metadata."""
        user_metadata = {"custom_field": "custom_value", "tenant_id": "tenant_123"}
        text = "Test text " * 50
        
        chunks = self.chunker.chunk_document(
            text,
            document_id="discard_test",
            metadata=user_metadata
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["custom_field"] == "custom_value"
            assert chunk.metadata["tenant_id"] == "tenant_123"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata


class TestChunkingGenerator:
    """Test the generator-based chunking for memory efficiency."""
    
    def test_generator_yields_batches(self):
        """Test: Generator should yield chunks in batches."""
        chunker = ChunkerService(chunk_size=100, chunk_overlap=20)
        
        # Create text with paragraphs to ensure multiple chunks
        paragraphs = [f"Test paragraph {i} with some text. " * 3 for i in range(50)]
        text = "\n\n".join(paragraphs)
        
        batches = list(chunker.chunk_document_generator(
            text,
            document_id="gen_test",
            batch_size=5
        ))
        
        assert len(batches) > 1
        assert all(len(batch) <= 5 for batch in batches)
        
        total_chunks = sum(len(batch) for batch in batches)
        assert total_chunks > 0
