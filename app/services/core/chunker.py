"""Document chunking service for large document support.

This service handles intelligent document chunking with semantic boundary detection.
It supports Persian/Arabic text and preserves document structure where possible.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a single chunk of a document."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class ChunkMetadata:
    """Metadata for a chunk."""
    document_id: str
    chunk_index: int
    total_chunks: int
    source: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_start_char: int = 0
    chunk_end_char: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_start": self.chunk_start_char,
            "chunk_end": self.chunk_end_char,
        }
        if self.source:
            result["source"] = self.source
        if self.section:
            result["section"] = self.section
        if self.page is not None:
            result["page"] = self.page
        return result


class ChunkerService:
    """Service for intelligent document chunking with semantic boundaries."""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _extract_section_headers(self, text: str) -> List[tuple]:
        """Extract section headers and their positions from text.
        
        Supports:
        - Persian/Arabic numbered sections: "بخش ۱", "سرفصل ۲"
        - English numbered sections: "Section 1", "Chapter 2"
        - Markdown headings: "# Heading", "## Heading"
        - Underlined headings: "======="
        """
        sections = []
        
        # Persian/Arabic numbered sections
        # Match patterns like: "بخش ۱"، "سرفصل دوم"، "فصل ۳"
        persian_numbered = re.compile(
            r'(?:بخش|سرفصل|فصل|بخش\s+[۰-۹۰-۹]+|سرفصل\s+[۰-۹۰-۹]+)',
            re.IGNORECASE
        )
        
        # English numbered sections
        english_numbered = re.compile(
            r'(?:Section|Chapter|Part)\s*[\d\s]+',
            re.IGNORECASE
        )
        
        # Markdown headings
        markdown_heading = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        for match in persian_numbered.finditer(text):
            sections.append((match.start(), match.group()))
        
        for match in english_numbered.finditer(text):
            sections.append((match.start(), match.group()))
            
        for match in markdown_heading.finditer(text):
            sections.append((match.start(), match.group(2)))
        
        # Sort by position
        sections.sort(key=lambda x: x[0])
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (max one chunk at a time).
        
        Handles various paragraph delimiters including:
        - Double newlines
        - Persian newline conventions
        - Bullet points
        """
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Also split by bullet patterns
        bullet_pattern = re.compile(
            r'(?:[•\-•▸▹›»❯▶►➜➔➕➖➘➚⚡── ──● ○ ◉ ◆ ■ □ ▣ ♦ ◊ ♪ ★ ☆ ⚡ 🔥 🎯 🎯 🎯]' 
            r'|[۰-۹۰-۹]+\.|[0-9]+\.)'
        )
        
        result = []
        for para in paragraphs:
            if bullet_pattern.match(para):
                # Split bullet lists into separate items
                parts = re.split(bullet_pattern, para)
                for part in parts:
                    part = part.strip()
                    if part:
                        result.append(part)
            else:
                if para.strip():
                    result.append(para.strip())
        
        return result
    
    def _count_chars(self, text: str) -> int:
        """Count characters, handling Persian/Arabic properly."""
        return len(text)
    
    def _calculate_word_count(self, text: str) -> int:
        """Estimate word count for chunk sizing."""
        # Split on whitespace
        words = text.split()
        return len(words)
    
    def chunk_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split a document into chunks with semantic boundaries.
        
        Strategy:
        1. First, try to split by section/paragaph boundaries
        2. Then, if a chunk exceeds size limit, split at sentence level
        3. Finally, use character limit as absolute boundary
        
        Args:
            text: The full document text
            document_id: Unique identifier for the document
            source: Source file/path of the document
            metadata: Additional metadata to include in each chunk
            
        Returns:
            List of Chunk objects with text and metadata
        """
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        if not text or not text.strip():
            logger.warning(f"Empty or whitespace-only document provided for chunk_id {document_id}")
            return []
        
        # Initialize with original text to preserve Unicode
        text = self._normalize_text(text)
        
        # Get section boundaries
        sections = self._extract_section_headers(text)
        
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk_text = ""
        current_chunk_start = 0
        chunk_index = 0
        total_chars_in_chunk = 0
        
        for para in paragraphs:
            para_len = self._count_chars(para)
            
            # If a single paragraph is larger than chunk_size, we need to split it
            # at sentence boundaries
            if para_len > self.chunk_size:
                # Flush current chunk if any
                if total_chars_in_chunk > 0:
                    chunk_meta = ChunkMetadata(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        source=source,
                        chunk_start_char=current_chunk_start,
                        chunk_end_char=current_chunk_start + total_chars_in_chunk
                    )
                    if metadata:
                        chunk_meta_dict = chunk_meta.to_dict()
                        chunk_meta_dict.update(metadata)
                        chunks.append(Chunk(text=current_chunk_text, metadata=chunk_meta_dict))
                    else:
                        chunks.append(Chunk(text=current_chunk_text, metadata=chunk_meta.to_dict()))
                    chunk_index += 1
                    current_chunk_text = ""
                    total_chars_in_chunk = 0
                
                # Split the long paragraph into sentence-sized chunks
                sentence_chunks = self._split_long_paragraph(para, document_id, source, metadata, chunk_index)
                for i, schunk in enumerate(sentence_chunks):
                    # Apply overlap between sentence chunks
                    if i > 0 and current_chunk_text:
                        overlap_text = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                        schunk_text = overlap_text + schunk.text
                    else:
                        schunk_text = schunk.text
                    
                    chunk_meta = ChunkMetadata(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        source=source,
                        chunk_start_char=current_chunk_start,
                        chunk_end_char=current_chunk_start + len(schunk_text)
                    )
                    if metadata:
                        chunk_meta_dict = chunk_meta.to_dict()
                        chunk_meta_dict.update(metadata)
                        chunks.append(Chunk(text=schunk_text, metadata=chunk_meta_dict))
                    else:
                        chunks.append(Chunk(text=schunk_text, metadata=chunk_meta.to_dict()))
                    
                    chunk_index += 1
                    current_chunk_start += len(schunk_text)
                    current_chunk_text = schunk_text
                    total_chars_in_chunk = len(schunk_text)
                # Reset current chunk after splitting long paragraph
                current_chunk_text = ""
                total_chars_in_chunk = 0
                continue
            
            # If adding this paragraph would exceed chunk size
            if total_chars_in_chunk + para_len > self.chunk_size and total_chars_in_chunk > 0:
                # Save current chunk
                chunk_meta = ChunkMetadata(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    source=source,
                    chunk_start_char=current_chunk_start,
                    chunk_end_char=current_chunk_start + total_chars_in_chunk
                )
                
                # Merge user metadata
                if metadata:
                    chunk_meta_dict = chunk_meta.to_dict()
                    chunk_meta_dict.update(metadata)
                    chunks.append(Chunk(text=current_chunk_text, metadata=chunk_meta_dict))
                else:
                    chunks.append(Chunk(text=current_chunk_text, metadata=chunk_meta.to_dict()))
                
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk_text = overlap_text + para
                current_chunk_start = current_chunk_start + total_chars_in_chunk - len(overlap_text)
                total_chars_in_chunk = self._count_chars(current_chunk_text)
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                    total_chars_in_chunk += 2
                else:
                    current_chunk_text = para
                total_chars_in_chunk += para_len
        
        # Don't forget the last chunk
        if current_chunk_text.strip():
            chunk_meta = ChunkMetadata(
                document_id=document_id,
                chunk_index=chunk_index,
                total_chunks=0,
                source=source,
                chunk_start_char=current_chunk_start,
                chunk_end_char=current_chunk_start + total_chars_in_chunk
            )
            
            if metadata:
                chunk_meta_dict = chunk_meta.to_dict()
                chunk_meta_dict.update(metadata)
                chunks.append(Chunk(text=current_chunk_text, metadata=chunk_meta_dict))
            else:
                chunks.append(Chunk(text=current_chunk_text, metadata=chunk_meta.to_dict()))
        
        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.info(f"Document {document_id} chunked into {len(chunks)} parts")
        return chunks
    
    def _split_long_paragraph(
        self,
        paragraph: str,
        document_id: str,
        source: Optional[str],
        metadata: Optional[Dict[str, Any]],
        start_index: int,
    ) -> List[Chunk]:
        """Split a very long paragraph into sentence-sized chunks.
        
        Priority:
        1. Sentence boundaries (Persian and English)
        2. Character limit as final boundary
        """
        # Try to split by sentences (Persian and English sentence boundaries)
        sentences = re.split(r'(?<=[.!?؟])\s+', paragraph)
        
        result = []
        current_text = ""
        
        for sentence in sentences:
            sent_len = len(sentence)
            
            if len(current_text) + sent_len > self.chunk_size and len(current_text) > 0:
                # Flush current chunk
                result.append(Chunk(text=current_text, metadata={}))
                # Start new chunk with overlap
                overlap = current_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_text = overlap + sentence
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence
        
        if current_text.strip():
            result.append(Chunk(text=current_text, metadata={}))
        
        # If sentence splitting didn't produce enough chunks, fall back to character splitting
        if len(result) <= 1 and len(paragraph) > self.chunk_size:
            result = []
            # Split by character limit
            start = 0
            while start < len(paragraph):
                end = min(start + self.chunk_size, len(paragraph))
                # Try to break at a sentence boundary
                if end < len(paragraph):
                    # Look for sentence end within the last 100 chars
                    search_region = paragraph[max(start, end - 100):end]
                    last_sentence_end = search_region.rfind('.')
                    if last_sentence_end == -1:
                        last_sentence_end = search_region.rfind('؟')
                    if last_sentence_end > 0:
                        end = max(start, end - 100) + last_sentence_end + 1
                chunk_text = paragraph[start:end]
                result.append(Chunk(text=chunk_text, metadata={}))
                start = end
                # Apply overlap
                if start < len(paragraph) and self.chunk_overlap > 0:
                    start -= self.chunk_overlap  # Backtrack for overlap
                    if start < 0:
                        start = 0
        
        return result if result else [Chunk(text=paragraph, metadata={})]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing while preserving original content.
        
        - Keep original Persian/Arabic characters intact
        - Normalize various whitespace characters
        - Preserve punctuation
        """
        # Replace various newlines with standard newline
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Keep the text intact - no case folding for Persian
        return text.strip()
    
    def chunk_document_generator(
        self,
        text: str,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 10
    ) -> Generator[List[Chunk], None, None]:
        """
        Generator that yields chunks in batches for memory efficiency.
        
        Args:
            text: The full document text
            document_id: Unique identifier for the document
            source: Source file/path of the document
            metadata: Additional metadata to include in each chunk
            batch_size: Number of chunks per batch
            
        Yields:
            Lists of Chunk objects (batches)
        """
        all_chunks = self.chunk_document(text, document_id, source, metadata)
        
        for i in range(0, len(all_chunks), batch_size):
            yield all_chunks[i:i + batch_size]


# Global instance
_chunker_service = None

def get_chunker_service(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> ChunkerService:
    """Get or create the chunker service instance."""
    global _chunker_service
    
    if _chunker_service is None:
        from app.config.settings import settings
        chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        _chunker_service = ChunkerService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return _chunker_service