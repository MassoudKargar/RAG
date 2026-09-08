"""Document chunking service for large document support.

This service handles intelligent document chunking with semantic boundary detection.
It supports Persian/Arabic text and preserves document structure where possible.

Chunking strategy (priority order):
1. Sections / headings
2. Paragraphs
3. Sentences
4. Character limit as final boundary

The chunker is deterministic:
- a given document + document_id always produces the same chunks and the same
  chunk ids (``<document_id>::chunk_<index>``), which makes re-inserting a
  document safe (replace semantics in the vector store).
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a single chunk of a document."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


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


# Heading / section patterns shared by the paragraph splitter and the section
# detector. Order matters: markdown first, then Persian numbered, then English.
_SECTION_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$"),                                # markdown
    re.compile(r"^(?:بخش|سرفصل|فصل|قسمت)\s*[۰-۹0-9]*\s*[:：.\-]?\s*(.+)?$"),  # Persian
    re.compile(r"^(?:Section|Chapter)\s+[A-Za-z0-9]+[:\-.]?\s*(.*)$", re.IGNORECASE),  # English
    # SEC 10-K structure: "PART I", "Item 1.", "Item 7A."
    re.compile(r"^(?:PART)\s+[IVXLCDM]+\s*[:\.\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^Item\s+\d+[A-Za-z]?[:\.\-]?\s*(.*)$", re.IGNORECASE),
]

# SEC filings print the page number as a standalone line (e.g. "42" alone on
# a line) at page boundaries. Used to attach page metadata to chunks. Only
# lines whose number forms a plausible ascending page sequence are treated as
# page markers (avoids false positives like TOC/CIK numbers).
_PAGE_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*$")


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _extract_years_from_text(text: str) -> List[int]:
    """Return sorted unique fiscal years mentioned in a chunk's text."""
    if not text:
        return []
    t = text.translate(_PERSIAN_DIGITS)
    return sorted({int(m) for m in _YEAR_RE.findall(t)})


def _detect_section(paragraph: str) -> Optional[str]:
    """Return a section label for a paragraph that looks like a heading.

    Returns None when the paragraph is body text.
    """
    stripped = paragraph.strip()
    if not stripped:
        return None
    for pattern in _SECTION_PATTERNS:
        m = pattern.match(stripped)
        if m:
            # patterns have either group 1 (label) or group 2 (label) depending
            # on the pattern; fall back to the whole line
            try:
                label = (m.group(2) or m.group(1) or stripped).strip().lstrip(":#.-—– ").rstrip(":#.-—–")
            except IndexError:
                label = (m.group(1) or stripped).strip().lstrip(":#.-—– ").rstrip(":#.-—–")
            return label or stripped
    return None


class ChunkerService:
    """Service for intelligent document chunking with semantic boundaries."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _detect_pages(self, paragraphs: List[Tuple[int, int, str]]) -> List[Optional[int]]:
        """Return a list aligned with ``paragraphs`` giving the page number in
        effect at each paragraph (None when undetermined).

        Heuristic: standalone numeric lines (e.g. "42") are page markers when
        they form an ascending sequence (step 1..2). Numbers that jump wildly
        (TOC page refs, CIK numbers) break the run and are ignored.
        """
        pages: List[Optional[int]] = [None] * len(paragraphs)
        # collect candidate number lines with their paragraph indices
        candidates = []
        for i, (_, _, para) in enumerate(paragraphs):
            m = _PAGE_LINE_RE.match(para)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 999:
                    candidates.append((i, n))
        # find the longest run where numbers ascend by 1 (tolerating small gaps
        # of repeated/missing pages, but not wild jumps)
        best_run = []
        cur = []
        for i in range(len(candidates)):
            if not cur:
                cur = [candidates[i]]
                continue
            prev_idx, prev_n = cur[-1]
            cur_idx, cur_n = candidates[i]
            step = cur_n - prev_n
            if 0 <= step <= 2 and cur_idx - prev_idx <= 20:
                cur.append(candidates[i])
            else:
                if len(cur) >= 3 and len(cur) > len(best_run or []):
                    best_run = cur
                cur = [candidates[i]]
        if len(cur) >= 3 and len(cur) > len(best_run or []):
            best_run = cur

        if not best_run:
            return pages

        run_start_n, run_end_n = best_run[0][1], best_run[-1][1]
        for i, n in best_run:
            if run_start_n <= n <= run_end_n:
                pages[i] = n
        # forward-fill: paragraphs after a page marker inherit that page until
        # the next one
        last_page = None
        for i in range(len(paragraphs)):
            if pages[i] is not None:
                last_page = pages[i]
            pages[i] = last_page
        return pages

    def _split_into_paragraphs(self, text: str) -> List[Tuple[int, int, str]]:
        """Split text into (start, end, text) paragraphs including positions.

        Positions are relative to the normalized text. Handles:
        - CRLF / CR newlines
        - double-newline paragraph separators
        - bullet list items (kept inside their paragraph; we do not hard-split
          bullets because that destroys list context — sentence/char fallback
          still bounds the size)
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs: List[Tuple[int, int, str]] = []
        # A paragraph is a run of non-blank lines; blank lines ("\n\n") split it.
        for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
            start, end = match.start(), match.end()
            para = text[start:end].strip()
            if para and para != "\n":
                paragraphs.append((start, start + len(para), para))
        if not paragraphs and text.strip():
            paragraphs.append((0, len(text.strip()), text.strip()))
        return paragraphs

    def _split_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into sentences (Persian/Arabic + English aware).

        Boundaries: ``.`` ``!`` ``?`` ``؟`` followed by whitespace. If no
        sentence boundary is found the paragraph is returned as a single item.
        """
        parts = re.split(r"(?<=[.!?؟])\s+", paragraph)
        return [p for p in parts if p]

    def _split_long_paragraph(
        self,
        paragraph: str,
        chunk_size: int,
    ) -> List[Tuple[int, int, str]]:
        """Split one over-long paragraph into (start, end, text) sentence chunks.

        Sentences are grouped up to chunk_size; a sentence longer than
        chunk_size is hard-split at the character limit.
        """
        sentences = self._split_sentences(paragraph)
        result: List[Tuple[int, int, str]] = []
        current_text = ""
        current_start = 0

        def flush() -> None:
            nonlocal current_text, current_start
            if current_text:
                start = paragraph.find(current_text, current_start - len(current_text))
                if start < 0:
                    start = current_start
                result.append((start, start + len(current_text), current_text))
                current_text = ""
                current_start = start + 1

        for sentence in sentences:
            if len(sentence) > chunk_size:
                # Hard-split a single enormous sentence at the character limit
                flush()
                start = 0
                while start < len(sentence):
                    end = min(start + chunk_size, len(sentence))
                    result.append(
                        (0, 0, sentence[start:end])
                    )
                    start = end
                continue
            if current_text and len(current_text) + len(sentence) + 1 > chunk_size:
                flush()
            current_text = f"{current_text} {sentence}".strip() if current_text else sentence

        flush()

        if not result:
            # paragraph shorter than chunk_size but the caller still routed it here
            result = [(0, len(paragraph), paragraph)]
        return result

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
        1. Split into paragraphs at blank lines.
        2. Track the current section heading (markdown / English / Persian).
        3. Accumulate paragraphs per chunk; flush at chunk_size with overlap.
        4. A single paragraph larger than chunk_size is split at sentence
           boundaries, then at the character limit as a final fallback.

        Args:
            text: The full document text
            document_id: Unique identifier for the document
            source: Source file/path of the document
            metadata: Additional metadata to include in each chunk

        Returns:
            List of Chunk objects with text, metadata and deterministic ids.
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        if not text or not text.strip():
            logger.warning(f"Empty or whitespace-only document provided (document_id={document_id})")
            return []

        original = text
        text = self._normalize_text(text)
        paragraphs = self._split_into_paragraphs(text)

        chunks: List[Chunk] = []
        current_chunk_text = ""
        current_chunk_start = 0  # position in the normalized paragraph stream
        current_chunk_size = 0
        current_section: Optional[str] = None
        current_has_body = False
        chunk_index = 0

        def make_chunk(chunk_text: str, start_pos: int) -> Chunk:
            nonlocal chunk_index
            meta = ChunkMetadata(
                document_id=document_id,
                chunk_index=chunk_index,
                total_chunks=0,  # patched at the end
                source=source,
                section=current_section,
                page=current_page,
                chunk_start_char=start_pos,
                chunk_end_char=start_pos + len(chunk_text),
            )
            meta_dict = meta.to_dict()
            if metadata:
                meta_dict.update(metadata)  # user metadata wins on conflicts
            chunk = Chunk(
                text=chunk_text,
                metadata=meta_dict,
                id=f"{document_id}::chunk_{chunk_index:05d}",
            )
            chunk.metadata["chunk_id"] = chunk.id
            chunk_index += 1
            return chunk

        stream_pos = 0  # cursor over "\n\n".join(paragraph texts)
        first_para = True
        page_stream = self._detect_pages(paragraphs)
        current_page: Optional[int] = None

        for idx, (start, end, para) in enumerate(paragraphs):
            if page_stream[idx] is not None:
                current_page = page_stream[idx]
            # Section headings start a fresh chunk (semantic boundary priority #1)
            heading = _detect_section(para)
            if heading is not None and len(para) <= 200:
                # Repeated page headers for the SAME section (e.g. "PART I" /
                # "Item 1A" printed at the top of every page) are not new
                # sections: do not flush or re-set the section. Only a heading
                # DIFFERENT from the current one starts a fresh section.
                is_repeat = heading == current_section
                if not is_repeat:
                    if current_chunk_text.strip() and current_has_body:
                        # Flush so the heading binds to its own content
                        chunks.append(make_chunk(current_chunk_text, current_chunk_start))
                        current_chunk_text = ""
                        current_chunk_size = 0
                        current_has_body = False
                    current_section = heading

            sep = 0 if first_para else 2  # "\n\n" separator length
            first_para = False

            para_len = len(para)

            # Track whether the current chunk contains body text (not just headings)
            if heading is None:
                current_has_body = True

            # Single paragraph larger than chunk_size -> sentence-level split
            if para_len > self.chunk_size:
                # Flush pending chunk first
                if current_chunk_text.strip():
                    chunks.append(make_chunk(current_chunk_text, current_chunk_start))
                    current_chunk_text = ""
                    current_chunk_size = 0

                for s_start, s_end, sentence_text in self._split_long_paragraph(para, self.chunk_size):
                    if not sentence_text:
                        continue
                    # Apply overlap between consecutive sentence chunks
                    if current_chunk_text and self.chunk_overlap > 0:
                        overlap = current_chunk_text[-self.chunk_overlap:]
                        chunk_text = overlap + sentence_text
                    else:
                        chunk_text = sentence_text
                    chunks.append(make_chunk(chunk_text, max(0, current_chunk_start + (s_start or 0) - (len(overlap) if current_chunk_text and self.chunk_overlap > 0 else 0))))
                    current_chunk_text = sentence_text
                    current_chunk_size = len(sentence_text)
                    current_chunk_start = chunks[-1].metadata["chunk_end"]
                # After the long paragraph, start fresh
                current_chunk_text = ""
                current_chunk_size = 0
                stream_pos += sep + para_len
                continue

            if current_chunk_size > 0 and current_chunk_size + sep + para_len > self.chunk_size:
                # Flush and start a new chunk with overlap from the tail
                chunks.append(make_chunk(current_chunk_text, current_chunk_start))
                overlap = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                new_start = current_chunk_start + current_chunk_size
                if overlap:
                    current_chunk_text = overlap + para
                    current_chunk_start = new_start - len(overlap)
                    current_chunk_size = len(overlap) + para_len
                else:
                    current_chunk_text = para
                    current_chunk_start = new_start
                    current_chunk_size = para_len
            else:
                added = (sep if current_chunk_size > 0 else 0) + para_len
                if current_chunk_size == 0:
                    current_chunk_text = para
                else:
                    current_chunk_text += "\n\n" + para
                current_chunk_size += added

            stream_pos += sep + para_len

        if current_chunk_text.strip():
            chunks.append(make_chunk(current_chunk_text, current_chunk_start))

        # Patch total_chunks on every chunk (and keep user metadata last so
        # document_id/chunk_index/total_chunks always reflect the truth)
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total
            # ensure the canonical keys are never overridden by user metadata
            chunk.metadata["document_id"] = document_id
            chunk.metadata["chunk_index"] = chunk.metadata.get("chunk_index", 0)
            # chunks before the first detected heading/page marker (cover page,
            # TOC, preamble) get a stable fallback so audits see a section/page
            if not chunk.metadata.get("section"):
                chunk.metadata["section"] = "Cover and Index"
            if chunk.metadata.get("page") is None:
                chunk.metadata["page"] = 1
            # fiscal years explicitly present in this chunk's text (used to
            # match multi-year tables: a table row with FY2024/FY2025/FY2026
            # values lives under one chunk but must answer any of those years)
            years = _extract_years_from_text(chunk.text)
            if years:
                chunk.metadata["years_present"] = years

        logger.info(
            "Document %s chunked: %d chunks (size=%d, overlap=%d)",
            document_id, total, self.chunk_size, self.chunk_overlap,
        )
        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize line endings/whitespace for processing.

        The original text is preserved in memory; this normalized copy is only
        used for chunk boundary decisions. Returned chunks contain the
        original (normalized line-ending) text.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
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
        chunk_size = chunk_size if chunk_size is not None else settings.RAG_CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.RAG_CHUNK_OVERLAP
        _chunker_service = ChunkerService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return _chunker_service