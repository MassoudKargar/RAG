"""Tests for query rewriting and conversational context handling."""
import pytest
import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("PROVIDER", "openai")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.rag_service import RAGService
from app.models.chat import ChatMessage


class TestQueryRewriting:
    """Test Requirement 9: Query rewriting for conversational context."""
    
    def test_short_follow_up_query_rewritten(self):
        """Test: Short follow-up queries are combined with previous context."""
        rs = RAGService()
        
        messages = [
            ChatMessage(role="user", content="شرایط دریافت حق شغل چیست؟"),
            ChatMessage(role="assistant", content="پاسخ اول..."),
            ChatMessage(role="user", content="برای سال قبل چطور؟"),
        ]
        
        rewritten = rs.rewrite_query(messages)
        
        # Should combine the two queries
        assert "شرایط دریافت حق شغل" in rewritten
        assert "برای سال قبل" in rewritten
    
    def test_long_self_contained_query_not_rewritten(self):
        """Test: Long, self-contained queries should not be rewritten."""
        rs = RAGService()
        
        messages = [
            ChatMessage(role="user", content="شرایط دریافت حق شغل چیست؟"),
            ChatMessage(role="assistant", content="پاسخ اول..."),
            ChatMessage(role="user", content="معیارهای دقیق دریافت حق شغل بر اساس سال ۱۴۰۲ چیست؟"),
        ]
        
        rewritten = rs.rewrite_query(messages)
        
        # Long query should be used as-is (more than 7 words)
        assert rewritten == "معیارهای دقیق دریافت حق شغل بر اساس سال ۱۴۰۲ چیست؟"
    
    def test_single_message_not_rewritten(self):
        """Test: Single user message is returned as-is."""
        rs = RAGService()
        
        messages = [
            ChatMessage(role="user", content="سوال اولیه"),
        ]
        
        rewritten = rs.rewrite_query(messages)
        assert rewritten == "سوال اولیه"
    
    def test_empty_messages(self):
        """Test: Empty messages returns empty string."""
        rs = RAGService()
        rewritten = rs.rewrite_query([])
        assert rewritten == ""
    
    def test_persian_reference_words_trigger_rewrite(self):
        """Test: Persian reference words trigger query rewriting."""
        rs = RAGService()
        
        messages = [
            ChatMessage(role="user", content="قوانین مالکیت معنوی در ایران چیست؟"),
            ChatMessage(role="assistant", content="پاسخ..."),
            ChatMessage(role="user", content="و قوانین مرتبط در سال قبل؟"),
        ]
        
        rewritten = rs.rewrite_query(messages)
        assert len(rewritten) > len(messages[-1].content)
        assert "قوانین مالکیت معنوی" in rewritten
    
    def test_rewritten_query_preserves_persian(self):
        """Test: Rewritten query preserves Persian content."""
        rs = RAGService()
        
        messages = [
            ChatMessage(role="user", content="حقوق کارمندان در قانون کار چیست؟"),
            ChatMessage(role="assistant", content="پاسخ..."),
            ChatMessage(role="user", content="حداقل قانونی آنها چیست؟"),
        ]
        
        rewritten = rs.rewrite_query(messages)
        # Should contain Persian text
        assert any(ord(c) > 0x0600 for c in rewritten)  # Persian/Arabic characters