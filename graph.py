packages/sta_agent_engine/src/sta_agent_engine/models/__init__.py
----
"""Model factory functions for LLM, embedding, and rerank providers."""

from .custom_chat_model import CustomChatModel, StreamingChatOpenAI, create_chat_model
from .embedding_model import create_embedding_model
from .fake_streaming_llm import FakeStreamingLLM, create_fake_streaming_llm

# LiteLLM-based rerank (for providers with native LiteLLM support)
from .litellm_rerank_model import (
    create_async_rerank_model as create_async_litellm_rerank,
    create_rerank_model as create_litellm_rerank,
)

# HTTP-based rerank (default - handles response format differences across providers)
from .rerank_model import (
    RerankClient,
    RerankResponse,
    RerankResult,
    create_async_rerank_model,
    create_rerank_model,
)


__all__ = [
    # Chat
    "create_chat_model",
    "CustomChatModel",
    "StreamingChatOpenAI",
    # Fake Streaming (for custom message streaming via stream_mode="messages")
    "FakeStreamingLLM",
    "create_fake_streaming_llm",
    # Embedding
    "create_embedding_model",
    # Rerank (HTTP-based - default)
    "create_rerank_model",
    "create_async_rerank_model",
    "RerankClient",
    "RerankResponse",
    "RerankResult",
    # Rerank (LiteLLM-based)
    "create_litellm_rerank",
    "create_async_litellm_rerank",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/models/fake_streaming_llm.py
----
"""
Fake Streaming LLM for custom message streaming.

This module provides a fake LLM that streams pre-defined content, allowing
custom messages to be streamed via `stream_mode="messages"` in LangGraph.

Use Case:
- Stream fallback messages when no data is found
- Stream cached responses with streaming UX
- Stream database query results as if from an LLM
- Testing streaming behavior without real LLM calls

Why this is needed:
- `writer({"messages": [AIMessageChunk]})` does NOT emit to the "messages" channel
- The "messages" channel only captures `on_chat_model_stream` events from LLMs
- This fake LLM properly implements `_astream()` to emit those events

Example:
    ```python
    from sta_agent_engine.models.fake_streaming_llm import FakeStreamingLLM

    async def my_node(state: State) -> dict:
        llm = FakeStreamingLLM(
            content_to_stream="❌ No ticket found in database.",
            chunk_size=2,
            delay=0.02,
        )
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}
    ```
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


logger = logging.getLogger(__name__)


class FakeStreamingLLM(BaseChatModel):
    """
    A fake LLM that streams pre-defined content.

    This class properly implements `_astream()` so that `stream_mode="messages"`
    works in LangGraph, enabling custom messages to be streamed to the frontend
    exactly like real LLM responses.

    Attributes:
        content_to_stream: The text content to stream.
        chunk_size: Number of characters per chunk (default: 1).
        delay: Delay in seconds between chunks (default: 0).

    Example:
        ```python
        llm = FakeStreamingLLM(
            content_to_stream="No items found.",
            chunk_size=2,
            delay=0.03,
        )

        # Use in a LangGraph node
        async def fallback_node(state):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}

        # Stream will work with stream_mode="messages"
        async for mode, chunk in graph.astream(inputs, config, stream_mode=["messages"]):
            if mode == "messages":
                msg, _ = chunk
                print(msg.content, end="")
        ```
    """

    content_to_stream: str = "No data found."
    chunk_size: int = 1
    delay: float = 0

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "fake_streaming"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - returns full content at once."""
        logger.debug(
            "FakeStreamingLLM._generate called with %d messages",
            len(messages),
        )
        generation = ChatGeneration(message=AIMessage(content=self.content_to_stream))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming - yields ChatGenerationChunk objects."""
        logger.debug(
            "FakeStreamingLLM._stream called, streaming %d chars in chunks of %d",
            len(self.content_to_stream),
            self.chunk_size,
        )
        for i in range(0, len(self.content_to_stream), self.chunk_size):
            chunk_text = self.content_to_stream[i : i + self.chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async streaming - THIS IS WHAT MAKES stream_mode='messages' WORK.

        Yields ChatGenerationChunk objects with delays between chunks,
        properly triggering on_chat_model_stream events that LangGraph
        captures in the "messages" stream mode.
        """
        logger.debug(
            "FakeStreamingLLM._astream called, streaming %d chars in chunks of %d with %.3fs delay",
            len(self.content_to_stream),
            self.chunk_size,
            self.delay,
        )
        for i in range(0, len(self.content_to_stream), self.chunk_size):
            chunk_text = self.content_to_stream[i : i + self.chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if self.delay > 0:
                await asyncio.sleep(self.delay)


def create_fake_streaming_llm(
    content: str,
    chunk_size: int = 1,
    delay: float = 0,
) -> FakeStreamingLLM:
    """
    Factory function to create a FakeStreamingLLM instance.

    Args:
        content: The text content to stream.
        chunk_size: Number of characters per chunk (default: 1).
        delay: Delay in seconds between chunks (default: 0).

    Returns:
        Configured FakeStreamingLLM instance.

    Example:
        ```python
        llm = create_fake_streaming_llm(
            content="❌ No ticket found. Please verify your ticket ID.",
            chunk_size=2,
            delay=0.03,
        )
        response = await llm.ainvoke([HumanMessage("check ticket")])
        ```
    """
    return FakeStreamingLLM(
        content_to_stream=content,
        chunk_size=chunk_size,
        delay=delay,
    )

-------

tests/test_ai_engine/models/__init__.py
----
"""Tests for sta_agent_engine.models module."""


-------

tests/test_ai_engine/models/test_fake_streaming_llm.py
----
"""
Unit tests for FakeStreamingLLM.

Tests cover:
- Synchronous generation
- Synchronous streaming
- Asynchronous streaming
- Integration with LangGraph stream_mode="messages"
"""

import asyncio
import math
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import RunnableConfig

from sta_agent_engine.models.fake_streaming_llm import FakeStreamingLLM


def expected_chunk_count(content: str, chunk_size: int) -> int:
    """Calculate the expected number of chunks for given content and chunk_size."""
    if not content:
        return 0
    return math.ceil(len(content) / chunk_size)


class TestFakeStreamingLLMUnit:
    """Unit tests for FakeStreamingLLM class."""

    def test_llm_type(self) -> None:
        """Test that _llm_type returns correct identifier."""
        llm = FakeStreamingLLM()
        assert llm._llm_type == "fake_streaming"

    def test_default_values(self) -> None:
        """Test default attribute values."""
        llm = FakeStreamingLLM()
        assert llm.content_to_stream == "No data found."
        assert llm.chunk_size == 1
        assert llm.delay == 0

    def test_custom_values(self) -> None:
        """Test custom attribute values."""
        llm = FakeStreamingLLM(
            content_to_stream="Custom content",
            chunk_size=5,
            delay=0.1,
        )
        assert llm.content_to_stream == "Custom content"
        assert llm.chunk_size == 5
        assert llm.delay == 0.1

    def test_generate_returns_full_content(self) -> None:
        """Test that _generate returns the full content."""
        content = "Test message"
        llm = FakeStreamingLLM(content_to_stream=content)

        result = llm._generate(messages=[HumanMessage(content="input")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == content
        assert isinstance(result.generations[0].message, AIMessage)

    def test_stream_yields_correct_chunk_count(self) -> None:
        """Test that _stream yields exactly the expected number of chunks."""
        content = "Hello"  # 5 chars
        chunk_size = 2
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks = list(llm._stream(messages=[HumanMessage(content="input")]))

        # 5 chars / 2 chunk_size = 3 chunks: "He", "ll", "o"
        assert len(chunks) == expected_chunk_count(content, chunk_size)
        assert len(chunks) == 3
        assert all(isinstance(c, ChatGenerationChunk) for c in chunks)
        assert chunks[0].message.content == "He"
        assert chunks[1].message.content == "ll"
        assert chunks[2].message.content == "o"

    def test_stream_single_char_chunks(self) -> None:
        """Test streaming with single character chunks."""
        content = "ABC"  # 3 chars
        chunk_size = 1
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks = list(llm._stream(messages=[]))

        assert len(chunks) == expected_chunk_count(content, chunk_size)
        assert len(chunks) == 3
        assert [c.message.content for c in chunks] == ["A", "B", "C"]

    def test_stream_exact_division(self) -> None:
        """Test streaming when content length divides evenly by chunk_size."""
        content = "ABCDEF"  # 6 chars
        chunk_size = 2
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks = list(llm._stream(messages=[]))

        # 6 / 2 = 3 chunks exactly
        assert len(chunks) == 3
        assert [c.message.content for c in chunks] == ["AB", "CD", "EF"]

    @pytest.mark.asyncio
    async def test_astream_yields_correct_chunk_count(self) -> None:
        """Test that _astream yields exactly the expected number of chunks."""
        content = "Test"  # 4 chars
        chunk_size = 2
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks = []
        async for chunk in llm._astream(messages=[]):
            chunks.append(chunk)

        # 4 chars / 2 chunk_size = 2 chunks: "Te", "st"
        assert len(chunks) == expected_chunk_count(content, chunk_size)
        assert len(chunks) == 2
        assert chunks[0].message.content == "Te"
        assert chunks[1].message.content == "st"

    @pytest.mark.asyncio
    async def test_astream_respects_delay(self) -> None:
        """Test that _astream respects the delay parameter."""
        content = "AB"  # 2 chars
        chunk_size = 1
        delay = 0.05
        llm = FakeStreamingLLM(
            content_to_stream=content,
            chunk_size=chunk_size,
            delay=delay,
        )

        start = asyncio.get_event_loop().time()
        chunks = []
        async for chunk in llm._astream(messages=[]):
            chunks.append(chunk)
        elapsed = asyncio.get_event_loop().time() - start

        assert len(chunks) == 2
        # 2 chunks * 0.05s delay = 0.1s minimum (with tolerance for execution time)
        expected_min_time = (len(chunks) * delay) * 0.8  # 80% tolerance
        assert elapsed >= expected_min_time

    @pytest.mark.asyncio
    async def test_astream_zero_delay(self) -> None:
        """Test that _astream works with zero delay."""
        content = "Fasts"  # 4 chars
        chunk_size = 2
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size, delay=0)

        chunks = []
        async for chunk in llm._astream(messages=[]):
            chunks.append(chunk)

        assert len(chunks) == expected_chunk_count(content, chunk_size)
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_ainvoke_returns_full_message(self) -> None:
        """Test that ainvoke returns the complete message."""
        content = "Complete response"
        llm = FakeStreamingLLM(content_to_stream=content)

        result = await llm.ainvoke([HumanMessage(content="test")])

        assert isinstance(result, AIMessage)
        assert result.content == content


class TestFakeStreamingLLMIntegration:
    """Integration tests with LangGraph."""

    @pytest.mark.asyncio
    async def test_langgraph_stream_mode_messages(self) -> None:
        """
        Test that FakeStreamingLLM works with stream_mode='messages'.

        This is the main use case - streaming custom content to a frontend
        that expects stream_mode='messages'.
        """
        content = "No ticket found."  # 16 chars
        chunk_size = 3

        class State(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]

        async def fallback_node(state: State) -> dict:
            llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(State)
        graph.add_node("fallback", fallback_node)
        graph.add_edge(START, "fallback")
        graph.add_edge("fallback", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config: RunnableConfig = {"configurable": {"thread_id": "test-integration"}}
        inputs: State = {"messages": [HumanMessage(content="check ticket")]}

        chunks_received: list[str] = []
        async for mode, chunk in compiled.astream(inputs, config, stream_mode=["messages"]):
            if mode == "messages":
                msg_chunk, _ = chunk
                if hasattr(msg_chunk, "content") and msg_chunk.content:
                    chunks_received.append(msg_chunk.content)

        # Expected: ceil(16 / 3) = 6 chunks: "No ", "tic", "ket", " fo", "und", "."
        assert len(chunks_received) == expected_chunk_count(content, chunk_size)
        assert len(chunks_received) == 6

        # Verify accumulated content matches original
        accumulated = "".join(chunks_received)
        assert accumulated == content

        # Verify individual chunks
        assert chunks_received == ["No ", "tic", "ket", " fo", "und", "."]

    @pytest.mark.asyncio
    async def test_langgraph_metadata_contains_node_info(self) -> None:
        """Test that streaming metadata contains correct node information."""
        content = "Hi"  # 2 chars
        chunk_size = 1

        class State(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]

        async def my_node(state: State) -> dict:
            llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(State)
        graph.add_node("my_node", my_node)
        graph.add_edge(START, "my_node")
        graph.add_edge("my_node", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config: RunnableConfig = {"configurable": {"thread_id": "test-metadata"}}
        inputs: State = {"messages": [HumanMessage(content="test")]}

        metadata_collected: list[dict] = []
        async for mode, chunk in compiled.astream(inputs, config, stream_mode=["messages"]):
            if mode == "messages":
                _, metadata = chunk
                metadata_collected.append(metadata)

        # Should have at least 2 metadata entries (LangGraph may emit extra events)
        assert len(metadata_collected) >= expected_chunk_count(content, chunk_size)

        # Each metadata should contain langgraph_node with correct value
        for meta in metadata_collected:
            assert "langgraph_node" in meta
            assert meta["langgraph_node"] == "my_node"

    @pytest.mark.asyncio
    async def test_langgraph_with_multiple_stream_modes(self) -> None:
        """Test streaming with both 'messages' and 'updates' modes."""
        content = "Done"  # 4 chars
        chunk_size = 2

        class State(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]

        async def process_node(state: State) -> dict:
            llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(State)
        graph.add_node("process", process_node)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config: RunnableConfig = {"configurable": {"thread_id": "test-multi-mode"}}
        inputs: State = {"messages": [HumanMessage(content="go")]}

        message_chunks: list[str] = []
        updates_received: list[str] = []

        async for mode, chunk in compiled.astream(inputs, config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                msg_chunk, _ = chunk
                if hasattr(msg_chunk, "content") and msg_chunk.content:
                    message_chunks.append(msg_chunk.content)
            elif mode == "updates":
                updates_received.extend(chunk.keys())

        # Expected: ceil(4 / 2) = 2 message chunks
        assert len(message_chunks) == 2
        assert message_chunks == ["Do", "ne"]

        # Should receive update for the process node
        assert "process" in updates_received


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_content(self) -> None:
        """Test with empty content string."""
        llm = FakeStreamingLLM(content_to_stream="")

        result = llm._generate(messages=[])
        assert result.generations[0].message.content == ""

        chunks = list(llm._stream(messages=[]))
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_unicode_content(self) -> None:
        """Test with unicode/emoji content."""
        content = "❌ 日本語 🎉"  # 7 chars: ❌, space, 日, 本, 語, space, 🎉
        chunk_size = 1
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks: list[str] = []
        async for chunk in llm._astream(messages=[]):
            chunks.append(chunk.message.content)

        assert len(chunks) == expected_chunk_count(content, chunk_size)
        assert len(chunks) == len(content)  # 7 chars = 7 chunks
        accumulated = "".join(chunks)
        assert accumulated == content

    def test_large_chunk_size(self) -> None:
        """Test with chunk_size larger than content."""
        content = "Short"  # 5 chars
        chunk_size = 100
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks = list(llm._stream(messages=[]))

        # Single chunk since chunk_size > content length
        assert len(chunks) == 1
        assert chunks[0].message.content == content

    def test_chunk_size_equals_content_length(self) -> None:
        """Test when chunk_size exactly equals content length."""
        content = "Exact"  # 5 chars
        chunk_size = 5
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)

        chunks = list(llm._stream(messages=[]))

        assert len(chunks) == 1
        assert chunks[0].message.content == content

    @pytest.mark.asyncio
    async def test_single_char_content(self) -> None:
        """Test with single character content."""
        content = "X"
        llm = FakeStreamingLLM(content_to_stream=content)

        chunks = []
        async for chunk in llm._astream(messages=[]):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].message.content == "X"

    @pytest.mark.parametrize(
        "content,chunk_size,expected_chunks",
        [
            ("Hello", 1, 5),
            ("Hello", 2, 3),
            ("Hello", 3, 2),
            ("Hello", 5, 1),
            ("Hello", 10, 1),
            ("ABCDEF", 2, 3),
            ("ABCDEF", 3, 2),
            ("A", 1, 1),
            ("AB", 1, 2),
        ],
    )
    def test_chunk_count_parametrized(self, content: str, chunk_size: int, expected_chunks: int) -> None:
        """Parametrized test for various content/chunk_size combinations."""
        llm = FakeStreamingLLM(content_to_stream=content, chunk_size=chunk_size)
        chunks = list(llm._stream(messages=[]))
        assert len(chunks) == expected_chunks

-------

