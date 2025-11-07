# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Anthropic API models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lionherd.services.third_party.anthropic_models import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockResponse,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    CreateMessageRequest,
    CreateMessageResponse,
    ImageContentBlock,
    ImageSource,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    StreamEvent,
    TextContentBlock,
    ToolChoice,
    ToolDefinition,
    Usage,
)

# ============================================================================
# TextContentBlock Tests
# ============================================================================


class TestTextContentBlock:
    """Test TextContentBlock model."""

    def test_text_content_when_minimal_then_succeeds(self):
        """Test minimal text content block."""
        block = TextContentBlock(text="Hello")
        assert block.type == "text"
        assert block.text == "Hello"
        assert block.cache_control is None

    def test_text_content_when_with_cache_control_then_succeeds(self):
        """Test text content with cache control."""
        block = TextContentBlock(text="Hello", cache_control={"type": "ephemeral"})
        assert block.cache_control == {"type": "ephemeral"}

    def test_text_content_when_type_mismatch_then_fails(self):
        """Test type field must be 'text'."""
        with pytest.raises(ValidationError):
            TextContentBlock(type="image", text="Hello")


# ============================================================================
# ImageSource Tests
# ============================================================================


class TestImageSource:
    """Test ImageSource model."""

    def test_image_source_when_valid_jpeg_then_succeeds(self):
        """Test image source with JPEG."""
        source = ImageSource(media_type="image/jpeg", data="base64data")
        assert source.type == "base64"
        assert source.media_type == "image/jpeg"
        assert source.data == "base64data"

    def test_image_source_when_valid_png_then_succeeds(self):
        """Test image source with PNG."""
        source = ImageSource(media_type="image/png", data="base64data")
        assert source.media_type == "image/png"

    def test_image_source_when_invalid_media_type_then_fails(self):
        """Test invalid media type fails."""
        with pytest.raises(ValidationError):
            ImageSource(media_type="image/bmp", data="data")


# ============================================================================
# ImageContentBlock Tests
# ============================================================================


class TestImageContentBlock:
    """Test ImageContentBlock model."""

    def test_image_content_when_valid_source_then_succeeds(self):
        """Test image content with valid source."""
        source = ImageSource(media_type="image/jpeg", data="data")
        block = ImageContentBlock(source=source)
        assert block.type == "image"
        assert block.source.media_type == "image/jpeg"

    def test_image_content_when_dict_source_then_converts(self):
        """Test image content with dict source."""
        block = ImageContentBlock(
            source={"type": "base64", "media_type": "image/png", "data": "data"}
        )
        assert block.source.media_type == "image/png"


# ============================================================================
# Message Tests
# ============================================================================


class TestMessage:
    """Test Message model."""

    def test_message_when_string_content_then_succeeds(self):
        """Test message with string content."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_when_list_of_strings_then_converts(self):
        """Test message with list of strings converts to content blocks."""
        msg = Message(role="user", content=["Hello", "World"])
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        # Validator converts strings to TextContentBlock dicts, then Pydantic converts to models
        # Check if it's a dict or TextContentBlock object
        if isinstance(msg.content[0], dict):
            assert msg.content[0]["type"] == "text"
            assert msg.content[0]["text"] == "Hello"
        else:
            # It's a TextContentBlock object
            assert msg.content[0].type == "text"
            assert msg.content[0].text == "Hello"

    def test_message_when_list_of_content_blocks_then_succeeds(self):
        """Test message with content block list."""
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "data",
                    },
                },
            ],
        )
        assert len(msg.content) == 2

    def test_message_when_assistant_role_then_succeeds(self):
        """Test message with assistant role."""
        msg = Message(role="assistant", content="Response")
        assert msg.role == "assistant"

    def test_message_when_invalid_role_then_fails(self):
        """Test invalid role fails."""
        with pytest.raises(ValidationError):
            Message(role="system", content="Hello")


# ============================================================================
# ToolDefinition Tests
# ============================================================================


class TestToolDefinition:
    """Test ToolDefinition model."""

    def test_tool_def_when_valid_name_then_succeeds(self):
        """Test tool definition with valid name."""
        tool = ToolDefinition(name="get_weather", input_schema={"type": "object"})
        assert tool.name == "get_weather"
        assert tool.description is None

    def test_tool_def_when_with_description_then_succeeds(self):
        """Test tool definition with description."""
        tool = ToolDefinition(
            name="get_weather",
            description="Gets weather",
            input_schema={"type": "object"},
        )
        assert tool.description == "Gets weather"

    def test_tool_def_when_empty_name_then_fails(self):
        """Test empty name fails."""
        with pytest.raises(ValidationError):
            ToolDefinition(name="", input_schema={})

    def test_tool_def_when_name_too_long_then_fails(self):
        """Test name exceeding max length fails."""
        with pytest.raises(ValidationError):
            ToolDefinition(name="a" * 65, input_schema={})

    def test_tool_def_when_invalid_name_pattern_then_fails(self):
        """Test invalid name pattern fails."""
        with pytest.raises(ValidationError):
            ToolDefinition(name="invalid name!", input_schema={})


# ============================================================================
# ToolChoice Tests
# ============================================================================


class TestToolChoice:
    """Test ToolChoice model."""

    def test_tool_choice_when_auto_then_succeeds(self):
        """Test tool choice with auto type."""
        choice = ToolChoice(type="auto")
        assert choice.type == "auto"
        assert choice.name is None

    def test_tool_choice_when_tool_with_name_then_succeeds(self):
        """Test tool choice with specific tool."""
        choice = ToolChoice(type="tool", name="get_weather")
        assert choice.type == "tool"
        assert choice.name == "get_weather"

    def test_tool_choice_when_invalid_type_then_fails(self):
        """Test invalid type fails."""
        with pytest.raises(ValidationError):
            ToolChoice(type="invalid")


# ============================================================================
# CreateMessageRequest Tests
# ============================================================================


class TestCreateMessageRequest:
    """Test CreateMessageRequest model."""

    def test_request_when_minimal_then_succeeds(self):
        """Test minimal valid request."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        assert req.model == "claude-3-sonnet"
        assert len(req.messages) == 1
        assert req.max_tokens == 100

    def test_request_when_missing_model_then_fails(self):
        """Test missing model fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(messages=[{"role": "user", "content": "Hello"}], max_tokens=100)

    def test_request_when_missing_messages_then_fails(self):
        """Test missing messages fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(model="claude-3-sonnet", max_tokens=100)

    def test_request_when_missing_max_tokens_then_fails(self):
        """Test missing max_tokens fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(
                model="claude-3-sonnet", messages=[{"role": "user", "content": "Hi"}]
            )

    def test_request_when_zero_max_tokens_then_fails(self):
        """Test zero max_tokens fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(
                model="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=0,
            )

    def test_request_when_with_system_string_then_succeeds(self):
        """Test system prompt as string."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            system="You are helpful",
        )
        assert req.system == "You are helpful"

    def test_request_when_with_system_blocks_then_succeeds(self):
        """Test system prompt as content blocks."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            system=[{"type": "text", "text": "You are helpful"}],
        )
        assert isinstance(req.system, list)

    def test_request_when_temperature_in_range_then_succeeds(self):
        """Test temperature within valid range."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.5,
        )
        assert req.temperature == 0.5

    def test_request_when_temperature_out_of_range_then_fails(self):
        """Test temperature outside valid range fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(
                model="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                temperature=1.5,
            )

    def test_request_when_top_p_in_range_then_succeeds(self):
        """Test top_p within valid range."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            top_p=0.9,
        )
        assert req.top_p == 0.9

    def test_request_when_top_p_out_of_range_then_fails(self):
        """Test top_p outside valid range fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(
                model="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                top_p=1.5,
            )

    def test_request_when_top_k_negative_then_fails(self):
        """Test negative top_k fails."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(
                model="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                top_k=-1,
            )

    def test_request_when_with_tools_then_succeeds(self):
        """Test request with tools."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        )
        assert len(req.tools) == 1

    def test_request_when_with_tool_choice_then_succeeds(self):
        """Test request with tool choice."""
        req = CreateMessageRequest(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            tool_choice={"type": "auto"},
        )
        # Tool choice is converted to ToolChoice model
        assert req.tool_choice.type == "auto"

    def test_request_when_extra_field_then_fails(self):
        """Test extra fields forbidden."""
        with pytest.raises(ValidationError):
            CreateMessageRequest(
                model="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                extra_field="invalid",
            )


# ============================================================================
# Usage Tests
# ============================================================================


class TestUsage:
    """Test Usage model."""

    def test_usage_when_valid_then_succeeds(self):
        """Test valid usage."""
        usage = Usage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_usage_when_missing_field_then_fails(self):
        """Test missing fields fail."""
        with pytest.raises(ValidationError):
            Usage(input_tokens=100)


# ============================================================================
# ContentBlockResponse Tests
# ============================================================================


class TestContentBlockResponse:
    """Test ContentBlockResponse model."""

    def test_content_block_response_when_valid_then_succeeds(self):
        """Test valid content block response."""
        block = ContentBlockResponse(type="text", text="Hello")
        assert block.type == "text"
        assert block.text == "Hello"


# ============================================================================
# CreateMessageResponse Tests
# ============================================================================


class TestCreateMessageResponse:
    """Test CreateMessageResponse model."""

    def test_response_when_minimal_then_succeeds(self):
        """Test minimal valid response."""
        resp = CreateMessageResponse(
            id="msg_123",
            content=[{"type": "text", "text": "Hello"}],
            model="claude-3-sonnet",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert resp.id == "msg_123"
        assert resp.type == "message"
        assert resp.role == "assistant"

    def test_response_when_with_stop_reason_then_succeeds(self):
        """Test response with stop reason."""
        resp = CreateMessageResponse(
            id="msg_123",
            content=[{"type": "text", "text": "Hello"}],
            model="claude-3-sonnet",
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason="end_turn",
        )
        assert resp.stop_reason == "end_turn"

    def test_response_when_invalid_stop_reason_then_fails(self):
        """Test invalid stop reason fails."""
        with pytest.raises(ValidationError):
            CreateMessageResponse(
                id="msg_123",
                content=[{"type": "text", "text": "Hello"}],
                model="claude-3-sonnet",
                usage={"input_tokens": 10, "output_tokens": 5},
                stop_reason="invalid_reason",
            )


# ============================================================================
# Streaming Event Tests
# ============================================================================


class TestMessageStartEvent:
    """Test MessageStartEvent model."""

    def test_message_start_when_valid_then_succeeds(self):
        """Test valid message start event."""
        event = MessageStartEvent(
            message={
                "id": "msg_123",
                "content": [],
                "model": "claude-3-sonnet",
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        )
        assert event.type == "message_start"
        assert event.message.id == "msg_123"


class TestContentBlockStartEvent:
    """Test ContentBlockStartEvent model."""

    def test_content_block_start_when_valid_then_succeeds(self):
        """Test valid content block start."""
        event = ContentBlockStartEvent(index=0, content_block={"type": "text", "text": ""})
        assert event.type == "content_block_start"
        assert event.index == 0


class TestContentBlockDeltaEvent:
    """Test ContentBlockDeltaEvent model."""

    def test_content_block_delta_when_valid_then_succeeds(self):
        """Test valid content block delta."""
        event = ContentBlockDeltaEvent(index=0, delta={"type": "text_delta", "text": "H"})
        assert event.type == "content_block_delta"
        assert event.delta["text"] == "H"


class TestContentBlockStopEvent:
    """Test ContentBlockStopEvent model."""

    def test_content_block_stop_when_valid_then_succeeds(self):
        """Test valid content block stop."""
        event = ContentBlockStopEvent(index=0)
        assert event.type == "content_block_stop"
        assert event.index == 0


class TestMessageDeltaEvent:
    """Test MessageDeltaEvent model."""

    def test_message_delta_when_valid_then_succeeds(self):
        """Test valid message delta."""
        event = MessageDeltaEvent(
            delta={"stop_reason": "end_turn"},
            usage={"input_tokens": 0, "output_tokens": 10},
        )
        assert event.type == "message_delta"
        assert event.delta["stop_reason"] == "end_turn"

    def test_message_delta_when_no_usage_then_succeeds(self):
        """Test message delta without usage."""
        event = MessageDeltaEvent(delta={"stop_reason": "end_turn"})
        assert event.usage is None


class TestMessageStopEvent:
    """Test MessageStopEvent model."""

    def test_message_stop_when_valid_then_succeeds(self):
        """Test valid message stop."""
        event = MessageStopEvent()
        assert event.type == "message_stop"


# ============================================================================
# Union Type Tests
# ============================================================================


class TestStreamEventUnion:
    """Test StreamEvent union type."""

    def test_stream_event_when_message_start_then_valid(self):
        """Test StreamEvent with MessageStartEvent."""
        event = MessageStartEvent(
            message={
                "id": "msg_123",
                "content": [],
                "model": "claude-3-sonnet",
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        )
        assert isinstance(event, MessageStartEvent)

    def test_stream_event_when_content_block_delta_then_valid(self):
        """Test StreamEvent with ContentBlockDeltaEvent."""
        event = ContentBlockDeltaEvent(index=0, delta={"text": "H"})
        assert isinstance(event, ContentBlockDeltaEvent)

    def test_stream_event_discriminates_message_start_from_dict(self):
        """Test StreamEvent parses MessageStartEvent from raw dict."""
        raw_data = {
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "content": [],
                "model": "claude-3-opus",
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
        }
        event = MessageStartEvent.model_validate(raw_data)
        assert isinstance(event, MessageStartEvent)
        assert event.type == "message_start"
        assert event.message.id == "msg_abc"

    def test_stream_event_discriminates_content_block_delta_from_dict(self):
        """Test StreamEvent parses ContentBlockDeltaEvent from raw dict."""
        raw_data = {"type": "content_block_delta", "index": 0, "delta": {"text": "Hello"}}
        event = ContentBlockDeltaEvent.model_validate(raw_data)
        assert isinstance(event, ContentBlockDeltaEvent)
        assert event.type == "content_block_delta"
        assert event.delta["text"] == "Hello"


class TestContentBlockUnion:
    """Test ContentBlock union type."""

    def test_content_block_when_text_then_valid(self):
        """Test ContentBlock with TextContentBlock."""
        block = TextContentBlock(text="Hello")
        assert isinstance(block, TextContentBlock)

    def test_content_block_when_image_then_valid(self):
        """Test ContentBlock with ImageContentBlock."""
        block = ImageContentBlock(
            source={"type": "base64", "media_type": "image/jpeg", "data": "data"}
        )
        assert isinstance(block, ImageContentBlock)

    def test_content_block_discriminates_text_from_dict(self):
        """Test ContentBlock parses text block from raw dict via type discriminator."""
        raw_data = {"type": "text", "text": "Hello world"}
        block = TextContentBlock.model_validate(raw_data)
        assert isinstance(block, TextContentBlock)
        assert block.type == "text"
        assert block.text == "Hello world"

    def test_content_block_discriminates_image_from_dict(self):
        """Test ContentBlock parses image block from raw dict via type discriminator."""
        raw_data = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }
        block = ImageContentBlock.model_validate(raw_data)
        assert isinstance(block, ImageContentBlock)
        assert block.type == "image"
        assert block.source.data == "abc123"
