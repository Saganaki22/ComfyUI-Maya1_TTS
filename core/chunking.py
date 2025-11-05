"""
Smart text chunking utilities for Maya1 TTS.
Handles splitting long texts into manageable chunks for generation.
"""

import re
from typing import List


def smart_chunk_text(text: str, max_chunk_chars: int = 200) -> List[str]:
    """
    Split text into chunks at sentence boundaries for natural TTS.

    Tries to split at:
    1. Sentence boundaries (. ! ?)
    2. Clause boundaries (, ; :)
    3. Word boundaries (spaces)

    Args:
        text: Full text to chunk
        max_chunk_chars: Maximum characters per chunk (soft limit)

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_chars:
        return [text]

    chunks = []

    # Split on sentence boundaries first
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    current_chunk = ""

    for sentence in sentences:
        # If sentence itself is too long, split it further
        if len(sentence) > max_chunk_chars:
            # First, save current chunk if exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split long sentence on clause boundaries
            clause_pattern = r'(?<=[,;:])\s+'
            clauses = re.split(clause_pattern, sentence)

            for clause in clauses:
                # If clause is still too long, split on words
                if len(clause) > max_chunk_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""

                    words = clause.split()
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_chunk_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            current_chunk += (" " if current_chunk else "") + word
                else:
                    # Add clause to current chunk
                    if len(current_chunk) + len(clause) + 1 > max_chunk_chars:
                        chunks.append(current_chunk.strip())
                        current_chunk = clause
                    else:
                        current_chunk += (" " if current_chunk else "") + clause
        else:
            # Try to add sentence to current chunk
            if len(current_chunk) + len(sentence) + 1 > max_chunk_chars:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add sentence to current chunk
                current_chunk += (" " if current_chunk else "") + sentence

    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def estimate_tokens_for_text(text: str) -> int:
    """
    Rough estimate of how many tokens the text will generate.

    Maya1 typically uses:
    - ~1 text token per word
    - ~7 SNAC tokens per frame
    - ~0.021 seconds per frame
    - Roughly 350 SNAC tokens per second of audio

    Args:
        text: Input text

    Returns:
        Estimated number of tokens
    """
    # Rough heuristic: 1 word = 3-4 SNAC frames = ~25 tokens
    word_count = len(text.split())
    estimated_tokens = word_count * 25

    return estimated_tokens


def should_chunk_text(text: str, max_tokens: int) -> bool:
    """
    Determine if text should be chunked based on estimated token count.

    Args:
        text: Input text
        max_tokens: Maximum tokens allowed per generation

    Returns:
        True if text should be chunked
    """
    estimated = estimate_tokens_for_text(text)

    # Use 80% of max_tokens as threshold to be safe
    threshold = int(max_tokens * 0.8)

    return estimated > threshold
