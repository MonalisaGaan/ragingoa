import re
from typing import Dict, List


def fixed_chunk(
    text: str,
    chunk_size: int = 500
) -> List[str]:
    """
    Fixed-size character chunking.
    """

    text = text.strip()

    if not text:
        return []

    return [
        text[i:i + chunk_size].strip()
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]


def overlapping_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Fixed-size chunking with overlap.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def sentence_chunk(
    text: str,
    max_chars: int = 600
) -> List[str]:
    """
    Sentence-aware chunking.

    Attempts to keep complete sentences
    together rather than cutting at arbitrary
    character positions.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current} {sentence}".strip()
        )

        if len(candidate) <= max_chars:

            current = candidate

        else:

            if current:
                chunks.append(current)

            current = sentence

    if current:
        chunks.append(current)

    return chunks


def create_chunks(
    text: str,
    metadata: Dict = None
) -> List[Dict]:
    """
    Generate chunks using all supported strategies.
    """

    metadata = metadata or {}

    strategies = {
        "fixed": fixed_chunk(text),
        "overlap": overlapping_chunk(text),
        "sentence": sentence_chunk(text)
    }

    results = []

    for strategy, chunks in strategies.items():

        for index, chunk in enumerate(chunks):

            results.append({
                "chunk_id": (
                    f"{metadata.get('id', 'doc')}"
                    f"-{strategy}-{index}"
                ),
                "text": chunk,
                "strategy": strategy,
                "metadata": metadata
            })

    return results