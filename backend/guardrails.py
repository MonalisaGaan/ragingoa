from typing import Dict, List, Tuple


BLOCKED_PATTERNS = [
    "how to make a bomb",
    "how to build a bomb",
    "how to build a weapon",
    "how to hurt someone",
]


def validate_query(
    query: str
) -> Tuple[bool, str]:

    query = query.strip()

    if not query:
        return False, "Please provide a question."

    if len(query) > 1000:
        return False, "Question is too long."

    lowered = query.lower()

    for pattern in BLOCKED_PATTERNS:

        if pattern in lowered:

            return (
                False,
                "I can't help with that request."
            )

    return True, ""


def check_relevance(
    results: List[Dict],
    threshold: float = 0.25
) -> bool:

    if not results:
        return False

    return (
        results[0]["score"] >= threshold
    )


def build_context(
    results: List[Dict],
    max_chars: int = 5000
) -> str:

    context = []
    total = 0

    for index, result in enumerate(results):

        text = result["text"]

        if total + len(text) > max_chars:
            break

        context.append(
            f"[Evidence {index + 1}]\n{text}"
        )

        total += len(text)

    return "\n\n".join(context)


def grounding_check(
    answer: str,
    context: str
) -> bool:
    """
    Conservative lexical grounding check.

    A production version can replace this with
    a dedicated NLI/LLM verification model.
    """

    if not answer.strip():
        return False

    if not context.strip():
        return False

    answer_words = {
        word.lower().strip(".,!?")
        for word in answer.split()
        if len(word) > 4
    }

    context_words = {
        word.lower().strip(".,!?")
        for word in context.split()
        if len(word) > 4
    }

    overlap = (
        answer_words.intersection(
            context_words
        )
    )

    return len(overlap) >= 2