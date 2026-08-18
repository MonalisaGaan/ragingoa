from typing import Dict, List


def extract_passage_text(passages) -> str:
    """Safely convert MSMARCO-XI passages into plain text."""

    if passages is None:
        return ""

    if isinstance(passages, str):
        return passages.strip()

    if isinstance(passages, list):
        parts = []

        for item in passages:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                for key in ("text", "passage", "content"):
                    if key in item and item[key]:
                        parts.append(str(item[key]))
                        break

            else:
                parts.append(str(item))

        return "\n".join(parts).strip()

    if isinstance(passages, dict):
        for key in ("text", "passage", "content"):
            if key in passages and passages[key]:
                return str(passages[key]).strip()

    return str(passages).strip()


def load_msmarco_sample(limit: int = 300) -> List[Dict]:
    """
    Stream a manageable MSMARCO-XI sample.

    We deliberately avoid downloading the complete dataset during
    development. Increase the limit later for benchmarking.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        "ai4bharat/MSMARCO-XI",
        split="train",
        streaming=True
    )

    documents = []

    for row in dataset:

        passage_text = extract_passage_text(
            row.get("passages")
        )

        if not passage_text:
            continue

        document = {
            "id": str(
                row.get(
                    "query_id",
                    f"doc-{len(documents)}"
                )
            ),

            "text": passage_text,

            "query": str(
                row.get("query", "")
            ),

            "answer": str(
                row.get("Answer", "")
            ),

            "english_query": str(
                row.get("Eng_Query", "")
            ),

            "english_answer": str(
                row.get("Eng_Answer", "")
            ),

            "query_type": str(
                row.get("query_type", "")
            ),

            "source_lang": str(
                row.get("source_lang", "")
            ),

            "target_lang": str(
                row.get("target_lang", "")
            )
        }

        documents.append(document)

        if len(documents) >= limit:
            break

    return documents