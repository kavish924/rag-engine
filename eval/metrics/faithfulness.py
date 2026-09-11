import json
import re

from eval.llm_judge_client import call_judge_llm


def score_faithfulness(
    answer: str,
    retrieved_chunks: list[dict],
) -> float:

    claims = _extract_claims(answer)

    if not claims:
        return 1.0

    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    verdicts = _judge_claims_grounded(claims, context)

    if not verdicts:
        return 0.0

    return sum(verdicts) / len(verdicts)


def _extract_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    cleaned = [
        re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", s).strip() for s in sentences
    ]
    return [s for s in cleaned if len(s) > 5]


def _judge_claims_grounded(claims: list[str], context: str) -> list[bool]:

    numbered_claims = "\n".join(
        f"[{i}] {claim}" for i, claim in enumerate(claims, start=1)
    )

    prompt = f"""
You are evaluating the faithfulness of a Retrieval-Augmented Generation (RAG) system.

Context:
{context}

Claims:
{numbered_claims}

For each numbered claim, determine whether it is directly supported by the supplied context.

Rules:
- Use ONLY the supplied context.
- Do not use outside knowledge.
- Return ONLY a JSON array of booleans.

Example:
[true,false,true]
"""

    raw = call_judge_llm(
        "You are a strict, precise evaluator. Follow the output format exactly.",
        prompt,
    )
    return _parse_boolean_array(raw, len(claims))


def _parse_boolean_array(raw: str, expected_len: int) -> list[bool]:
    match = re.search(r"\[[^\]]*\]", raw, re.IGNORECASE)
    if not match:
        print(f"WARNING: faithfulness judge returned unparseable response: {raw!r}")
        return [False] * expected_len

    try:
        verdicts = json.loads(match.group().lower())
    except json.JSONDecodeError:
        print(f"WARNING: faithfulness judge returned malformed JSON: {raw!r}")
        return [False] * expected_len
    ...