import json
import re

from groq import Groq

from app.config import settings


def score_faithfulness(
    answer: str,
    retrieved_chunks: list[dict],
) -> float:

    claims = _extract_claims(answer)

    if not claims:
        return 1.0

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    verdicts = _judge_claims_grounded(
        claims,
        context,
    )

    if not verdicts:
        return 0.0

    return sum(verdicts) / len(verdicts)


def _extract_claims(answer: str) -> list[str]:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    cleaned = [
        re.sub(
            r"\[\d+(?:\s*,\s*\d+)*\]",
            "",
            s,
        ).strip()
        for s in sentences
    ]

    return [
        s
        for s in cleaned
        if len(s) > 5
    ]


def _judge_claims_grounded(
    claims: list[str],
    context: str,
) -> list[bool]:

    numbered_claims = "\n".join(
        f"[{i}] {claim}"
        for i, claim in enumerate(claims, start=1)
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

    raw = _call_llm(prompt)

    return _parse_boolean_array(
        raw,
        len(claims),
    )


def _call_llm(prompt: str) -> str:

    provider = settings.llm_provider.lower()

    if provider == "groq":
        return _call_groq(prompt)

    if provider == "openai":
        return _call_openai(prompt)

    raise ValueError(f"Unsupported provider: {provider}")


def _call_groq(prompt: str) -> str:

    client = Groq(
        api_key=settings.groq_api_key,
    )

    response = client.chat.completions.create(
        model=settings.generation_model,
        temperature=0,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content.strip()


def _call_openai(prompt: str) -> str:

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
    )

    response = client.chat.completions.create(
        model=settings.generation_model,
        temperature=0,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content.strip()


def _parse_boolean_array(
    raw: str,
    expected_len: int,
) -> list[bool]:

    match = re.search(
        r"\[[^\]]*\]",
        raw,
        re.IGNORECASE,
    )

    if not match:
        return [False] * expected_len

    try:
        verdicts = json.loads(
            match.group().lower()
        )
    except json.JSONDecodeError:
        return [False] * expected_len

    if len(verdicts) != expected_len:
        verdicts = (
            verdicts
            + [False] * expected_len
        )[:expected_len]

    return [bool(v) for v in verdicts]