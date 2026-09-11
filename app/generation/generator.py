import re

from openai import OpenAI

from app.config import settings
from app.generation.prompts import build_grounded_prompt


_CITATION_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def generate_answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I don't have enough information to answer this question.",
            "raw_citations": [],
        }

    system_prompt = build_grounded_prompt(chunks)
    answer = _call_gemini(system_prompt, question)
    citations = _parse_citations(answer, chunks)

    return {
        "answer": answer,
        "raw_citations": citations,
    }


def _parse_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    citations = []
    seen = set()

    for sentence in _split_into_sentences(answer_text):
        for match in _CITATION_MARKER_RE.finditer(sentence):
            indices = [int(x.strip()) for x in match.group(1).split(",")]
            for idx in indices:
                if idx < 1 or idx > len(chunks):
                    continue
                claim = chunks[idx - 1]["text"][:250]
                key = (idx, claim)
                if key in seen:
                    continue
                seen.add(key)
                citations.append({
                    "marker": f"[{idx}]",
                    "excerpt": claim,
                    "chunk_index": idx,
                    "chunk": chunks[idx - 1],
                })

    return citations


def _split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


def _call_gemini(system_prompt: str, question: str) -> str:
    client = OpenAI(
        api_key=settings.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=settings.gemini_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
    )
    return response.choices[0].message.content.strip()