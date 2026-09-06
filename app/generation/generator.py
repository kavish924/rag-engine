

import re

from groq import Groq

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

    answer = _call_llm(system_prompt, question)

    citations = _parse_citations(answer, chunks)

    print("\n" + "=" * 60)
    print("GENERATOR OUTPUT")
    print("=" * 60)

    print("\nAnswer:\n")
    print(answer)

    print("\nParsed Citations:\n")

    for citation in citations:
        print(citation)

    print("=" * 60 + "\n")

    return {
        "answer": answer,
        "raw_citations": citations,
    }


def _call_llm(system_prompt: str, question: str) -> str:
    provider = settings.llm_provider.lower()

    if provider == "groq":
        return _call_groq(system_prompt, question)

    raise ValueError(f"Unknown provider: {provider}")


def _call_groq(system_prompt: str, question: str) -> str:
    client = Groq(
        api_key=settings.groq_api_key,
    )

    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def _parse_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    citations = []
    seen = set()

    for sentence in _split_into_sentences(answer_text):

        for match in _CITATION_MARKER_RE.finditer(sentence):

            indices = [
                int(x.strip())
                for x in match.group(1).split(",")
            ]

            for idx in indices:

                if idx < 1 or idx > len(chunks):
                    continue

                claim = chunks[idx - 1]["text"][:250]

                key = (idx, claim)

                if key in seen:
                    continue

                seen.add(key)

                citations.append(
                    {
                        "marker": f"[{idx}]",
                        "excerpt": claim,
                        "chunk_index": idx,
                        "chunk": chunks[idx - 1],
                    }
                )

    print("\n========== PARSED CITATIONS ==========")

    for c in citations:
        print(f"Marker : {c['marker']}")
        print(f"Excerpt: {c['excerpt']}")
        print("-" * 40)

    print("=====================================\n")

    return citations


def _split_into_sentences(text: str):
    return [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if s.strip()
    ]