
import re

from groq import Groq

from app.config import settings


# Create the client once
client = Groq(api_key=settings.groq_api_key)


def score_correctness(
    generated_answer: str,
    golden_answer: str,
    question: str,
) -> float:

    if golden_answer.strip().upper() == "NOT_IN_CORPUS":
        return _score_refusal_correctness(generated_answer)

    if golden_answer.strip().upper().startswith("AMBIGUOUS"):
        return _score_ambiguous_handling(
            generated_answer,
            golden_answer,
        )

    return _score_standard_correctness(
        generated_answer,
        golden_answer,
        question,
    )


def _score_standard_correctness(
    generated_answer: str,
    golden_answer: str,
    question: str,
) -> float:

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Question:
{question}

Golden Answer:
{golden_answer}

Generated Answer:
{generated_answer}

Rate how correct the generated answer is.

Scoring:
10 = Completely correct
8 = Mostly correct
5 = Partially correct
2 = Mostly incorrect
0 = Completely incorrect

Reply ONLY with a single number between 0 and 10.
"""

    score = _parse_single_score(_call_llm(prompt))

    return max(0.0, min(score / 10.0, 1.0))


def _score_refusal_correctness(
    generated_answer: str,
) -> float:

    prompt = f"""
Should the following answer be considered a correct refusal?

Answer:
{generated_answer}

Reply ONLY with:

yes

or

no
"""

    response = _call_llm(prompt).strip().lower()

    return 1.0 if response.startswith("yes") else 0.0


def _score_ambiguous_handling(
    generated_answer: str,
    golden_answer: str,
) -> float:

    prompt = f"""
The reference says this question is ambiguous.

Reference:
{golden_answer}

Generated answer:
{generated_answer}

Did the generated answer acknowledge ambiguity?

Reply ONLY with:

yes

or

no
"""

    response = _call_llm(prompt).strip().lower()

    return 1.0 if response.startswith("yes") else 0.0


def _call_llm(prompt: str) -> str:

    provider = settings.llm_provider.lower()

    if provider != "groq":
        raise ValueError(f"Unsupported provider: {provider}")

    return _call_groq(prompt)


def _call_groq(prompt: str) -> str:

    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=20,
    )

    return response.choices[0].message.content.strip()


def _parse_single_score(raw_response: str) -> float:

    match = re.search(r"\d+(\.\d+)?", raw_response)

    if not match:
        return 0.0

    return float(match.group())