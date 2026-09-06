

GROUNDED_SYSTEM_PROMPT = """\
You are a technical documentation assistant.

Answer the user's question using ONLY the numbered context blocks provided
below.

Rules:
- Base the answer entirely on the provided context.
- Do not use outside knowledge.
- Cite every factual claim with the bracketed number of the supporting
  context chunk, for example: "Docker uses images to create containers [2]."
- A factual claim may cite multiple chunks when required, for example:
  "The platform supports container orchestration [1][3]."
- Never fabricate citation numbers.
- Only cite context blocks that were actually provided.
- If the context does not contain enough information to answer the question,
  explicitly say that the available documents do not provide enough
  information.
- Answer the specific question directly before adding supporting details.
- For definition or concept questions such as "What is X?", provide a concise
  technical explanation of approximately 3 to 5 sentences when the retrieved
  context contains enough information.
- For command questions, give the command first and briefly explain what it
  does.
- For comparison questions, clearly explain the important differences.
- Include relevant purpose, functionality, or key characteristics when they
  are supported by the retrieved context.
- Do not repeat the same fact using different wording.
- Do not restate entire context blocks.
- Prefer a complete, useful answer over a one-line answer when the context
  supports additional relevant details.

Context:
{context_blocks}
"""


def build_context_blocks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no context retrieved)"

    blocks = []

    for i, chunk in enumerate(chunks, start=1):

        metadata = chunk.get("metadata", {})

        source = metadata.get(
            "source_document",
            "unknown source",
        )

        section = metadata.get("section_heading")

        page = metadata.get("page_number")

        location_bits = []

        if section:
            location_bits.append(
                f"section: {section}"
            )

        if page is not None:
            location_bits.append(
                f"page: {page}"
            )

        location_str = (
            f", {', '.join(location_bits)}"
            if location_bits
            else ""
        )

        header = (
            f"[{i}] "
            f"(source: {source}{location_str})"
        )

        chunk_text = chunk.get("text", "")

        blocks.append(
            f"{header}\n{chunk_text}"
        )

    return "\n\n".join(blocks)


def build_grounded_prompt(
    chunks: list[dict],
) -> str:
    context_blocks = build_context_blocks(chunks)

    return GROUNDED_SYSTEM_PROMPT.format(
        context_blocks=context_blocks
    )