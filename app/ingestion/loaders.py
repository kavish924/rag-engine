

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass
class LoadedDocument:
    source_file: str
    text: str
    section_heading: str | None = None
    page_number: int | None = None

def _clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_document(file_path: str) -> list[LoadedDocument]:
    """
    Dispatch to the correct loader based on file extension.
    """

    suffix = Path(file_path).suffix.lower()

    if suffix in [".md", ".markdown"]:
        return load_markdown(file_path)

    if suffix == ".txt":
        return load_text(file_path)

    if suffix in [".html", ".htm"]:
        return load_html(file_path)

    if suffix == ".pdf":
        return load_pdf(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")

def load_markdown(file_path: str) -> list[LoadedDocument]:

    raw = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    header_pattern = re.compile(
        r"^(#{1,6})\s+(.*)$",
        re.MULTILINE,
    )

    matches = list(header_pattern.finditer(raw))

    if not matches:
        cleaned = _clean_whitespace(raw)

        if cleaned:
            return [
                LoadedDocument(
                    source_file=file_path,
                    text=cleaned,
                )
            ]
        return []

    blocks = []

    preamble = raw[:matches[0].start()].strip()

    if preamble:
        blocks.append(
            LoadedDocument(
                source_file=file_path,
                text=_clean_whitespace(preamble),
            )
        )

    for i, match in enumerate(matches):

        heading = match.group(2).strip()

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(raw)
        )

        body = raw[start:end].strip()

        text = _clean_whitespace(
            f"{heading}\n{body}" if body else heading
        )

        if text:
            blocks.append(
                LoadedDocument(
                    source_file=file_path,
                    text=text,
                    section_heading=heading,
                )
            )

    return blocks
def load_text(file_path: str) -> list[LoadedDocument]:

    raw = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    cleaned = _clean_whitespace(raw)

    if not cleaned:
        return []

    return [
        LoadedDocument(
            source_file=file_path,
            text=cleaned,
        )
    ]
def load_html(file_path: str) -> list[LoadedDocument]:

    raw_html = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    headers = soup.find_all(["h1", "h2", "h3"])

    if not headers:

        text = _clean_whitespace(
            soup.get_text(separator="\n")
        )

        if text:
            return [
                LoadedDocument(
                    source_file=file_path,
                    text=text,
                )
            ]

        return []

    blocks = []

    for header in headers:

        heading = header.get_text(strip=True)

        body_parts = []

        for sibling in header.find_next_siblings():

            if sibling.name in ["h1", "h2", "h3"]:
                break

            body_parts.append(
                sibling.get_text(
                    separator="\n",
                    strip=True,
                )
            )

        body = "\n".join(body_parts)

        text = _clean_whitespace(
            f"{heading}\n{body}" if body else heading
        )

        if text:
            blocks.append(
                LoadedDocument(
                    source_file=file_path,
                    text=text,
                    section_heading=heading,
                )
            )

    return blocks
def load_pdf(file_path: str) -> list[LoadedDocument]:

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is not installed. Run: pip install pypdf"
        )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(file_path)

    reader = PdfReader(path)

    blocks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = _clean_whitespace(
            page.extract_text() or ""
        )

        if text:
            blocks.append(
                LoadedDocument(
                    source_file=str(path),
                    text=text,
                    page_number=page_number,
                )
            )

    return blocks