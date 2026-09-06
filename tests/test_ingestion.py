
from app.ingestion.chunkers.fixed_size import chunk_fixed_size
from app.ingestion.chunkers.recursive_structure import chunk_recursive_structure
from app.ingestion.chunkers.semantic import chunk_semantic
from app.ingestion.dedup import filter_duplicates, is_near_duplicate
from app.ingestion.loaders import load_markdown


def test_load_markdown_splits_on_headers(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Title\n\nIntro text.\n\n## Section A\nBody A.\n\n## Section B\nBody B.")
    blocks = load_markdown(str(md_file))
    headings = [b.section_heading for b in blocks]
    assert "Title" in headings
    assert "Section A" in headings
    assert "Section B" in headings


def test_chunk_fixed_size_respects_overlap():
    text = "a" * 250
    chunks = chunk_fixed_size(text, chunk_size=100, overlap=20)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_recursive_structure_keeps_paragraphs_together():
    text = "Short paragraph one.\n\nShort paragraph two."
    chunks = chunk_recursive_structure(text, max_chunk_size=200, overlap=0)
    assert len(chunks) == 1  


def test_chunk_semantic_splits_on_topic_shift():
    def fake_embed(sentences):
        return [[1.0, 0.0] if "cat" in s.lower() else [0.0, 1.0] for s in sentences]

    text = "Cats are great. Cats sleep a lot. Rockets are fast. Rockets fly high."
    chunks = chunk_semantic(text, similarity_threshold=0.9, embed_fn=fake_embed)
    assert len(chunks) == 2


def test_dedup_flags_near_duplicates():
    original = [1.0, 0.0, 0.0]
    duplicate = [0.999, 0.001, 0.0]
    distinct = [0.0, 1.0, 0.0]
    assert is_near_duplicate(duplicate, [original], threshold=0.95) is True
    assert is_near_duplicate(distinct, [original], threshold=0.95) is False


def test_filter_duplicates_batch():
    keep, dup = filter_duplicates([[1, 0, 0], [0.999, 0.001, 0], [0, 1, 0]], [], threshold=0.95)
    assert keep == [0, 2]
    assert dup == [1]
