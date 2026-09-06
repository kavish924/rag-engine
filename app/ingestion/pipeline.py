

import hashlib
from typing import Literal

from app.ingestion.chunkers.fixed_size import chunk_fixed_size
from app.ingestion.chunkers.recursive_structure import chunk_recursive_structure
from app.ingestion.chunkers.semantic import chunk_semantic
from app.ingestion.dedup import filter_duplicates
from app.ingestion.embeddings import embed_texts
from app.ingestion.loaders import load_document
from app.storage.bm25_store import BM25Store
from app.storage.vector_store import VectorStore

ChunkingStrategy = Literal[
    "fixed_size",
    "recursive_structure",
    "semantic",
]

_CHUNKERS = {
    "fixed_size": chunk_fixed_size,
    "recursive_structure": chunk_recursive_structure,
    "semantic": chunk_semantic,
}




def _chunk_id(
    source_file: str,
    strategy: str,
    index: int,
    text: str,
) -> str:

    filename = source_file.replace("\\", "/").split("/")[-1]

    digest = hashlib.sha1(
        f"{source_file}|{strategy}|{index}|{text}".encode("utf-8")
    ).hexdigest()[:16]

    return f"{filename}-{strategy}-{index}-{digest}"



def run_ingestion(
    file_paths: list[str],
    strategy: ChunkingStrategy = "recursive_structure",
    vector_store: VectorStore | None = None,
    bm25_store: BM25Store | None = None,
    dedup_threshold: float = 0.95,
) -> dict:

    vector_store = vector_store or VectorStore()
    bm25_store = bm25_store or BM25Store()

    chunk_fn = _CHUNKERS[strategy]

    all_chunk_texts: list[str] = []
    all_chunk_metadatas: list[dict] = []

    files_processed: list[str] = []



    for file_path in file_paths:

        print(f"Loading: {file_path}")

        blocks = load_document(file_path)

        files_processed.append(file_path)

        for block in blocks:

            chunks = chunk_fn(block.text)

            for idx, chunk in enumerate(chunks):

                all_chunk_texts.append(chunk)

                all_chunk_metadatas.append(
                    {
                        "source_document": block.source_file,
                        "chunk_index": idx,
                        "section_heading": block.section_heading,
                        "page_number": block.page_number,
                        "chunking_strategy": strategy,
                        "character_count": len(chunk),
                    }
                )

    if not all_chunk_texts:
        return {
            "chunks_created": 0,
            "duplicates_skipped": 0,
            "files_processed": files_processed,
        }

    print(f"\nGenerated {len(all_chunk_texts)} chunks")

    

    print("Generating embeddings...")

    new_embeddings = embed_texts(all_chunk_texts)

    

    existing_embeddings = vector_store.get_all_embeddings()

    keep_indices, duplicate_indices = filter_duplicates(
        new_embeddings,
        existing_embeddings,
        threshold=dedup_threshold,
    )

    print(f"Duplicate chunks skipped: {len(duplicate_indices)}")

   

    chunks_to_store = []

    seen_ids = set()

    for i in keep_indices:

        chunk_id = _chunk_id(
            all_chunk_metadatas[i]["source_document"],
            strategy,
            all_chunk_metadatas[i]["chunk_index"],
            all_chunk_texts[i],
        )


        if chunk_id in seen_ids:
            print(f"Duplicate ID skipped: {chunk_id}")
            continue

        seen_ids.add(chunk_id)

        chunks_to_store.append(
            {
                "id": chunk_id,
                "text": all_chunk_texts[i],
                "embedding": new_embeddings[i],
                "metadata": all_chunk_metadatas[i],
            }
        )

    

    ids = [c["id"] for c in chunks_to_store]

    print("\n========== DEBUG ==========")
    print("Chunks to store :", len(ids))
    print("Unique IDs      :", len(set(ids)))

    duplicates = [x for x in ids if ids.count(x) > 1]

    if duplicates:
        print("Duplicate IDs:")
        for d in sorted(set(duplicates)):
            print("   ", d)
    else:
        print("No duplicate IDs detected.")

    print("===========================\n")

   

    vector_store.upsert_chunks(chunks_to_store)

    bm25_store.add_chunks(
        [
            {
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"],
            }
            for c in chunks_to_store
        ]
    )

    print(f"Stored {len(chunks_to_store)} chunks.")

    return {
        "chunks_created": len(chunks_to_store),
        "duplicates_skipped": len(duplicate_indices),
        "files_processed": files_processed,
    }