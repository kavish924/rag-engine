
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Engine", layout="wide")
st.title("RAG Engine — Query Dashboard")


def get_filename(path: str) -> str:
    if not path:
        return "unknown"

    return path.replace("\\", "/").split("/")[-1]


def call_ask_api(
    question: str,
    mode: str,
) -> dict | None:

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/ask",
            json={
                "question": question,
                "retrieval_mode": mode,
            },
            timeout=(10, 300),
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectTimeout:
        st.error(
            "Connection to the RAG API timed out."
        )

    except requests.exceptions.ReadTimeout:
        st.error(
            "The RAG pipeline is taking too long to respond. "
            "The backend is still processing the request."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend. "
            "Make sure the API is running on port 8000."
        )

    except requests.exceptions.HTTPError as e:
        st.error(
            f"RAG API returned an HTTP error: {e}"
        )

    except requests.exceptions.RequestException as e:
        st.error(
            f"Request to API failed: {e}"
        )

    return None

def format_latency(value) -> str:
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if value >= 1000:
        return f"{value / 1000:.2f}s"

    return f"{value:.1f}ms"

def render_answer_column(result: dict, label: str):
    if result is None:
        st.warning("No response received.")
        return

    st.subheader(label)


    timings = result.get("timings") or {}

    if timings:
        st.markdown("## ⚡ Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Retrieval",
            format_latency(
                timings.get("retrieval_ms")
            ),
        )

        col2.metric(
            "Generation",
            format_latency(
                timings.get("generation_ms")
            ),
        )

        col3.metric(
            "Verification",
            format_latency(
                timings.get("verification_ms")
            ),
        )

        col4.metric(
            "Confidence",
            format_latency(
                timings.get("confidence_ms")
            ),
        )

        col5.metric(
            "Total",
            format_latency(
                timings.get("total_ms")
            ),
        )
    else:
        st.info(
            "Performance metrics are not available for this response."
        )

    if result.get("is_fallback"):
        st.warning(result["answer"])
    else:
        st.markdown("---")
        st.markdown("# 💬 Answer")
        st.markdown(result["answer"])

    confidence = result["confidence"]

    score = confidence["composite"]

    st.markdown("## 🎯 Confidence")

    st.progress(min(max(score, 0.0), 1.0))

    if score >= 0.80:
        st.success("🟢 High Confidence")

    elif score >= 0.60:
        st.warning("🟡 Medium Confidence")

    elif score >= 0.45:
        st.warning("🟠 Accepted — Near Confidence Threshold")

    else:
        st.error("🔴 Low Confidence — Fallback Recommended")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Composite", f"{score:.2f}")

    col2.metric(
        "Retrieval",
        f"{confidence['retrieval_confidence']:.2f}"
    )

    col3.metric(
        "Citation",
        f"{confidence['citation_coverage']:.2f}"
    )

    col4.metric(
        "Completeness",
        f"{confidence['answer_completeness']:.2f}"
    )

    citations = result.get("citations", [])

    if citations:

        st.markdown("## 📚 Citations")

        supported = sum(1 for c in citations if c["supported"])
        unsupported = len(citations) - supported

        col1, col2 = st.columns(2)

        col1.metric("✅ Verified", supported)
        col2.metric("⚠ Unsupported", unsupported)

        for c in citations:

            if c["supported"]:
                status = "✅ Verified Citation"
                color = "🟢"
            else:
                status = "⚠ Unsupported Citation"
                color = "🔴"

            source_name = get_filename(
            c.get("source_document", "")
            )
            

            with st.expander(
            f"{color} {c['marker']} — {source_name}",
            expanded=False,
            ):

                st.markdown(f"### {status}")

                st.markdown(
                f"**Source Document:** `{source_name}`"
            )

                if c.get("section_heading"):
                    st.markdown(
                        f"**Section:** `{c['section_heading']}`"
                    )

                st.markdown("**Excerpt Used**")

                st.code(
                    c["excerpt"],
                    language="text",
                )

                if c["supported"]:

                    st.success(
                    "The citation successfully supports the generated claim."
                    )

                else:

                    st.error(
                        "The generated claim could not be verified using this source chunk."
                )

    retrieved_chunks = result.get("retrieved_chunks", [])

    if retrieved_chunks:

        st.markdown("---")
        st.markdown("## 📄 Retrieved Chunks")

        num_chunks = len(retrieved_chunks)

        num_docs = len(
            {
                c["source_document"]
                for c in retrieved_chunks
            }
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Chunks Retrieved",
            num_chunks,
        )

        col2.metric(
            "Source Documents",
            num_docs,
        )

    table = pd.DataFrame(
        [
            {
                "Rank": c["rank"],
                "Score": (
                    round(c["score"], 2)
                    if c["score"] is not None
                    else "-"
                ),
                "Source": get_filename(c["source_document"]),
                "Section": c["section_heading"] or "-",
                "Preview": c["preview"],
            }
            for c in retrieved_chunks
        ]
    )

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=250,
    )

    st.markdown("### 🔍 Chunk Details")

    for chunk in retrieved_chunks:

        source_name = get_filename(
        chunk.get("source_document", "")
    )

        score_display = (
        round(chunk["score"], 2)
        if chunk["score"] is not None
        else "-"
        )

        title = (
            f"Rank {chunk['rank']} "
            f"| Score: {score_display} "
            f"| {source_name}"
        )

        with st.expander(title):

            st.markdown(
                f"**Section:** "
                f"{chunk['section_heading'] or 'Unknown'}"
            )

            st.markdown("**Chunk Preview**")

            st.code(
                chunk["preview"],
                language="text",
            )

            st.caption(
                f"Chunk ID: {chunk['chunk_id']}"
            )
question = st.text_input("Ask a question about the indexed documents")
compare = st.checkbox("Compare hybrid vs. dense-only side by side")

if not compare:
    mode = st.radio("Retrieval mode", ["hybrid", "dense_only"], horizontal=True)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and not question:
    st.warning("Enter a question first.")

elif ask_clicked and question:
    if compare:
        with st.spinner("Running both retrieval modes..."):
            hybrid_result = call_ask_api(question, "hybrid")
            dense_result = call_ask_api(question, "dense_only")

        col1, col2 = st.columns(2)
        with col1:
            render_answer_column(hybrid_result, "Hybrid (dense + sparse)")
        with col2:
            render_answer_column(dense_result, "Dense-only")
    else:
        with st.spinner("Retrieving and generating..."):
            result = call_ask_api(question, mode)
        render_answer_column(result, f"Answer ({mode})")

with st.sidebar:
    st.header("Indexed Documents")
    if st.button("Refresh"):
        st.rerun()

    try:
        docs_response = requests.get(f"{API_BASE_URL}/v1/documents", timeout=10)
        docs_response.raise_for_status()
        documents = docs_response.json().get("documents", [])

        if documents:

            documents_df = pd.DataFrame(documents)

            documents_df["source_file"] = (
        documents_df["source_file"]
        .apply(get_filename)
    )

            documents_df = documents_df[
        [
            "source_file",
            "num_chunks",
            "chunking_strategies_used",
        ]
    ]

            documents_df = documents_df.rename(
        columns={
            "source_file": "Document",
            "num_chunks": "Chunks",
            "chunking_strategies_used": "Strategy",
        }
    )

            st.dataframe(
        documents_df,
        hide_index=True,
        use_container_width=True,
    )
        else:
            st.info("No documents indexed yet. Run scripts/seed_corpus.py.")
    except requests.exceptions.RequestException:
        st.warning("Could not reach the API to list documents.")