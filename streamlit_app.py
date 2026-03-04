import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import RefineryPipeline


st.set_page_config(page_title="Document Refinery", layout="wide")


@st.cache_resource
def get_pipeline() -> RefineryPipeline:
    return RefineryPipeline(config_path="rubric/extraction_rules.yaml", output_dir=".refinery")


def main() -> None:
    st.title("Document Intelligence Refinery")
    st.markdown(
        "Upload one or more documents to run them through the triage → extraction → chunking → PageIndex → query pipeline."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Awaiting document upload.")
        return

    pipeline = get_pipeline()
    results: list[tuple[str, object]] = []

    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner(f"Running refinery pipeline for {uploaded.name}..."):
            res = pipeline.run(tmp_path)
            results.append((uploaded.name, res))

    st.success("Pipeline complete for all uploaded documents.")

    # High-level answers and provenance per document
    st.subheader("Answers (auto-summarize prompt)")
    for name, result in results:
        st.markdown(f"**{name}**")
        st.write(result.answer)

        st.markdown("**Provenance**")
        if result.provenance.citations:
            for i, cit in enumerate(result.provenance.citations, start=1):
                with st.expander(f"Citation {i}: page {cit.page_number}"):
                    st.json(
                        {
                            "document_name": cit.document_name,
                            "page_number": cit.page_number,
                            "bbox": cit.bbox.model_dump(),
                            "content_hash": cit.content_hash,
                        }
                    )
        else:
            st.write("No citations found.")

    # Artifacts
    output_dir = Path(".refinery")
    st.subheader("Artifacts")
    profiles_dir = output_dir / "profiles"
    pageindex_dir = output_dir / "pageindex"

    if profiles_dir.exists():
        st.markdown("**Profiles (.refinery/profiles)**")
        for p in sorted(profiles_dir.glob("*.json")):
            with st.expander(p.name):
                st.code(p.read_text(encoding="utf-8"), language="json")

    if pageindex_dir.exists():
        st.markdown("**PageIndex (.refinery/pageindex)**")
        for p in sorted(pageindex_dir.glob("*.json")):
            with st.expander(p.name):
                st.code(p.read_text(encoding="utf-8"), language="json")


if __name__ == "__main__":
    main()

