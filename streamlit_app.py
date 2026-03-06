import json
import tempfile
from pathlib import Path

import streamlit as st

from src.agents.extractor import BudgetExceededError, HumanReviewRequiredError
from src.pipeline import RefineryPipeline
from src.utils.hashing import stable_hash


st.set_page_config(page_title="Document Refinery", layout="wide")


@st.cache_resource
def get_pipeline() -> RefineryPipeline:
    return RefineryPipeline(config_path="rubric/extraction_rules.yaml", output_dir=".refinery")


def _init_state() -> None:
    st.session_state.setdefault("doc_runs", {})
    st.session_state.setdefault("current_upload_doc_ids", [])


def main() -> None:
    _init_state()
    st.title("Document Intelligence Refinery")
    st.markdown(
        "Upload one or more documents to run triage -> extraction -> chunking -> PageIndex -> query, then chat per document."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Awaiting document upload.")
        return

    initial_question = st.text_input("Initial question to run after processing", value="Summarize key points")
    process_clicked = st.button("Process Uploaded Files", type="primary", use_container_width=True)

    pipeline = get_pipeline()

    if process_clicked:
        st.session_state["doc_runs"] = {}
        st.session_state["current_upload_doc_ids"] = []

        for uploaded in uploaded_files:
            safe_name = Path(uploaded.name).name
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / safe_name
                tmp_path.write_bytes(uploaded.read())
                doc_id = stable_hash(str(tmp_path.resolve()))[:16]

                status = "ok"
                processed = None
                answer = None
                with st.spinner(f"Processing {uploaded.name}..."):
                    try:
                        processed = pipeline.process_document(str(tmp_path))
                        answer = pipeline.answer_question(initial_question.strip() or "Summarize key points", processed["chunks"], processed["page_index"])
                    except HumanReviewRequiredError as exc:
                        status = f"human_review_required: {exc}"
                    except BudgetExceededError as exc:
                        status = f"budget_exceeded: {exc}"
                    except Exception as exc:  # noqa: BLE001
                        status = f"error: {exc}"

            history = []
            if answer is not None:
                history.append({"question": initial_question.strip() or "Summarize key points", "answer": answer})

            st.session_state["doc_runs"][doc_id] = {
                "name": uploaded.name,
                "status": status,
                "processed": processed,
                "history": history,
            }
            st.session_state["current_upload_doc_ids"].append(doc_id)

        st.success("Processing complete for uploaded documents.")

    doc_runs: dict = st.session_state.get("doc_runs", {})
    current_ids: list[str] = st.session_state.get("current_upload_doc_ids", [])
    if not doc_runs:
        st.info("Click 'Process Uploaded Files' to run the pipeline and start chatting.")
        return

    st.subheader("Answers")
    for doc_id in current_ids:
        run = doc_runs.get(doc_id)
        if not run:
            continue
        st.markdown(f"**{run['name']}**")
        status = run["status"]
        if status != "ok":
            if status.startswith("human_review_required"):
                st.warning(f"Requires human review: {status}")
            elif status.startswith("budget_exceeded"):
                st.error(f"Budget exceeded: {status}")
            else:
                st.error(f"Processing failed: {status}")
            continue
        if run["history"]:
            latest = run["history"][-1]["answer"]
            st.write(latest.answer)
            st.markdown("**Provenance**")
            if latest.provenance.citations:
                for i, cit in enumerate(latest.provenance.citations, start=1):
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

    st.subheader("Chat With Document")
    selectable = [(doc_id, doc_runs[doc_id]["name"]) for doc_id in current_ids if doc_id in doc_runs and doc_runs[doc_id]["status"] == "ok"]
    if selectable:
        label_to_id = {name: doc_id for doc_id, name in selectable}
        selected_name = st.selectbox("Select processed document", options=[name for _, name in selectable])
        selected_id = label_to_id[selected_name]
        selected_run = doc_runs[selected_id]

        for turn in selected_run["history"]:
            st.markdown(f"**Q:** {turn['question']}")
            st.markdown(f"**A:** {turn['answer'].answer}")

        follow_up = st.text_input("Ask a follow-up question", key=f"follow_{selected_id}")
        if st.button("Send Follow-up", key=f"send_{selected_id}"):
            if follow_up.strip():
                processed = selected_run["processed"]
                ans = pipeline.answer_question(follow_up.strip(), processed["chunks"], processed["page_index"])
                selected_run["history"].append({"question": follow_up.strip(), "answer": ans})
                st.rerun()
    else:
        st.info("No successful documents available for chat.")

    output_dir = Path(".refinery")
    st.subheader("Artifacts")
    profiles_dir = output_dir / "profiles"
    pageindex_dir = output_dir / "pageindex"
    ledger_path = output_dir / "extraction_ledger.jsonl"

    if profiles_dir.exists():
        st.markdown("**Profiles (.refinery/profiles)**")
        for p in sorted(profiles_dir.glob("*.json")):
            if p.stem not in set(current_ids):
                continue
            with st.expander(p.name):
                st.code(p.read_text(encoding="utf-8"), language="json")

    if pageindex_dir.exists():
        st.markdown("**PageIndex (.refinery/pageindex)**")
        for p in sorted(pageindex_dir.glob("*.json")):
            if p.stem not in set(current_ids):
                continue
            with st.expander(p.name):
                st.code(p.read_text(encoding="utf-8"), language="json")

    if ledger_path.exists():
        st.markdown("**Extraction Ledger (filtered to current upload)**")
        filtered_entries: list[dict] = []
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("doc_id") in set(current_ids):
                filtered_entries.append(row)
        if filtered_entries:
            for i, entry in enumerate(filtered_entries, start=1):
                title = f"Entry {i}: {entry.get('document_name', 'unknown')} ({entry.get('strategy_used', 'n/a')})"
                with st.expander(title):
                    st.json(entry)
        else:
            st.write("No ledger entries found for the current upload batch.")


if __name__ == "__main__":
    main()
