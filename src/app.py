import os
import uuid
from typing import Any

import gradio as gr

from src.rag_pipeline import ingest, ask
from src.session import Session


# ============================================================
# SESSION HELPERS
# ============================================================

def make_session() -> dict[str, Any]:
    return {
        "session": Session(session_id=str(uuid.uuid4())),
    }


def initialize_session():
    return make_session()


def document_choices(session: Session) -> list[tuple[str, str]]:
    return [
        (info.get("filename", document_id), document_id)
        for document_id, info in session.get_documents().items()
    ]


def get_memory(session: Session, document_id: str):
    """
    Supports the current Session implementation and also keeps
    this UI compatible with an older Session implementation that
    may not expose get_memory().
    """
    if hasattr(session, "get_memory"):
        return session.get_memory(document_id)

    from memory import ConversationMemory

    memories = getattr(session, "_ui_memories", None)

    if memories is None:
        memories = {}
        setattr(session, "_ui_memories", memories)

    if document_id not in memories:
        memories[document_id] = ConversationMemory()

    return memories[document_id]


def clear_memory(session: Session, document_id: str) -> None:
    if hasattr(session, "clear_memory"):
        session.clear_memory(document_id)
        return

    memory = get_memory(session, document_id)
    memory.clear()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

def upload_document(
    file: Any,
    state: dict[str, Any] | None,
):
    if state is None:
        state = make_session()

    if file is None:
        return (
            state,
            gr.update(choices=document_choices(state["session"]), value=None),
            "Ready to upload a document.",
            gr.update(visible=False),
        )

    file_path = file if isinstance(file, str) else getattr(file, "name", None)

    if not file_path:
        return (
            state,
            gr.update(choices=document_choices(state["session"]), value=None),
            "Unable to read the selected file.",
            gr.update(visible=False),
        )

    filename = os.path.basename(file_path)
    session: Session = state["session"]

    try:
        result = ingest(
            file_path=file_path,
            session_id=session.session_id,
            session=session,
        )
    except Exception as error:
        return (
            state,
            gr.update(choices=document_choices(session), value=None),
            f"Upload failed: {error}",
            gr.update(visible=False),
        )

    document_id = result["document_id"]
    state["selected_document_id"] = document_id

    return (
        state,
        gr.update(
            choices=document_choices(session),
            value=document_id,
        ),
        f"{filename} uploaded successfully.",
        gr.update(visible=False),
    )


# ============================================================
# WEB URL
# ============================================================

def ingest_url(
    url: str,
    state: dict[str, Any] | None,
):
    if state is None:
        state = make_session()

    url = (url or "").strip()

    if not url:
        return "Enter a valid document URL."

    # The current backend exposes ingest(file_path=...), not URL
    # ingestion. Keep this explicit rather than pretending the URL
    # was indexed.
    return (
        "URL ingestion is not enabled by the current RAG backend. "
        "Use the file upload option."
    )


# ============================================================
# DOCUMENT SELECTION
# ============================================================

def select_document(
    document_id: str | None,
    state: dict[str, Any] | None,
):
    if state is None:
        return state

    session: Session = state["session"]

    if document_id and session.has_document(document_id):
        state["selected_document_id"] = document_id
    else:
        state["selected_document_id"] = None

    return state


# ============================================================
# ASK
# ============================================================

def answer_question(
    query: str,
    state: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
):
    history = list(history or [])

    if state is None:
        state = make_session()

    query = (query or "").strip()

    if not query:
        return (
            gr.update(value=history, visible=bool(history)),
            "",
            state,
        )

    session: Session = state["session"]
    document_id = state.get("selected_document_id")

    if not document_id or not session.has_document(document_id):
        history.extend(
            [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": "Upload and select a document before asking a question.",
                },
            ]
        )

        return (
            gr.update(value=history, visible=True),
            "",
            state,
        )

    memory = get_memory(session, document_id)

    try:
        result = ask(
            query=query,
            document_id=document_id,
            session_id=session.session_id,
            memory=memory,
            session=session,
        )
        answer = result["answer"]
    except Exception as error:
        answer = f"Unable to answer the question.\n\n{error}"

    history.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
    )

    return (
        gr.update(value=history, visible=True),
        "",
        state,
    )


# ============================================================
# CLEAR
# ============================================================

def clear_conversation(
    state: dict[str, Any] | None,
):
    if state is None:
        return [], gr.update(visible=False)

    session: Session = state["session"]
    document_id = state.get("selected_document_id")

    if document_id:
        clear_memory(session, document_id)

    return [], gr.update(visible=False)


# ============================================================
# NEW SESSION
# ============================================================

def new_session():
    state = make_session()

    return (
        state,
        gr.update(choices=[], value=None),
        [],
        "",
        "Ready to upload a document.",
        gr.update(visible=False),
        "File",
        gr.update(visible=False),
    )


# ============================================================
# CSS
# ============================================================

CSS = r"""
:root {
    --bg: #0b0f19;
    --panel: #111827;
    --panel-2: #172033;
    --border: rgba(148, 163, 184, 0.16);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #6366f1;
    --accent-2: #3b82f6;
}

* {
    box-sizing: border-box !important;
}

html,
body {
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: var(--bg) !important;
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", Roboto, sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.gradio-container > div,
.gradio-container .main {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

footer {
    display: none !important;
}

.app-shell {
    width: 100% !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    margin: 0 !important;
    background: var(--bg) !important;
}

.main-page {
    width: min(1180px, calc(100vw - 72px)) !important;
    height: calc(100vh - 44px) !important;
    max-height: calc(100vh - 44px) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
    padding: 18px 0 92px !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* Header */
.brand {
    flex: 0 0 auto !important;
    margin: 0 !important;
    padding: 0 8px !important;
    color: #ffffff !important;
    font-size: 30px !important;
    line-height: 1.15 !important;
    font-weight: 800 !important;
    letter-spacing: -0.7px !important;
}

.brand-subtitle {
    margin-top: 5px !important;
    color: var(--muted) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}

/* Upload card */
.upload-card {
    flex: 0 0 auto !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

.upload-card > .label-wrap,
.upload-card summary {
    min-height: 48px !important;
    padding: 0 18px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    color: var(--text) !important;
    background: rgba(255,255,255,0.018) !important;
    border-bottom: 1px solid rgba(148,163,184,0.10) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.upload-card .wrap {
    padding: 0 !important;
}

.upload-body {
    padding: 12px 18px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 9px !important;
}

/* The old Collection Name and upload-type controls are removed
   from the visible interface. */

.upload-type-title,
.upload-radio,
.url-picker {
    display: none !important;
}

/* Compact file picker */
.file-picker {
    width: 100% !important;
    height: 58px !important;
    min-height: 58px !important;
    max-height: 58px !important;
    margin: 0 !important;
    overflow: hidden !important;
}

.file-picker > label {
    display: none !important;
}

.file-picker .wrap,
.file-picker .upload-container,
.file-picker .drop-zone,
.file-picker [data-testid="file-upload"],
.file-picker .file-preview,
.file-picker .single-file {
    min-height: 58px !important;
    height: 58px !important;
    max-height: 58px !important;
    width: 100% !important;
    padding: 7px 12px !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    background: var(--panel-2) !important;
    border: 1px dashed rgba(99,102,241,0.45) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
}

.file-picker .drop-zone:hover,
.file-picker [data-testid="file-upload"]:hover,
.file-picker .upload-container:hover {
    border-color: var(--accent) !important;
    background: #1b263b !important;
}

.file-picker svg {
    width: 21px !important;
    height: 21px !important;
}

.file-picker p,
.file-picker span,
.file-picker div {
    font-size: 12.5px !important;
}

.file-picker button {
    min-height: 32px !important;
    height: 32px !important;
    padding: 4px 14px !important;
    border: 0 !important;
    border-radius: 7px !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Status */
.status {
    min-height: 30px !important;
    margin: 0 !important;
    padding: 6px 11px !important;
    background: rgba(16,185,129,0.07) !important;
    border: 1px solid rgba(16,185,129,0.16) !important;
    border-radius: 8px !important;
    color: #34d399 !important;
    font-size: 12px !important;
    line-height: 1.35 !important;
}

.status p {
    margin: 0 !important;
}

/* Document selector remains available internally but is not shown. */
.document-selector {
    display: none !important;
}

/* Chat */
.chat-window {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 12px !important;
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: auto !important;
}

.chat-window .message {
    font-size: 14px !important;
    line-height: 1.55 !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    margin-bottom: 9px !important;
}

.chat-window [data-testid="user"],
.chat-window .user {
    max-width: 72% !important;
    margin-left: auto !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #ffffff !important;
    border-radius: 14px 14px 4px 14px !important;
}

.chat-window [data-testid="bot"],
.chat-window .bot {
    max-width: 78% !important;
    margin-right: auto !important;
    background: var(--panel-2) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 14px 14px 14px 4px !important;
}

/* Fixed composer */
.ask-dock {
    position: fixed !important;
    z-index: 1000 !important;
    left: 50% !important;
    bottom: 16px !important;
    transform: translateX(-50%) !important;
    width: min(1180px, calc(100vw - 72px)) !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}

.ask-row {
    position: relative !important;
    width: 100% !important;
    height: 54px !important;
    min-height: 54px !important;
    padding: 5px 10px 5px 17px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: rgba(17,24,39,0.96) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 28px !important;
    box-shadow: 0 12px 34px rgba(0,0,0,0.42) !important;
    backdrop-filter: blur(14px) !important;
}

.question-box {
    flex: 1 1 auto !important;
    width: 100% !important;
    margin: 0 !important;
}

.question-box textarea {
    width: 100% !important;
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    padding: 0 48px 0 0 !important;
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    color: #f8fafc !important;
    font-size: 14px !important;
    line-height: 42px !important;
    resize: none !important;
}

.question-box textarea::placeholder {
    color: #64748b !important;
}

.ask-button {
    position: absolute !important;
    right: 9px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #ffffff !important;
    font-size: 0 !important;
}

.ask-button::after {
    content: "→" !important;
    display: block !important;
    font-size: 19px !important;
    line-height: 36px !important;
    text-align: center !important;
    font-weight: 600 !important;
}

.ask-button:hover {
    transform: translateY(-50%) scale(1.05) !important;
}

/* Smaller screens */
@media (max-width: 900px) {
    .main-page {
        width: calc(100vw - 28px) !important;
        height: calc(100vh - 30px) !important;
        padding-top: 12px !important;
    }

    .ask-dock {
        width: calc(100vw - 28px) !important;
        bottom: 10px !important;
    }

    .brand {
        font-size: 25px !important;
    }
}
"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="RAG Chat Assistant") as demo:

    state = gr.State()

    with gr.Column(elem_classes="app-shell"):

        with gr.Column(elem_classes="main-page"):

            gr.HTML(
                """
                <div class="brand">
                    RAG Chat Assistant
                    <div class="brand-subtitle">
                        Ask questions grounded in your documents
                    </div>
                </div>
                """
            )

            # Clean first-load design:
            # Collection Name and Choose Upload Type are removed.
            with gr.Accordion(
                "Upload Documents",
                open=True,
                elem_classes="upload-card",
            ):
                with gr.Column(elem_classes="upload-body"):

                    file_upload = gr.File(
                        label="Upload document",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="single",
                        type="filepath",
                        elem_classes="file-picker",
                    )

                    upload_status = gr.Markdown(
                        "Ready to upload a document.",
                        elem_classes="status",
                    )

            # Kept internally so the existing session/document logic
            # continues to work. It is hidden from the user.
            document_selector = gr.Dropdown(
                choices=[],
                value=None,
                show_label=False,
                visible=False,
                elem_classes="document-selector",
            )

            chatbot = gr.Chatbot(
                show_label=False,
                visible=False,
                height=300,
                elem_classes="chat-window",
            )

        # Fixed question composer
        with gr.Column(elem_classes="ask-dock"):

            with gr.Column(elem_classes="ask-row"):

                question_box = gr.Textbox(
                    placeholder="Ask me anything...",
                    show_label=False,
                    lines=1,
                    max_lines=1,
                    elem_classes="question-box",
                )

                ask_button = gr.Button(
                    "",
                    elem_classes="ask-button",
                )


# ============================================================
# EVENTS
# ============================================================

    file_upload.upload(
        fn=upload_document,
        inputs=[file_upload, state],
        outputs=[
            state,
            document_selector,
            upload_status,
            chatbot,
        ],
    )

    document_selector.change(
        fn=select_document,
        inputs=[document_selector, state],
        outputs=[state],
    )

    ask_button.click(
        fn=answer_question,
        inputs=[question_box, state, chatbot],
        outputs=[chatbot, question_box, state],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, state, chatbot],
        outputs=[chatbot, question_box, state],
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Base(),
    )