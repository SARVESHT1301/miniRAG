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
    --bg: #080b12;
    --surface: #101521;
    --surface-2: #151c2b;
    --surface-3: #1b2435;
    --border: rgba(148, 163, 184, 0.15);
    --border-strong: rgba(99, 102, 241, 0.45);
    --text: #f8fafc;
    --muted: #94a3b8;
    --muted-2: #64748b;
    --accent: #6366f1;
    --accent-2: #3b82f6;
    --success: #34d399;
}

* {
    box-sizing: border-box !important;
}

html,
body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 100% !important;
    background: var(--bg) !important;
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", Roboto, sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    background:
        radial-gradient(circle at 50% -20%, rgba(99,102,241,0.14), transparent 42%),
        var(--bg) !important;
    color: var(--text) !important;
}

footer {
    display: none !important;
}

/* Main application */
.app-shell {
    width: 100% !important;
    min-height: 100vh !important;
    display: flex !important;
    justify-content: center !important;
    padding: 34px 22px 30px !important;
}

.main-page {
    width: min(1000px, 100%) !important;
    min-height: calc(100vh - 64px) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 18px !important;
    padding: 0 !important;
}

/* Header */
.header {
    text-align: center !important;
    padding: 8px 0 6px !important;
}

.logo {
    width: 48px !important;
    height: 48px !important;
    margin: 0 auto 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    box-shadow: 0 10px 30px rgba(59,130,246,0.22) !important;
    font-size: 23px !important;
}

.brand {
    margin: 0 !important;
    color: var(--text) !important;
    font-size: 31px !important;
    line-height: 1.15 !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px !important;
}

.brand-subtitle {
    margin-top: 8px !important;
    color: var(--muted) !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* Document card */
.document-card {
    width: 100% !important;
    margin: 0 !important;
    padding: 20px !important;
    background: rgba(16,21,33,0.92) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 50px rgba(0,0,0,0.20) !important;
}

.section-title {
    margin: 0 0 5px !important;
    color: var(--text) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.section-subtitle {
    margin: 0 0 15px !important;
    color: var(--muted) !important;
    font-size: 12.5px !important;
}

/* Upload area */
.file-picker {
    width: 100% !important;
    min-height: 118px !important;
    height: 118px !important;
    margin: 0 !important;
}

.file-picker > label {
    display: none !important;
}

.file-picker .wrap,
.file-picker .upload-container,
.file-picker .drop-zone,
.file-picker [data-testid="file-upload"] {
    width: 100% !important;
    min-height: 118px !important;
    height: 118px !important;
    margin: 0 !important;
    padding: 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: var(--surface-2) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: 14px !important;
    color: var(--muted) !important;
    transition: 0.2s ease !important;
}

.file-picker .drop-zone:hover,
.file-picker [data-testid="file-upload"]:hover,
.file-picker .upload-container:hover {
    background: var(--surface-3) !important;
    border-color: var(--accent) !important;
}

.file-picker svg {
    width: 25px !important;
    height: 25px !important;
}

.file-picker p,
.file-picker span,
.file-picker div {
    font-size: 13px !important;
}

.file-picker button {
    min-height: 34px !important;
    height: 34px !important;
    padding: 5px 16px !important;
    border: 0 !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: white !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}

/* Upload status */
.status {
    min-height: 0 !important;
    margin: 12px 0 0 !important;
    padding: 9px 12px !important;
    background: rgba(52,211,153,0.06) !important;
    border: 1px solid rgba(52,211,153,0.13) !important;
    border-radius: 9px !important;
    color: var(--success) !important;
    font-size: 12px !important;
}

.status p {
    margin: 0 !important;
}

/* Hide internal document selector */
.document-selector {
    display: none !important;
}

/* Chat area */
.chat-card {
    width: 100% !important;
    min-height: 0 !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    margin: 0 !important;
    padding: 18px !important;
    background: rgba(16,21,33,0.92) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 50px rgba(0,0,0,0.18) !important;
}

.chat-heading {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 12px !important;
}

.chat-heading-title {
    color: var(--text) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.chat-heading-badge {
    padding: 4px 9px !important;
    border-radius: 999px !important;
    background: rgba(99,102,241,0.10) !important;
    color: #a5b4fc !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
}

.chat-window {
    width: 100% !important;
    min-height: 320px !important;
    flex: 1 1 auto !important;
    margin: 0 !important;
    padding: 6px !important;
    background: transparent !important;
    border: 0 !important;
    overflow: auto !important;
}

.chat-window .message {
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 11px 14px !important;
    border-radius: 13px !important;
    margin-bottom: 10px !important;
}

.chat-window [data-testid="user"],
.chat-window .user {
    max-width: 76% !important;
    margin-left: auto !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: white !important;
    border-radius: 14px 14px 4px 14px !important;
}

.chat-window [data-testid="bot"],
.chat-window .bot {
    max-width: 82% !important;
    margin-right: auto !important;
    background: var(--surface-2) !important;
    color: #e2e8f0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px 14px 14px 4px !important;
}

/* Empty chat state */
.empty-chat {
    width: 100% !important;
    min-height: 250px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    color: var(--muted) !important;
}

.empty-icon {
    width: 54px !important;
    height: 54px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-bottom: 12px !important;
    border-radius: 16px !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    font-size: 24px !important;
}

.empty-title {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.empty-text {
    max-width: 430px !important;
    margin-top: 5px !important;
    color: var(--muted-2) !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

/* Bottom composer */
.composer {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.ask-row {
    position: relative !important;
    width: 100% !important;
    min-height: 58px !important;
    padding: 6px 62px 6px 18px !important;
    display: flex !important;
    align-items: center !important;
    background: rgba(16,21,33,0.96) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 18px !important;
    box-shadow: 0 14px 40px rgba(0,0,0,0.28) !important;
}

.question-box {
    width: 100% !important;
    margin: 0 !important;
}

.question-box textarea {
    width: 100% !important;
    min-height: 44px !important;
    height: 44px !important;
    max-height: 44px !important;
    padding: 10px 0 !important;
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    color: var(--text) !important;
    font-size: 14px !important;
    resize: none !important;
}

.question-box textarea::placeholder {
    color: var(--muted-2) !important;
}

.ask-button {
    position: absolute !important;
    right: 9px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: white !important;
    font-size: 0 !important;
    transition: 0.2s ease !important;
}

.ask-button::after {
    content: "↑" !important;
    display: block !important;
    font-size: 20px !important;
    line-height: 40px !important;
    text-align: center !important;
    font-weight: 700 !important;
}

.ask-button:hover {
    transform: translateY(-50%) scale(1.04) !important;
}

/* Clear button */
.clear-button {
    align-self: center !important;
    margin-top: 9px !important;
    padding: 4px 9px !important;
    background: transparent !important;
    border: 0 !important;
    color: var(--muted-2) !important;
    font-size: 11px !important;
}

.clear-button:hover {
    color: #cbd5e1 !important;
    background: transparent !important;
}

/* Hide clear button when not needed */
.clear-button.hidden {
    display: none !important;
}

/* Mobile */
@media (max-width: 700px) {
    .app-shell {
        padding: 18px 12px 20px !important;
    }

    .main-page {
        min-height: calc(100vh - 38px) !important;
        gap: 12px !important;
    }

    .brand {
        font-size: 26px !important;
    }

    .document-card,
    .chat-card {
        padding: 14px !important;
        border-radius: 14px !important;
    }

    .file-picker,
    .file-picker .wrap,
    .file-picker .upload-container,
    .file-picker .drop-zone,
    .file-picker [data-testid="file-upload"] {
        min-height: 105px !important;
        height: 105px !important;
    }

    .chat-window {
        min-height: 260px !important;
    }

    .chat-window [data-testid="user"],
    .chat-window .user,
    .chat-window [data-testid="bot"],
    .chat-window .bot {
        max-width: 92% !important;
    }
}
"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="MiniRAG — Document AI Assistant") as demo:

    state = gr.State()

    with gr.Column(elem_classes="app-shell"):

        with gr.Column(elem_classes="main-page"):

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------
            with gr.Column(elem_classes="header"):

                gr.HTML(
                    """
                    <div class="logo">✦</div>
                    <div class="brand">MiniRAG</div>
                    <div class="brand-subtitle">
                        Chat with your documents using retrieval-augmented generation
                    </div>
                    """
                )

            # ------------------------------------------------
            # DOCUMENT UPLOAD
            # ------------------------------------------------
            with gr.Column(elem_classes="document-card"):

                gr.HTML(
                    """
                    <div class="section-title">📄 Add a document</div>
                    <div class="section-subtitle">
                        Upload a PDF, DOCX, or TXT file to start asking questions.
                    </div>
                    """
                )

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

            # ------------------------------------------------
            # INTERNAL DOCUMENT SELECTOR
            # ------------------------------------------------
            document_selector = gr.Dropdown(
                choices=[],
                value=None,
                show_label=False,
                visible=False,
                elem_classes="document-selector",
            )

            # ------------------------------------------------
            # CHAT
            # ------------------------------------------------
            with gr.Column(elem_classes="chat-card"):

                gr.HTML(
                    """
                    <div class="chat-heading">
                        <div class="chat-heading-title">💬 Conversation</div>
                        <div class="chat-heading-badge">Document grounded</div>
                    </div>
                    """
                )

                chatbot = gr.Chatbot(
                    show_label=False,
                    visible=False,
                    height=390,
                    elem_classes="chat-window",
                )

                gr.HTML(
                    """
                    <div class="empty-chat">
                        <div class="empty-icon">📚</div>
                        <div class="empty-title">Your document conversation starts here</div>
                        <div class="empty-text">
                            Upload a document above, then ask a question.
                            MiniRAG will answer using the information retrieved from your document.
                        </div>
                    </div>
                    """
                )

            # ------------------------------------------------
            # QUESTION COMPOSER
            # ------------------------------------------------
            with gr.Column(elem_classes="composer"):

                with gr.Column(elem_classes="ask-row"):

                    question_box = gr.Textbox(
                        placeholder="Ask a question about your document...",
                        show_label=False,
                        lines=1,
                        max_lines=1,
                        elem_classes="question-box",
                    )

                    ask_button = gr.Button(
                        "",
                        elem_classes="ask-button",
                    )

                clear_button = gr.Button(
                    "Clear conversation",
                    elem_classes="clear-button",
                    visible=False,
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