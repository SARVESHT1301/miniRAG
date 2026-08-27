import uuid
from typing import Any

from src.ingestion import ingest_document
from src.retriever import retrieve
from src.reranker import rerank
from src.generator import generate_answer
from src.memory import ConversationMemory
from src.session import Session


# ==================================================
# DOCUMENT INGESTION
# ==================================================

def ingest(
    file_path: str,
    session_id: str,
    session: Session
) -> dict[str, Any]:

    # --------------------------------------------------
    # 1. Ingest document
    # --------------------------------------------------

    result = ingest_document(
        file_path=file_path,
        session_id=session_id
    )


    # --------------------------------------------------
    # 2. Register document in session
    # --------------------------------------------------

    session.add_document(
        document_id=result["document_id"],
        filename=file_path,
        file_path=file_path,
        chunks=result["chunks"]
    )


    return result


# ==================================================
# QUERY REWRITING
# ==================================================

def build_retrieval_query(
    query: str,
    memory: ConversationMemory
) -> str:
    """
    Convert simple conversational follow-up questions
    into self-contained questions.

    Example:

    Previous:
        What is AWS Lambda?

    Current:
        What does it do?

    Result:
        What does AWS Lambda do?
    """

    messages = memory.get_recent_messages(
        limit=6
    )

    if not messages:
        return query

    previous_user_questions = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]

    if not previous_user_questions:
        return query

    previous_question = previous_user_questions[-1]

    query_lower = query.lower().strip()

    # ==================================================
    # SIMPLE FOLLOW-UP REWRITING
    # ==================================================

    # Example:
    # Previous: What is AWS Lambda?
    # Current:  What does it do?

    if query_lower == "what does it do?":

        if previous_question.lower().startswith(
            "what is "
        ):

            subject = previous_question[
                len("what is "):
            ].rstrip("?").strip()

            return (
                f"What does {subject} do?"
            )


    # ==================================================
    # "WHAT IS IT?"
    # ==================================================

    if query_lower == "what is it?":

        if previous_question.lower().startswith(
            "what is "
        ):

            subject = previous_question[
                len("what is "):
            ].rstrip("?").strip()

            return (
                f"What is {subject}?"
            )


    # ==================================================
    # "HOW DOES IT WORK?"
    # ==================================================

    if query_lower == "how does it work?":

        if previous_question.lower().startswith(
            "what is "
        ):

            subject = previous_question[
                len("what is "):
            ].rstrip("?").strip()

            return (
                f"How does {subject} work?"
            )


    # ==================================================
    # "WHY IS IT IMPORTANT?"
    # ==================================================

    if query_lower == "why is it important?":

        if previous_question.lower().startswith(
            "what is "
        ):

            subject = previous_question[
                len("what is "):
            ].rstrip("?").strip()

            return (
                f"Why is {subject} important?"
            )


    # ==================================================
    # GENERIC FOLLOW-UP
    # ==================================================

    follow_up_patterns = [
        "what kind of",
        "which",
        "why",
        "how",
        "where",
        "when",
        "tell me more",
        "explain it",
        "explain this",
        "what about it",
    ]

    is_follow_up = any(
        pattern in query_lower
        for pattern in follow_up_patterns
    )

    if is_follow_up:

        return (
            f"Previous question: "
            f"{previous_question}\n"
            f"Current question: "
            f"{query}"
        )

    return query


# ==================================================
# ASK QUESTION
# ==================================================

def ask(
    query: str,
    document_id: str,
    session_id: str,
    memory: ConversationMemory,
    session: Session,
    top_k: int = 3
) -> dict[str, Any]:

    # --------------------------------------------------
    # 1. Verify document ownership
    # --------------------------------------------------

    if not session.has_document(
        document_id
    ):

        raise ValueError(
            "Document does not belong to this session."
        )


    # --------------------------------------------------
    # 2. Build retrieval query
    # --------------------------------------------------

    retrieval_query = build_retrieval_query(
        query=query,
        memory=memory
    )


    # --------------------------------------------------
    # 3. Retrieve candidates
    # --------------------------------------------------

    retrieved_chunks = retrieve(
        query=retrieval_query,
        document_id=document_id,
        session_id=session_id,
        top_k=8
    )


    # --------------------------------------------------
    # 4. Rerank candidates
    # --------------------------------------------------

    reranked_chunks = rerank(
        query=retrieval_query,
        retrieved_chunks=retrieved_chunks,
        top_k=top_k
    )


    # --------------------------------------------------
    # 5. Conversation history
    # --------------------------------------------------

    conversation_history = (
        memory.get_recent_messages(
            limit=6
        )
    )


    # --------------------------------------------------
    # 6. Generate answer
    # --------------------------------------------------

    answer = generate_answer(
    query=retrieval_query,
    retrieved_chunks=reranked_chunks,
    conversation_history=conversation_history
)


    # --------------------------------------------------
    # 7. Save conversation
    # --------------------------------------------------

    memory.add_user_message(
        query
    )

    memory.add_assistant_message(
        answer
    )


    # --------------------------------------------------
    # 8. Return result
    # --------------------------------------------------

    return {
        "answer": answer,
        "sources": reranked_chunks,
        "conversation": memory.get_messages(),
        "retrieval_query": retrieval_query
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DOCUMENT-AWARE CONVERSATIONAL RAG")
    print("=" * 60)


    # ==================================================
    # CREATE SESSION
    # ==================================================

    session_id = str(
        uuid.uuid4()
    )

    session = Session(
        session_id=session_id
    )


    print("\nSession ID:")
    print(session_id)


    # ==================================================
    # DOCUMENT A
    # ==================================================

    document_a = ingest(
        file_path="data/cloud_computing.txt",
        session_id=session_id,
        session=session
    )

    cloud_document_id = (
        document_a["document_id"]
    )


    # ==================================================
    # DOCUMENT B
    # ==================================================

    document_b = ingest(
        file_path="data/test_document.docx",
        session_id=session_id,
        session=session
    )

    ml_document_id = (
        document_b["document_id"]
    )


    # ==================================================
    # SHOW DOCUMENTS
    # ==================================================

    print("\n" + "=" * 60)
    print("SESSION DOCUMENTS")
    print("=" * 60)

    for document_id, document in (
        session.get_documents().items()
    ):

        print(
            document_id,
            "->",
            document["filename"]
        )


    # ==================================================
    # DOCUMENT A MEMORY
    # ==================================================

    cloud_memory = session.get_memory(
        cloud_document_id
    )


    # ==================================================
    # QUESTION 1
    # ==================================================

    question_1 = (
        "What is AWS Lambda?"
    )


    result_1 = ask(
        query=question_1,
        document_id=cloud_document_id,
        session_id=session_id,
        memory=cloud_memory,
        session=session
    )


    print("\n" + "=" * 60)
    print("CLOUD DOCUMENT")
    print("=" * 60)

    print("\nQUESTION:")
    print(question_1)

    print("\nANSWER:")
    print(result_1["answer"])


    # ==================================================
    # FOLLOW-UP QUESTION
    # ==================================================

    question_2 = (
        "What does it do?"
    )


    result_2 = ask(
        query=question_2,
        document_id=cloud_document_id,
        session_id=session_id,
        memory=cloud_memory,
        session=session
    )


    print("\nQUESTION:")
    print(question_2)

    print("\nANSWER:")
    print(result_2["answer"])


    # ==================================================
    # DOCUMENT B MEMORY
    # ==================================================

    ml_memory = session.get_memory(
        ml_document_id
    )


    # ==================================================
    # QUESTION 3
    # ==================================================

    question_3 = (
        "What is supervised learning?"
    )


    result_3 = ask(
        query=question_3,
        document_id=ml_document_id,
        session_id=session_id,
        memory=ml_memory,
        session=session
    )


    print("\n" + "=" * 60)
    print("ML DOCUMENT")
    print("=" * 60)

    print("\nQUESTION:")
    print(question_3)

    print("\nANSWER:")
    print(result_3["answer"])


    # ==================================================
    # DOCUMENT B FOLLOW-UP
    # ==================================================

    question_4 = (
        "What kind of data does it use?"
    )


    result_4 = ask(
        query=question_4,
        document_id=ml_document_id,
        session_id=session_id,
        memory=ml_memory,
        session=session
    )


    print("\nQUESTION:")
    print(question_4)

    print("\nANSWER:")
    print(result_4["answer"])


    # ==================================================
    # SHOW MEMORY ISOLATION
    # ==================================================

    print("\n" + "=" * 60)
    print("MEMORY ISOLATION")
    print("=" * 60)

    print("\nCloud conversation:")

    for message in cloud_memory.get_messages():

        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )


    print("\nML conversation:")

    for message in ml_memory.get_messages():

        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )