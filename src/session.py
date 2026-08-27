from typing import Any

from src.memory import ConversationMemory


# ==================================================
# SESSION
# ==================================================

class Session:
    """
    Represents one user's RAG session.

    A session can contain multiple documents.

    Each document has:
        - filename
        - file path
        - chunk count
        - its own conversation memory
    """

    def __init__(
        self,
        session_id: str
    ) -> None:

        self.session_id = session_id

        self.documents: dict[
            str,
            dict[str, Any]
        ] = {}


    # ==================================================
    # ADD DOCUMENT
    # ==================================================

    def add_document(
        self,
        document_id: str,
        filename: str,
        file_path: str | None = None,
        chunks: int | None = None
    ) -> None:

        self.documents[document_id] = {
            "filename": filename,
            "file_path": file_path or filename,
            "chunks": chunks,
            "memory": ConversationMemory()
        }


    # ==================================================
    # REMOVE DOCUMENT
    # ==================================================

    def remove_document(
        self,
        document_id: str
    ) -> bool:

        if document_id not in self.documents:
            return False

        del self.documents[document_id]

        return True


    # ==================================================
    # GET DOCUMENTS
    # ==================================================

    def get_documents(
        self
    ) -> dict[str, dict[str, Any]]:

        return self.documents


    # ==================================================
    # GET DOCUMENT IDS
    # ==================================================

    def get_document_ids(
        self
    ) -> list[str]:

        return list(
            self.documents.keys()
        )


    # ==================================================
    # GET DOCUMENT
    # ==================================================

    def get_document(
        self,
        document_id: str
    ) -> dict[str, Any] | None:

        return self.documents.get(
            document_id
        )


    # ==================================================
    # CHECK DOCUMENT
    # ==================================================

    def has_document(
        self,
        document_id: str
    ) -> bool:

        return document_id in self.documents


    # ==================================================
    # GET DOCUMENT MEMORY
    # ==================================================

    def get_memory(
        self,
        document_id: str
    ) -> ConversationMemory:

        document = self.get_document(
            document_id
        )

        if document is None:
            raise ValueError(
                "Document does not belong to this session."
            )

        return document["memory"]


    # ==================================================
    # CLEAR DOCUMENT MEMORY
    # ==================================================

    def clear_memory(
        self,
        document_id: str
    ) -> None:

        memory = self.get_memory(
            document_id
        )

        memory.clear()


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-DOCUMENT SESSION TEST")
    print("=" * 60)

    session = Session(
        session_id="session-123"
    )

    # --------------------------------------------------
    # Document A
    # --------------------------------------------------

    session.add_document(
        document_id="document-123",
        filename="cloud_computing.txt"
    )

    # --------------------------------------------------
    # Document B
    # --------------------------------------------------

    session.add_document(
        document_id="document-456",
        filename="machine_learning.txt"
    )

    # --------------------------------------------------
    # Document C
    # --------------------------------------------------

    session.add_document(
        document_id="document-789",
        filename="project_report.pdf"
    )

    print("\nDocuments:")

    print(
        session.get_documents()
    )

    # --------------------------------------------------
    # Add conversation to Document A
    # --------------------------------------------------

    memory_a = session.get_memory(
        "document-123"
    )

    memory_a.add_user_message(
        "What is AWS Lambda?"
    )

    memory_a.add_assistant_message(
        "AWS Lambda is a serverless compute service."
    )

    # --------------------------------------------------
    # Add conversation to Document B
    # --------------------------------------------------

    memory_b = session.get_memory(
        "document-456"
    )

    memory_b.add_user_message(
        "What is supervised learning?"
    )

    memory_b.add_assistant_message(
        "Supervised learning uses labeled training data."
    )

    print("\nCloud document memory:")

    print(
        memory_a.get_messages()
    )

    print("\nMachine learning document memory:")

    print(
        memory_b.get_messages()
    )

    # --------------------------------------------------
    # Verify isolation
    # --------------------------------------------------

    print("\nMemory A count:")
    print(memory_a.count())

    print("\nMemory B count:")
    print(memory_b.count())

    # --------------------------------------------------
    # Remove document
    # --------------------------------------------------

    removed = session.remove_document(
        "document-123"
    )

    print("\nRemoved document-123:")
    print(removed)

    print("\nFinal documents:")
    print(
        session.get_documents()
    )