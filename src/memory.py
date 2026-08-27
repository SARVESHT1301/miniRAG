from typing import Any


# ==================================================
# CONVERSATION MEMORY
# ==================================================

class ConversationMemory:

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []


    # ==================================================
    # ADD USER MESSAGE
    # ==================================================

    def add_user_message(
        self,
        message: str
    ) -> None:

        self.messages.append(
            {
                "role": "user",
                "content": message
            }
        )


    # ==================================================
    # ADD ASSISTANT MESSAGE
    # ==================================================

    def add_assistant_message(
        self,
        message: str
    ) -> None:

        self.messages.append(
            {
                "role": "assistant",
                "content": message
            }
        )


    # ==================================================
    # GET MESSAGES
    # ==================================================

    def get_messages(
        self
    ) -> list[dict[str, str]]:

        return self.messages.copy()


    # ==================================================
    # GET RECENT MESSAGES
    # ==================================================

    def get_recent_messages(
        self,
        limit: int = 6
    ) -> list[dict[str, str]]:

        return self.messages[-limit:]


    # ==================================================
    # CLEAR MEMORY
    # ==================================================

    def clear(self) -> None:

        self.messages.clear()


    # ==================================================
    # MESSAGE COUNT
    # ==================================================

    def count(self) -> int:

        return len(self.messages)


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    memory = ConversationMemory()

    print("=" * 60)
    print("CONVERSATION MEMORY TEST")
    print("=" * 60)

    memory.add_user_message(
        "What is supervised learning?"
    )

    memory.add_assistant_message(
        "Supervised learning uses labeled training data."
    )

    memory.add_user_message(
        "What kind of data does it use?"
    )

    memory.add_assistant_message(
        "It uses labeled training data."
    )

    print("\nMessages:")
    print(memory.get_messages())

    print("\nRecent messages:")
    print(memory.get_recent_messages())

    print("\nMessage count:")
    print(memory.count())

    memory.clear()

    print("\nAfter clearing:")
    print(memory.get_messages())