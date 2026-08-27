import re


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [sentence.strip() for sentence in sentences if sentence.strip()]

def recursive_chunk(text, max_words):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()

        # Paragraph already fits
        if len(paragraph_words) <= max_words:
            chunks.append(paragraph)
            continue

        # Paragraph is too large → split into sentences
        sentences = split_into_sentences(paragraph)

        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()

            # A single sentence is too large → split by words
            if len(sentence_words) > max_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0

                for start in range(0, len(sentence_words), max_words):
                    word_chunk = sentence_words[start:start + max_words]
                    chunks.append(" ".join(word_chunk))

                continue

            if (
                current_chunk
                and current_word_count + len(sentence_words) > max_words
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += len(sentence_words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks
def chunk_sentences(sentences, max_words, overlap_sentences):
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if (
            current_chunk
            and current_word_count + sentence_word_count > max_words
        ):
            chunks.append(" ".join(current_chunk))

            # Keep the last few sentences for overlap
            overlap_chunk = current_chunk[-overlap_sentences:]

            current_chunk = overlap_chunk.copy()
            current_word_count = sum(
                len(s.split()) for s in current_chunk
            )

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# if __name__ == "__main__":
#     from loader import load_document

#     document = load_document("data/cloud_computing.txt")

#     sentences = split_into_sentences(document)

#     chunks = chunk_sentences(
#     sentences,
#     max_words=40,
#     overlap_sentences=1
# )

#     print("Number of sentences:", len(sentences))
#     print("Number of chunks:", len(chunks))

#     for i, chunk in enumerate(chunks):
#         print(f"\n--- Chunk {i + 1} ---")
#         print(repr(chunk))
if __name__ == "__main__":
    from loader import load_document

    document = load_document("data/cloud_computing.txt")

    chunks = recursive_chunk(
        document,
        max_words=40
    )

    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(repr(chunk))