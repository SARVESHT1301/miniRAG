from pathlib import Path

from pypdf import PdfReader
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Clean extracted document text while preserving
    paragraph structure.
    """

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = " ".join(
            line.strip().split()
        )

        if line:
            lines.append(line)

    return "\n\n".join(lines).strip()


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file(file_path: str) -> Path:
    """
    Validate that the supplied path exists, is a file,
    and has a supported extension.

    Returns:
        Path object for the validated file.
    """

    if not file_path:
        raise ValueError(
            "No document was provided."
        )

    path = Path(file_path)

    if not path.exists():
        raise ValueError(
            "The uploaded document could not be found."
        )

    if not path.is_file():
        raise ValueError(
            "The selected path is not a file."
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported formats are .txt, .pdf, and .docx."
        )

    return path


# ============================================================
# TXT LOADER
# ============================================================

def load_txt(file_path: str) -> str:
    """
    Load a plain text file.
    """

    path = validate_file(
        file_path
    )

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            text = file.read()

    except OSError as error:

        raise ValueError(
            f"Unable to read the text document: {error}"
        ) from error

    text = normalize_text(
        text
    )

    if not text:

        raise ValueError(
            "The text document is empty."
        )

    return text


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf(file_path: str) -> str:
    """
    Extract text from all pages of a PDF.
    """

    path = validate_file(
        file_path
    )

    try:

        reader = PdfReader(
            str(path)
        )

    except Exception as error:

        raise ValueError(
            "Unable to read the PDF document. "
            "The file may be corrupted or invalid."
        ) from error

    pages: list[str] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:

            text = page.extract_text()

        except Exception:

            # Ignore an individual page that cannot
            # be extracted and continue processing.
            continue

        if text:

            cleaned = normalize_text(
                text
            )

            if cleaned:

                pages.append(
                    cleaned
                )

    document_text = "\n\n".join(
        pages
    ).strip()

    if not document_text:

        raise ValueError(
            "No readable text was found in the PDF. "
            "The document may contain only scanned images "
            "or unsupported content."
        )

    return document_text


# ============================================================
# DOCX LOADER
# ============================================================

def load_docx(file_path: str) -> str:
    """
    Extract text from a DOCX document.
    """

    path = validate_file(
        file_path
    )

    try:

        document = Document(
            str(path)
        )

    except Exception as error:

        raise ValueError(
            "Unable to read the DOCX document. "
            "The file may be corrupted or invalid."
        ) from error

    paragraphs: list[str] = []

    for paragraph in document.paragraphs:

        text = normalize_text(
            paragraph.text
        )

        if text:

            paragraphs.append(
                text
            )

    document_text = "\n\n".join(
        paragraphs
    ).strip()

    if not document_text:

        raise ValueError(
            "The DOCX document does not contain readable text."
        )

    return document_text


# ============================================================
# MAIN DOCUMENT LOADER
# ============================================================

def load_document(file_path: str) -> str:
    """
    Load a supported document and return extracted text.

    Supported formats:
        .txt
        .pdf
        .docx
    """

    path = validate_file(
        file_path
    )

    extension = path.suffix.lower()

    if extension == ".txt":

        return load_txt(
            str(path)
        )

    if extension == ".pdf":

        return load_pdf(
            str(path)
        )

    if extension == ".docx":

        return load_docx(
            str(path)
        )

    # This should normally never be reached because
    # validate_file() already checks the extension.
    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported types are .txt, .pdf, and .docx."
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    file_path = "data/test_document.docx"

    print("=" * 60)
    print("DOCUMENT LOADER TEST")
    print("=" * 60)

    print()
    print("File:", file_path)

    try:

        document = load_document(
            file_path
        )

        print(
            "Type:",
            type(document)
        )

        print(
            "Characters:",
            len(document)
        )

        print(
            "Words:",
            len(document.split())
        )

        print("\n" + "=" * 60)
        print("CONTENT")
        print("=" * 60)

        print(document)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)

        print(
            "Document loaded successfully."
        )

    except Exception as error:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(error)