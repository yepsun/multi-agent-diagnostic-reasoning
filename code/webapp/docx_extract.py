"""Minimal .docx text extraction (word/document.xml → plain text)."""
import io
import re
import zipfile
import xml.etree.ElementTree as ET

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx_bytes(data: bytes) -> str:
    """Extract paragraph text from a .docx byte stream.

    Raises zipfile.BadZipFile / KeyError if the stream is not a valid docx.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    lines = []
    for p in root.iter(f"{_W_NS}p"):
        line = "".join(t.text or "" for t in p.iter(f"{_W_NS}t")).strip()
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_any(data: bytes, filename: str) -> str:
    """Route by filename: .docx → docx extraction, else UTF-8 text."""
    name = (filename or "").lower()
    if name.endswith(".docx"):
        return extract_docx_bytes(data)
    return data.decode("utf-8", errors="replace").strip()
