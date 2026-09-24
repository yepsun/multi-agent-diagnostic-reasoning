import io
import zipfile

from webapp.docx_extract import extract_any, extract_docx_bytes


def _make_docx(paragraphs):
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>'
        for p in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


class TestDocxExtract:
    def test_paragraphs_extracted_in_order(self):
        data = _make_docx(["患者男性，55岁。", "主诉：发热8个月。", ""])
        text = extract_docx_bytes(data)
        assert text.splitlines() == ["患者男性，55岁。", "主诉：发热8个月。"]

    def test_multiple_runs_in_one_paragraph(self):
        w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        body = ('<w:p><w:r><w:t>体温</w:t></w:r><w:r><w:t>39.0℃</w:t></w:r></w:p>')
        xml = ('<?xml version="1.0"?>'
               f'<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", xml)
        assert extract_docx_bytes(buf.getvalue()) == "体温39.0℃"

    def test_extract_any_routes_by_extension(self):
        assert extract_any("纯文本病例".encode("utf-8"), "case.txt") == "纯文本病例"
        assert extract_any(_make_docx(["docx内容"]), "case.DOCX") == "docx内容"

    def test_invalid_docx_raises(self):
        import pytest
        with pytest.raises(Exception):
            extract_docx_bytes(b"not a zip")
