import unicodedata

import pytest

from talk_to_documents.ingest.parser import ParseError, parse_file


def test_md(tmp_path):
    p = tmp_path / 'a.md'
    p.write_text(
        '---\ntitle: x\n---\n# Top\nintro\n## Install\n### Docker\nrun it\n'
        '```\n# not heading\n```\n## Other\nbye\n'
    )
    d = parse_file(p)
    assert d.title == 'Top'
    paths = [b.heading_path for b in d.blocks]
    assert paths == ['Top', 'Top > Install > Docker', 'Top > Other'], paths
    assert '# not heading' in d.blocks[1].text


def test_txt_nfc(tmp_path):
    p = tmp_path / 'a.txt'
    decomposed = unicodedata.normalize('NFD', 'Šiandien ąžuolas')
    p.write_text(decomposed + '\n\n\n\nantra pastraipa\r\n', encoding='utf-8')
    d = parse_file(p)
    assert len(d.blocks) == 2
    assert d.blocks[0].text == unicodedata.normalize('NFC', decomposed)


def test_errors(tmp_path):
    e = tmp_path / 'e.txt'
    e.write_text('')
    with pytest.raises(ParseError):
        parse_file(e)
    f = tmp_path / 'f.pdf'
    f.write_bytes(b'hello not pdf')
    with pytest.raises(ParseError):
        parse_file(f)
    b = tmp_path / 'b.txt'
    b.write_bytes(b'\xff\xfe\x00bad\xc3')
    with pytest.raises(ParseError):
        parse_file(b)
    big = tmp_path / 'big.txt'
    big.write_text('a' * (2 * 1024 * 1024))
    with pytest.raises(ParseError):
        parse_file(big, max_upload_mb=1)
    x = tmp_path / 'x.docx'
    x.write_text('hi')
    with pytest.raises(ParseError):
        parse_file(x)


def test_pdf(tmp_path):
    canvas = pytest.importorskip('reportlab.pdfgen.canvas')
    p = tmp_path / 'a.pdf'
    c = canvas.Canvas(str(p))
    c.drawString(72, 720, 'Hello page one')
    c.showPage()
    c.drawString(72, 720, 'Second page text')
    c.showPage()
    c.save()
    d = parse_file(p)
    assert [b.page for b in d.blocks] == [1, 2]
    assert 'Hello' in d.blocks[0].text


def test_pdf_blank(tmp_path):
    canvas = pytest.importorskip('reportlab.pdfgen.canvas')
    p = tmp_path / 'b.pdf'
    c = canvas.Canvas(str(p))
    c.showPage()
    c.save()
    with pytest.raises(ParseError, match='scanned'):
        parse_file(p)


def test_dehyphen():
    from talk_to_documents.ingest.parser import _dehyphenate

    assert _dehyphenate('inter-\nnational') == 'international'
    assert _dehyphenate('well-\nKnown') == 'well-\nKnown'
