import pytest

from talk_to_documents.ingest.chunker import MAX_TOKENS_FACTOR, chunk_blocks
from talk_to_documents.ingest.parser import Block


def words(text):
    return len(text.split())


def sentences(n, start=0, length=8):
    return ' '.join(' '.join(f'w{start + i}x{j}' for j in range(length)) + '.' for i in range(n))


TARGET, OVERLAP = 50, 10


def run(blocks, target=TARGET, overlap=OVERLAP):
    return chunk_blocks(blocks, target, overlap, words)


def test_order_ordinals_nonempty_max():
    blocks = [Block(sentences(40), page=1), Block(sentences(30, 100), page=2)]
    chunks = run(blocks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.content.strip() for c in chunks)
    assert all(c.token_count <= TARGET * MAX_TOKENS_FACTOR for c in chunks)
    pages = [c.page for c in chunks]
    assert pages == sorted(pages)


def test_overlap_and_coverage():
    text = sentences(40)
    chunks = run([Block(text)])
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:], strict=False):
        a_words = a.content.split()
        b_words = b.content.split()
        assert set(a_words[-OVERLAP:]) & set(b_words[:OVERLAP]), 'no overlap'
    all_words = ' '.join(c.content for c in chunks).split()
    assert set(text.split()) <= set(all_words)


def test_no_overlap_across_headings_or_pages():
    blocks = [Block(sentences(10), heading_path='A'), Block(sentences(10, 500), heading_path='B')]
    chunks = run(blocks)
    a = [c for c in chunks if c.heading_path == 'A']
    b = [c for c in chunks if c.heading_path == 'B']
    assert a and b
    assert not (set(a[-1].content.split()) & set(b[0].content.split()))


def test_deterministic():
    blocks = [Block(sentences(40))]
    assert run(blocks) == run(blocks)


def test_small_paragraphs_pack_together():
    blocks = [Block(f'Short paragraph number {i}.') for i in range(20)]
    chunks = run(blocks, target=30, overlap=5)
    assert len(chunks) < 10
    assert all(c.page is None for c in chunks)


def test_long_sentence_hard_split():
    long = ' '.join(f'word{i}' for i in range(200)) + '.'
    chunks = run([Block(long)])
    assert len(chunks) >= 4
    assert all(c.token_count <= TARGET * MAX_TOKENS_FACTOR for c in chunks)
    assert {f'word{i}' for i in range(200)} <= set(
        ' '.join(c.content for c in chunks).replace('.', '').split()
    )


def test_code_fence_kept_together():
    code = '```\n' + '\n\n'.join(f'line{i}' for i in range(5)) + '\n```'
    chunks = run([Block('Intro text here.\n\n' + code + '\n\nAfter text.')], target=100, overlap=10)
    joined = [c for c in chunks if '```' in c.content]
    assert joined and all(c.content.count('```') == 2 for c in joined)


def test_tiny_tail_merged():
    # 6 sentences of 8 words = 48 words fits one chunk; add 1 short trailing sentence
    text = sentences(6) + ' End.'
    chunks = run([Block(text)], target=50, overlap=10)
    assert len(chunks) == 1


def test_unsplittable_word_allowed():
    chunks = run([Block('x' * 10 + ' ' + 'ok.')], target=5, overlap=1)
    assert chunks


def test_invalid_args():
    with pytest.raises(ValueError):
        run([Block('a')], target=0)
    with pytest.raises(ValueError):
        run([Block('a')], target=10, overlap=10)


def test_empty_blocks():
    assert run([Block('   '), Block('')]) == []


def test_lithuanian():
    text = 'Šiandien graži diena. Ąžuolas auga prie upės. Žuvys plaukia upe. ' * 30
    chunks = run([Block(text)])
    assert all('Š' in c.content or 'Ą' in c.content or 'Ž' in c.content for c in chunks)
