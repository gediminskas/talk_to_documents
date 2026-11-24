import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'fetch_corpus', Path(__file__).resolve().parents[2] / 'scripts' / 'fetch_corpus.py'
)
fetch_corpus = importlib.util.module_from_spec(spec)
sys.modules['fetch_corpus'] = fetch_corpus
spec.loader.exec_module(fetch_corpus)


def write_manifest(path: Path, rows: list[str]) -> None:
    path.write_text('file,url,licence,language,title\n' + '\n'.join(rows) + '\n')


def test_downloads_skips_existing_and_reports_failures(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('hello corpus')
    manifest = tmp_path / 'corpus.csv'
    write_manifest(
        manifest,
        [
            f'a.txt,{source.as_uri()},MIT,en,A',
            f'b.txt,{(tmp_path / "missing.txt").as_uri()},MIT,en,B',
            'c.md,,self-authored,en,C',
            f'd.txt,{source.as_uri()},MIT,en,D',
        ],
    )
    destination = tmp_path / 'corpus'
    destination.mkdir()
    (destination / 'd.txt').write_text('already here')

    outcome = fetch_corpus.fetch_all(manifest, destination)
    assert outcome['downloaded'] == ['a.txt']
    assert (destination / 'a.txt').read_text() == 'hello corpus'
    assert len(outcome['failed']) == 1 and outcome['failed'][0].startswith('b.txt')
    assert 'd.txt' in outcome['skipped'] and any(s.startswith('c.md') for s in outcome['skipped'])
    assert (destination / 'd.txt').read_text() == 'already here'  # never overwritten

    again = fetch_corpus.fetch_all(manifest, destination)
    assert again['downloaded'] == []  # second run downloads nothing new
