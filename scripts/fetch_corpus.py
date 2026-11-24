"""Download the corpus listed in data/corpus.csv into data/corpus/.

Manifest columns: file, url, licence, language, title. Rows without a URL (for example
documents you wrote yourself, with licence 'self-authored') are skipped, and files that
already exist are not downloaded again. The documents themselves are never committed;
the manifest is what makes the corpus reproducible.

Run: uv run python scripts/fetch_corpus.py
"""

import csv
import sys
import urllib.request
from pathlib import Path

USER_AGENT = 'talk_to_documents corpus fetcher (research project)'


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as handle:
        return [row for row in csv.DictReader(handle) if row.get('file')]


def fetch_all(manifest: Path, destination: Path, timeout: float = 60.0) -> dict[str, list[str]]:
    """Returns the file names that were downloaded, skipped, or failed (with the reason)."""
    destination.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, list[str]] = {'downloaded': [], 'skipped': [], 'failed': []}

    for row in read_manifest(manifest):
        name = row['file'].strip()
        url = (row.get('url') or '').strip()
        target = destination / name

        if target.exists() and target.stat().st_size > 0:
            outcome['skipped'].append(name)
            continue
        if not url:
            outcome['skipped'].append(f'{name} (no URL; add the file by hand)')
            continue

        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if not data:
                raise ValueError('empty download')
            target.write_bytes(data)
            outcome['downloaded'].append(name)
        except Exception as exc:  # report and continue with the next file
            outcome['failed'].append(f'{name}: {exc}')
    return outcome


def main() -> int:
    manifest = Path('data/corpus.csv')
    if not manifest.exists():
        print(f'{manifest} not found')
        return 1
    outcome = fetch_all(manifest, Path('data/corpus'))
    print(
        f'downloaded {len(outcome["downloaded"])}, skipped {len(outcome["skipped"])}, '
        f'failed {len(outcome["failed"])}'
    )
    for item in outcome['failed']:
        print(f'  FAILED {item}')
    return 1 if outcome['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
