"""Preview the contexts the LLM would write for the first chunks of one file.

Run: uv run python scripts/preview_contexts.py data/some-file.md [--chunks 5]

Nothing is written to the database and the cache is not used, so the seconds per chunk you
see are real. Read the output critically: is each context correct, in the right language,
and free of invented facts?
"""

import argparse
import time

from talk_to_documents.config import settings
from talk_to_documents.ingest.chunker import chunk_blocks, hf_token_counter
from talk_to_documents.ingest.contextual import generate_contexts
from talk_to_documents.ingest.parser import parse_file
from talk_to_documents.llm import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('file')
    parser.add_argument('--chunks', type=int, default=5, help='how many chunks to preview')
    args = parser.parse_args()

    parsed = parse_file(args.file, settings.max_upload_mb)
    count = hf_token_counter(settings.embed_model)
    chunks = chunk_blocks(
        parsed.blocks, settings.chunk_target_tokens, settings.chunk_overlap_tokens, count
    )
    document_text = '\n\n'.join(block.text for block in parsed.blocks)
    model = settings.context_model or settings.llm_model or 'unknown'

    print(f'{parsed.title}: {len(chunks)} chunks, {count(document_text)} tokens, model {model}')
    started = time.perf_counter()
    result = generate_contexts(
        parsed.title,
        document_text,
        chunks,
        get_llm(settings),
        model=model,
        max_doc_tokens=settings.context_max_doc_tokens,
        max_output_tokens=settings.context_max_output_tokens,
        count_tokens=count,
        cache=None,
        max_chunks=args.chunks,
    )
    elapsed = time.perf_counter() - started

    for chunk, context in list(zip(chunks, result.contexts, strict=True))[: args.chunks]:
        print('\n--- chunk', chunk.ordinal, f'({chunk.heading_path or "no heading"})')
        print(chunk.content[:400] + ('...' if len(chunk.content) > 400 else ''))
        print('>>> context:', context or '(none)')

    done = min(args.chunks, len(chunks))
    print(f'\n{done} chunks in {elapsed:.1f}s = {elapsed / max(done, 1):.1f}s per chunk')
    if result.failed:
        print('Generation FAILED (several errors in a row); check the LLM server.')
    per_chunk = elapsed / max(done, 1)
    total_minutes = per_chunk * len(chunks) / 60
    print(f'Estimated for all {len(chunks)} chunks of this file: {total_minutes:.1f} min')


if __name__ == '__main__':
    main()
