# talk_to_documents

Ask questions about your own documents. `talk_to_documents` ingests PDF, Markdown and text
files (English and Lithuanian), stores them in Postgres with pgvector, and searches them with
both semantic (vector) and keyword search.

The long-term goal is an MCP server and a small LangGraph agent on top of this retrieval
core, so that any MCP client (Claude Desktop, Cursor) can query your knowledge base with cited
answers. **This is a work in progress.** See [Status](#status) for what exists today.

## Status

| Area | State |
|------|-------|
| Ingestion: PDF, Markdown, TXT with parsing, chunking, embeddings | working |
| Storage: Postgres 16 + pgvector (HNSW) + full-text search | working |
| Search from the command line (vector and keyword) | working |
| Tests: unit, integration (real Postgres), model checks | working |
| Hybrid fusion, reranking, contextual retrieval | in progress |
| Evaluation suite with retrieval and faithfulness metrics | planned |
| MCP server | planned |
| LangGraph agent with citation verification | planned |
| Web UI (chat, upload, save conversation) | planned |

## Requirements

- macOS or Linux
- [uv](https://docs.astral.sh/uv/) (installs the Python version pinned in `pyproject.toml`)
- Docker with Compose (for Postgres)
- About 3 GB of free disk space for the embedding model, downloaded on first use
- A few GB of free RAM while the embedding model is loaded

## Quickstart

```bash
git clone <your-repo-url> talk_to_documents
cd talk_to_documents

cp .env.example .env          # defaults work for local development
make up                       # starts Postgres 16 with pgvector on localhost:5433
uv sync                       # creates .venv and installs dependencies

uv run ttd db init            # creates the extension, tables and indexes
uv run ttd ingest data/       # ingest every PDF, Markdown and TXT file in a folder
uv run ttd search "your question here"
```

The first ingest or search downloads the embedding model
(`intfloat/multilingual-e5-large`, about 2.2 GB) once and caches it in `.cache/fastembed`.
Expect that first run to be slow.

Put your own documents in `data/`. They are ignored by git on purpose.

## Using it

| Command | What it does |
|---------|--------------|
| `uv run ttd db init` | Create or update the schema. Safe to run repeatedly. |
| `uv run ttd ingest PATH` | Ingest a file or a folder. Add `-r` to descend into subfolders and `-t TAG` (repeatable) to tag documents. |
| `uv run ttd list` | List documents with their chunk counts. |
| `uv run ttd delete ID` | Delete a document and its chunks. `ID` is a full id or a unique prefix, as shown by `list`. |
| `uv run ttd search "text"` | Search the knowledge base. Run `uv run ttd search --help` for the options. |

Ingesting the same file twice is safe: files are identified by a SHA-256 hash of their
contents, so a second ingest reports `duplicate` and changes nothing.

Tip: run `source .venv/bin/activate` once per terminal to use `ttd` without `uv run`.

### Example session

```text
$ uv run ttd ingest data/
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ file            ┃ status   ┃ chunks ┃ seconds ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ guide.md        │ ingested │ 14     │ 6.2     │
│ ataskaita.pdf   │ ingested │ 31     │ 11.8    │
└─────────────────┴──────────┴────────┴─────────┘

$ uv run ttd search "how do I start the database"
```

(Example output only; timings and chunk counts depend on your documents and machine.)

## Configuration

Settings are read from environment variables and a local `.env` file. `.env.example` lists
every setting with its default; the important ones are:

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://ttd:ttd@localhost:5433/ttd` | Postgres connection. Development credentials only. |
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` | Embedding model (1024 dimensions). The schema in `schema.sql` must match its dimension. |
| `FTS_CONFIG` | `simple` | Postgres text search configuration used for keyword search. |
| `CHUNK_TARGET_TOKENS` | `350` | Target chunk size, measured with the embedding model's tokenizer. |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between neighbouring chunks. |
| `MAX_UPLOAD_MB` | `10` | Largest file accepted. |
| `FASTEMBED_CACHE_PATH` | `./.cache/fastembed` | Where models are cached. |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | none | Any OpenAI-compatible endpoint (a commercial provider or a local server such as Unsloth Studio or Ollama). Only used by features that call an LLM. |

Never commit `.env`. It is listed in `.gitignore`.

## How it works

```text
 file ──► parse ──► chunk ──► embed ──► Postgres
        (PDF, MD,  (token-    (e5-large)   documents + chunks
         TXT)       aware)                  vector(1024) HNSW index
                                            tsvector GIN index

 query ──► embed query ──► nearest chunks (cosine)   ┐
      └──► full-text query ──► ranked chunks          ┘ shown side by side
```

Design notes:

- **One multilingual model for everything.** E5 expects the prefixes `query: ` and
  `passage: `. fastembed does not add them for this model family, so the embedder does.
- **Chunks are sized with the model's own tokenizer** and never exceed its 512-token limit.
  Consecutive paragraphs under the same heading or page are packed together, and neighbouring
  chunks overlap so sentences at a boundary are not lost.
- **Keyword search uses the `simple` text search configuration**, which works for any
  language but does no stemming. Inflected word forms (common in Lithuanian) will not match
  by keyword; vector search covers that gap.
- **Each file is written in a single transaction.** If anything fails, nothing is stored.
- **Text is normalized to Unicode NFC** so letters with diacritics are always stored the same
  way.

## Testing

```bash
make test         # unit tests: no database, no model download, a few seconds
make test-int     # integration tests against real Postgres (needs `make up`)
make test-model   # runs the real embedding model (slow on the first run)
make lint         # ruff check and format check
```

What each level covers:

- **Unit tests** cover the parsers, the chunker, hashing and path collection, and the e5
  prefix rule.
- **Integration tests** ingest documents into Postgres using a small deterministic stand-in
  for the embedding model, then check storage, duplicate handling, vector and keyword search,
  deletion, and that a failed insert leaves nothing behind. They use a separate database called
  `ttd_test`, created automatically, and never touch your real data.
- **Model tests** load the real model and check its dimension, that query and passage
  embeddings differ, that an English sentence is closer to its Lithuanian translation than to
  an unrelated sentence, and that real chunks stay under the model's token limit.

Test selection is configured with pytest markers in `pyproject.toml`, so a plain `pytest`
runs only the fast unit tests.

### Useful scripts

```bash
uv run python scripts/check_embedder.py   # verify the embedding model: dimension, prefixes
uv run python scripts/llm_smoke.py        # check your LLM endpoint and how well it returns JSON
```

## Project layout

```text
src/talk_to_documents/
  cli.py              command line interface (thin: calls the library)
  config.py           settings
  db.py               connections, schema, listing and deleting documents
  schema.sql          tables and indexes
  ingest/
    parsers.py        PDF, Markdown, TXT to normalized text blocks
    chunker.py        token-aware chunking
    pipeline.py       parse, chunk, embed, store
  retrieval/
    embed.py          embedding model wrapper
    search.py         vector and keyword queries
tests/
  unit/               fast tests, no services needed
  integration/        needs Postgres
  model/              needs the real embedding model
scripts/              helper scripts
data/                 your documents (ignored by git)
```

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| `zsh: command not found: ttd` | The virtual environment is not active. Use `uv run ttd ...` or `source .venv/bin/activate`. |
| `Failed to spawn: ttd` | The script entry is missing. `pyproject.toml` needs `[project.scripts]` with `ttd = "talk_to_documents.cli:app"`, then run `uv sync`. |
| `Cannot connect to the database` | Postgres is not running. Run `make up` and check `docker compose ps` for `healthy`. |
| `The database has no pgvector tables yet` | Run `uv run ttd db init`. |
| Port 5433 already in use | Change the host port in `docker-compose.yml` and in `DATABASE_URL`. |
| `expected 1024 dimensions, not N` | The embedding model's output size differs from the `vector(...)` size in `schema.sql`. Make them match, then recreate the database (`docker compose down -v`). |
| First ingest or search is very slow | The embedding model is being downloaded and loaded. This happens once. |
| `401 Invalid or expired API key` from the LLM script | The key in `.env` is wrong. Copy it again without spaces or quotes. |
| Integration tests cannot connect | Run `make up` first. They use the same server as the app. |

## Known limitations

- Scanned PDFs without a text layer are rejected (no OCR).
- Keyword search does not stem words (see design notes).
- A multi-word keyword query requires every word to appear in the chunk.
- Local development credentials are hard-coded in `docker-compose.yml`. Do not expose this
  database to a network.
- Embedding with a large model on CPU is slow for big document collections.

## Licence

MIT. Documents you ingest remain subject to their own licences, which is why `data/` is not
committed.
