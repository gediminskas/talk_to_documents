# talk_to_documents

Ask questions about your own documents. `talk_to_documents` ingests PDF, Markdown and text
files (English and Lithuanian), stores them in Postgres with pgvector, and searches them with
semantic (vector) search, keyword search, or both fused together, with optional reranking and
optional LLM-written context for every chunk.

The long-term goal is an MCP server and a small LangGraph agent on top of this retrieval
core, so that any MCP client (Claude Desktop, Cursor) can query your knowledge base with cited
answers. **This is a work in progress.** See [Status](#status) for what exists today.

## Status

| Area | State |
|------|-------|
| Ingestion: PDF, Markdown, TXT with parsing, chunking, embeddings | working |
| Storage: Postgres 16 + pgvector (HNSW) + full-text search | working |
| Search from the command line: vector, keyword, hybrid (RRF fusion) | working |
| Reranking with a multilingual cross-encoder | working |
| Contextual retrieval (LLM-written context per chunk, optional) | working |
| LLM client for any OpenAI-compatible endpoint (cloud or local) | working |
| Tests: unit, integration (real Postgres), model checks, real-LLM checks | working |
| Evaluation harness (retrieval metrics, answer generation, LLM judge, report) | code written and unit-tested; **not yet run on the real corpus** |
| MCP server | planned |
| LangGraph agent with citation verification | planned |
| Web UI (chat, upload, save conversation) | planned |

No evaluation results are published yet. The harness exists and is tested with fakes, but it
has not been run on a real corpus with a hand-reviewed question set. Whether hybrid search,
reranking and contextual retrieval actually help on this project's documents is what that run
will measure; until then, treat them as options, not proven improvements. Results will be
added to a Results section here, whatever they turn out to be.

## Requirements

- macOS or Linux
- [uv](https://docs.astral.sh/uv/) (installs the Python version pinned in `pyproject.toml`)
- Docker with Compose (for Postgres)
- About 3 GB of free disk space for the models, downloaded on first use (embedding model about
  2.2 GB, reranker about 0.6 GB)
- A few GB of free RAM while the models are loaded
- Optional: an LLM endpoint, needed only for contextual retrieval and later features (see
  [Using an LLM](#using-an-llm))

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
uv run ttd search "your question here" --rerank
```

The first ingest or search downloads the embedding model
(`intfloat/multilingual-e5-large`, about 2.2 GB) once and caches it in `.cache/fastembed`.
The first `--rerank` search downloads the reranker the same way. Expect those first runs to
be slow.

Put your own documents in `data/`. They are ignored by git on purpose.

## Using it

| Command | What it does |
|---------|--------------|
| `uv run ttd db init` | Create or update the schema. Safe to run repeatedly. |
| `uv run ttd ingest PATH` | Ingest a file or a folder. Add `-r` to descend into subfolders, `-t TAG` (repeatable) to tag documents, `--contextual` to add LLM-written context (see below). |
| `uv run ttd list` | List documents with their chunk counts. |
| `uv run ttd delete ID` | Delete a document and its chunks. `ID` is a full id or a unique prefix, as shown by `list`. |
| `uv run ttd search "text"` | Search the knowledge base. Run `uv run ttd search --help` for all options. |

Search options:

| Option | Meaning |
|--------|---------|
| `-m`, `--mode` | `hybrid` (default), `vector` or `keyword`. |
| `--rerank` / `--no-rerank` | Rescore the top fused hits with a cross-encoder (default: off). |
| `-k`, `--top-k` | How many results to show, 1 to 50 (default 5). |
| `--source-type` | Only documents of one source type (`upload`, `conversation`, `note`). |
| `-t`, `--tag` | Only documents with this tag (repeatable). |

Ingesting the same file twice is safe: files are identified by a SHA-256 hash of their
contents, so a second ingest reports `duplicate` and changes nothing.

Tip: run `source .venv/bin/activate` once per terminal to use `ttd` without `uv run`.

### Example session

```text
$ uv run ttd ingest data/
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ file            ┃ status   ┃ chunks ┃ contexts ┃ seconds ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ guide.md        │ ingested │ 14     │ 0        │ 6.2     │
│ ataskaita.pdf   │ ingested │ 31     │ 0        │ 11.8    │
└─────────────────┴──────────┴────────┴──────────┴─────────┘

$ uv run ttd search "how do I start the database" --rerank
```

The search table shows, for each hit, the fused score (`rrf`, or `rerank` when reranking),
the rank in the vector list (`vec`) and in the keyword list (`kw`), the document, the page and
a snippet. A line below it shows where the time went (embedding, each query, fusion, rerank).

(Example output only; timings and chunk counts depend on your documents and machine.)

## How search works

```text
 query ──► embed query ──► nearest chunks (cosine, HNSW)      ──┐
      └──► full-text query ──► ranked chunks (ts_rank_cd, GIN) ──┤
                                                                 ▼
                              Reciprocal Rank Fusion (ranks only, k = 60)
                                                                 ▼
                       optional: cross-encoder rescoring of the top 20
                                                                 ▼
                                                            top-k results
```

- **Hybrid** runs both searches, takes up to `CANDIDATES` (default 30) rows from each, and
  merges the two lists with Reciprocal Rank Fusion. RRF uses only each result's *rank*, never
  its score. That is deliberate: cosine scores from the embedding model sit in a narrow band
  (roughly 0.7 to 1.0), and keyword scores are on a different scale, so mixing scores would
  let one list drown out the other.
- **Reranking** is optional. A cross-encoder (`BAAI/bge-reranker-v2-m3`, int8) reads the query
  and each of the top `RERANK_TOP_N` fused chunks together and rescores them. It is slower than
  plain search, since every pair needs a model pass, but it can fix ordering mistakes.
- **Keyword search** uses the `simple` Postgres text configuration and `websearch_to_tsquery`,
  which requires every query word to appear in the chunk.

## Contextual retrieval

A chunk taken out of its document can be ambiguous ("the company's revenue grew 3%": which
company?). With contextual retrieval, an LLM writes one or two sentences placing each chunk in
its document at ingest time. That context is prepended to the text that is **embedded and
keyword-indexed**, but the chunk text shown in results stays unchanged.

```bash
# default is off; either set CONTEXTUAL=1 in .env or pass the flag per run
uv run ttd ingest data/ --contextual

# preview what the LLM writes for the first chunks of one file (writes nothing)
uv run python scripts/preview_contexts.py data/some-file.md --chunks 5
```

Things worth knowing:

- It needs a working LLM endpoint and costs one LLM call per chunk. Replies are cached in
  `.cache/contexts.jsonl`, so re-ingesting never pays twice for the same context.
- A flaky LLM never makes an ingest fail. After five consecutive failures the whole document
  falls back to no contexts and is marked `contextual = failed` in its metadata. Documents that
  were not ingested contextually are searched exactly as before.
- Documents longer than `CONTEXT_MAX_DOC_TOKENS` send the model only a window of neighbouring
  chunks around each chunk instead of the whole document.
- **Read the output of the preview script before ingesting a whole corpus.** Check that each
  context is correct, in the right language, and free of invented facts. A small local model
  can write plausible but wrong contexts, and those would make search worse.
- It is a trade-off, not a free upgrade. Whether it improves retrieval on your documents is
  something the evaluation measures; do not assume it does.

## Evaluation

The `eval/` package measures retrieval quality and answer quality on a question set you
write and review yourself. Documents and questions are not committed (see the notes below).

**Retrieval configurations compared**

| Config | Search | Reranker | Contextual ingest |
|--------|--------|----------|-------------------|
| A | keyword | no | no |
| B | vector | no | no |
| C | hybrid (RRF) | no | no |
| D | hybrid (RRF) | yes | no |
| E | hybrid (RRF) | yes | yes (second database, `ttd_ctx`) |

**Question set** (`eval/questions.jsonl`, one JSON object per line): 30 questions: 12 single-
document English, 5 single-document Lithuanian, 3 cross-language, 5 multi-document, 3
unanswerable, 2 prompt-injection. Split 10 dev / 20 test: tune on dev, report on test.
Each answerable question carries a short verbatim gold passage. Unanswerable questions have no
gold passage and the correct behaviour is to say the answer is not in the documents.

**Metrics**

- Retrieval: Hit@k, Recall@5, MRR@10, document-level hit, and p50/p95 latency.
- Generation (LLM judge): faithfulness, correctness, abstention on unanswerable questions, and
  a canary-word check that an instruction hidden in a document is not followed.
- Optional cost per query, from `LLM_PRICE_IN_PER_MTOK` and `LLM_PRICE_OUT_PER_MTOK`.

**Workflow**

```bash
uv run python scripts/fetch_corpus.py   # download the corpus listed in data/corpus.csv
uv run ttd ingest data/corpus           # plain ingest (databases A to D use this)
DATABASE_URL=postgresql://ttd:ttd@localhost:5433/ttd_ctx CONTEXTUAL=1 \
  uv run ttd db init                    # once, then:
DATABASE_URL=postgresql://ttd:ttd@localhost:5433/ttd_ctx CONTEXTUAL=1 \
  uv run ttd ingest data/corpus         # contextual ingest for config E (slow)

make draft-questions    # the LLM drafts candidates into eval/draft.md: review every one
# write the reviewed questions to eval/questions.jsonl
make eval-validate      # checks questions against the corpus and the database
make eval-retrieval     # retrieval metrics for A to E, no LLM calls
make eval               # plus answer generation and judging
```

Outputs go to `results/results.json` and `results/eval.md`. Retrieval is deterministic, so a
rerun gives the same numbers (latency aside). LLM replies are cached in `.cache/`, so reruns
are cheap and answers do not drift.

**Honest limits of this evaluation**

- Thirty questions is small. Differences of a few points between configurations are noise;
  treat results as indications, not proof.
- Questions are drafted by an LLM and then reviewed by hand. Questions that share too many
  words with their passage are flagged, because they flatter keyword search.
- The judge is an LLM, and judging with the same model that generated the answer is biased.
  Use `JUDGE_LLM_*` to point the judge at a different, stronger model when you can.
- The corpus is not committed. `data/corpus.csv` lists each source, licence and language, and
  `scripts/fetch_corpus.py` downloads it, so the setup can be reproduced.

## Using an LLM

Only contextual retrieval and later features (answer generation, the agent) call an LLM.
Plain ingest and search never do. Any server that speaks the OpenAI chat completions API works,
whether a commercial provider or a model running on your own machine. Set these in `.env`:

```text
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=local            # any non-empty value for a local server
LLM_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF
```

Check the setup with:

```bash
uv run python scripts/llm_smoke.py      # reachability, a plain reply, and how well JSON works
make test-llm                           # the real-LLM tests
```

A plain local server works well, for example llama.cpp's `llama-server`:

```bash
llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF --port 8080 --jinja -c 16384
```

Hints from experience:

- **Prefer a plain server for batch work.** One local chat-style server injected a large hidden
  prompt (more than 1,000 tokens) into every request and sometimes returned empty replies. Batch
  jobs such as contextual ingest need plain text back from every call.
- **Give the context window room.** Contextual retrieval can send up to 6,000 document tokens
  plus the chunk, so a server started with a context of 8,192 is tight.
- **Environment variables beat `.env`.** If you ever ran `source .env`, those values stay in
  that terminal and override the file. Use a fresh terminal, or
  `unset LLM_BASE_URL LLM_API_KEY LLM_MODEL`.
- **A reply cut off by `max_tokens` with no text is an error, not an empty answer.** The
  client raises a clear message, so raise `max_tokens` or `CONTEXT_MAX_OUTPUT_TOKENS`.

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
| `CANDIDATES` | `30` | Rows taken from each list before fusion. |
| `RRF_K` | `60` | Smoothing constant of Reciprocal Rank Fusion. |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3-int8` | Cross-encoder used by `--rerank`. |
| `RERANK_TOP_N` | `20` | How many fused hits the reranker scores. |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | none | Any OpenAI-compatible endpoint. Only used by features that call an LLM. |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request timeout. |
| `JUDGE_LLM_BASE_URL`, `JUDGE_LLM_API_KEY`, `JUDGE_LLM_MODEL` | empty | LLM used to judge answers. Each empty value falls back to the matching `LLM_*` value. |
| `LLM_PRICE_IN_PER_MTOK`, `LLM_PRICE_OUT_PER_MTOK` | `0` | Optional prices per million tokens, used to report cost per query. |
| `CONTEXTUAL` | `0` | Set to `1` to add LLM-written context at ingest. |
| `CONTEXT_MODEL` | empty | Model for contexts; empty means `LLM_MODEL`. |
| `CONTEXT_MAX_DOC_TOKENS` | `6000` | Longest document sent whole to the LLM. |
| `CONTEXT_MAX_OUTPUT_TOKENS` | `120` | Token limit of each generated context. |
| `CONTEXT_MAX_CHUNKS_PER_DOC` | `300` | In a longer document only the first chunks get a context. |
| `CONTEXT_CACHE_PATH` | `./.cache/contexts.jsonl` | Where generated contexts are cached. |

Never commit `.env`. It is listed in `.gitignore`.

## Design notes

- **One multilingual model for everything.** E5 expects the prefixes `query: ` and
  `passage: `. fastembed does not add them for this model family, so the embedder does.
- **Chunks are sized with the model's own tokenizer** and never exceed its 512-token limit.
  Consecutive paragraphs under the same heading or page are packed together, and neighbouring
  chunks overlap so sentences at a boundary are not lost.
- **Keyword search uses the `simple` text search configuration**, which works for any
  language but does no stemming. Inflected word forms (common in Lithuanian) will not match
  by keyword; vector search covers that gap, which is one reason to fuse the two.
- **Each file is written in a single transaction.** If anything fails, nothing is stored.
- **Text is normalized to Unicode NFC** so letters with diacritics are always stored the same
  way.
- **Fusion by rank, not by score** (see [How search works](#how-search-works)).
- **The reranker is Apache-2.0 licensed.** A popular alternative, the Jina multilingual
  reranker, is non-commercial only, so it is avoided.

## Testing

```bash
make test         # unit tests: no database, no model download, a few seconds
make test-int     # integration tests against real Postgres (needs `make up`)
make test-model   # runs the real embedding model and reranker (slow on the first run)
make test-llm     # calls your real LLM endpoint (needs LLM_* in .env)
make test-all     # everything
make lint         # ruff check and format check
```

Evaluation commands are listed under [Evaluation](#evaluation).

What each level covers:

- **Unit tests** cover the parsers, the chunker, hashing and path collection, the e5 prefix
  rule, rank fusion, reranker plumbing, the LLM helpers and the contextual-retrieval logic
  (using fake LLMs, including one that always fails).
- **Integration tests** ingest documents into Postgres using a small deterministic stand-in
  for the embedding model, then check storage, duplicate handling, vector, keyword and hybrid
  search, filters, deletion, that a failed insert leaves nothing behind, and that contextual
  ingest stores and uses contexts. They use a separate database called `ttd_test`, created
  automatically, and never touch your real data.
- **Model tests** load the real models and check the embedding dimension, that query and
  passage embeddings differ, that an English sentence is closer to its Lithuanian translation
  than to an unrelated sentence, that chunks stay under the token limit, and that the reranker
  puts a relevant passage first.
- **LLM tests** send a couple of tiny requests to your configured server.
- **Evaluation tests** cover the metric functions, question-file validation, tolerant JSON
  extraction from LLM replies, the response cache, answer generation and judging with fake
  LLMs, aggregation and the report, and, against real Postgres, the retrieval evaluation loop.

Test selection is configured with pytest markers in `pyproject.toml` (`integration`, `model`,
`llm`). A plain `pytest` runs only the fast unit tests. If `make test` unexpectedly calls your
LLM or database, the `addopts` line in `pyproject.toml` is out of date; it should read
`addopts = "-m 'not integration and not model and not llm'"`.

### Useful scripts

```bash
uv run python scripts/check_embedder.py            # embedding model: dimension, prefixes
uv run python scripts/llm_smoke.py                 # LLM endpoint and JSON reliability
uv run python scripts/preview_contexts.py FILE     # what contextual retrieval would write
uv run python scripts/compare_modes.py QUERIES     # keyword vs vector vs hybrid vs rerank
uv run python scripts/fetch_corpus.py              # download the evaluation corpus
```

`compare_modes.py` takes a text file with one query per line, optionally followed by a TAB and
the title of the document that should be found. It prints a rough side-by-side and is only a
preview of the proper evaluation, not a benchmark.

## Project layout

```text
src/talk_to_documents/
  cli.py              command line interface (thin: calls the library)
  config.py           settings
  db.py               connections, schema, listing and deleting documents
  llm.py              LLM client for any OpenAI-compatible endpoint, response cache
  llm_json.py         JSON replies from LLMs: tolerant extraction, retries
  schema.sql          tables and indexes
  ingest/
    parsers.py        PDF, Markdown, TXT to normalized text blocks
    chunker.py        token-aware chunking
    contextual.py     LLM-written context per chunk, with cache and failure handling
    pipeline.py       parse, chunk, (context), embed, store
  retrieval/
    embed.py          embedding model wrapper
    search.py         vector and keyword queries
    fusion.py         Reciprocal Rank Fusion
    rerank.py         cross-encoder reranking
    hybrid.py         the search entry point: modes, fusion, rerank, timings
eval/
  metrics.py          Hit@k, Recall@k, MRR, document hit, latency percentiles
  dataset.py          question file format and validation
  draft_questions.py  LLM-drafted candidate questions for hand review
  validate_dataset.py checks the question file against the corpus and database
  run_eval.py         runs configurations A to E, writes results/
  generate.py         answer generation from retrieved chunks
  judge.py            LLM judge: faithfulness and correctness
  answers.py          generation and judging for one question
  aggregate.py        per-configuration numbers
  report.py           results/eval.md
tests/
  unit/               fast tests, no services needed
  integration/        needs Postgres
  model/              needs the real models
  llm/                needs your LLM endpoint
scripts/              helper scripts (corpus fetching, model and LLM checks, previews)
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
| First ingest or search is very slow | A model is being downloaded and loaded. This happens once per model. |
| `AttributeError: 'Settings' object has no attribute 'candidates'` or `ImportError: cannot import name ...` after an update | Some files are still from an older day. Compare your tree with the release (`diff -rq`) and copy the newer files. |
| `401 Invalid or expired API key` from the LLM | The key in `.env` is wrong. Copy it again without spaces or quotes. |
| `404 ... model ... downloaded but not loaded` | The server has the model on disk but is not serving it. Load it, and make sure `LLM_MODEL` names the model that is actually running. |
| LLM connection error although the server is running | Wrong port in `LLM_BASE_URL`, or an old value exported in your shell. Check the port, then `unset LLM_BASE_URL LLM_API_KEY LLM_MODEL` or open a new terminal. |
| LLM replies are empty | The reply was cut off by `max_tokens`, or the server returns text outside `content`. Raise the limit and check the raw reply with `curl`. |
| Integration tests cannot connect | Run `make up` first. They use the same server as the app. |
| `make test` runs LLM or database tests | `addopts` in `pyproject.toml` is out of date (see [Testing](#testing)). |

## Known limitations

- Scanned PDFs without a text layer are rejected (no OCR).
- Keyword search does not stem words (see design notes).
- A multi-word keyword query requires every word to appear in the chunk.
- Contextual retrieval depends on the quality of your LLM, and its benefit is unmeasured so far.
- The evaluation set is small (30 questions) and has not been run on the real corpus yet.
- Reranking adds noticeable latency, because every candidate needs a model pass.
- Local development credentials are hard-coded in `docker-compose.yml`. Do not expose this
  database to a network.
- Embedding with a large model on CPU is slow for big document collections.

## Licence

MIT. Documents you ingest remain subject to their own licences, which is why `data/` is not
committed.
