"""LLM-as-judge scoring: faithfulness and correctness of generated answers.

Both judges return a JudgeOutcome whose `value` is None when no score could be produced.
Use the strongest judge model you can reach for published numbers; a small local judge is
noisy and is only a development signal.
"""

from dataclasses import dataclass, field

from eval.generate import format_chunks
from talk_to_documents.llm import LLM
from talk_to_documents.llm_json import complete_json

FAITHFULNESS_SYSTEM = (
    'You are a strict evaluator of retrieval-augmented answers. You only judge whether claims '
    'are supported by the provided chunks, not whether they are true in the real world.'
)

FAITHFULNESS_USER = """Question: {question}

{chunks}

Answer to evaluate:
{answer}

Split the answer into its individual factual claims. For each claim decide whether the \
chunks support it. Ignore citation markers like [1] and any hedging words.
Return ONLY a JSON object of this shape:
{{"claims": [{{"claim": "...", "supported": true}}]}}"""

CORRECTNESS_SYSTEM = (
    'You are a strict grader. You compare an answer with a reference answer and judge '
    'whether the answer gives the same information.'
)

CORRECTNESS_USER = """Question: {question}

Reference answer: {reference}

Answer to grade: {answer}

Is the answer correct compared with the reference? Use exactly one verdict:
"correct" (same essential information), "partially_correct" (some of it, or correct but \
missing an essential part), "incorrect" (wrong, contradicting, or off topic).
Return ONLY a JSON object of this shape:
{{"verdict": "correct", "reason": "one short sentence"}}"""

VERDICT_SCORES = {'correct': 1.0, 'partially_correct': 0.5, 'incorrect': 0.0}


@dataclass(frozen=True)
class JudgeOutcome:
    value: float | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    detail: dict = field(default_factory=dict)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'true', 'yes', 'supported', '1'}
    return bool(value)


def judge_faithfulness(
    llm: LLM, question: str, chunks: list[dict], answer: str, model: str | None = None
) -> JudgeOutcome:
    """Share of the answer's claims supported by the chunks (None if the answer has no claims)."""
    reply = complete_json(
        llm,
        FAITHFULNESS_SYSTEM,
        FAITHFULNESS_USER.format(question=question, chunks=format_chunks(chunks), answer=answer),
        required_keys=['claims'],
        max_tokens=800,
        model=model,
    )
    claims = reply.data['claims']
    if not isinstance(claims, list):
        return JudgeOutcome(None, reply.input_tokens, reply.output_tokens, reply.data)
    judged = [c for c in claims if isinstance(c, dict) and 'supported' in c]
    value = None
    if judged:
        value = sum(1 for c in judged if _as_bool(c['supported'])) / len(judged)
    return JudgeOutcome(value, reply.input_tokens, reply.output_tokens, reply.data)


def judge_correctness(
    llm: LLM, question: str, reference: str, answer: str, model: str | None = None
) -> JudgeOutcome:
    """1.0 correct, 0.5 partially correct, 0.0 incorrect (None if the verdict is unreadable)."""
    reply = complete_json(
        llm,
        CORRECTNESS_SYSTEM,
        CORRECTNESS_USER.format(question=question, reference=reference, answer=answer),
        required_keys=['verdict'],
        max_tokens=300,
        model=model,
    )
    verdict = str(reply.data['verdict']).strip().lower().replace(' ', '_').replace('-', '_')
    if verdict == 'partial':
        verdict = 'partially_correct'
    return JudgeOutcome(
        VERDICT_SCORES.get(verdict), reply.input_tokens, reply.output_tokens, reply.data
    )
