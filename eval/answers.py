"""Generate and judge the answer for one question."""

from eval.dataset import Question
from eval.generate import generate_answer
from eval.judge import judge_correctness, judge_faithfulness
from talk_to_documents.llm import LLM, LLMError

TOP_CHUNKS_FOR_ANSWER = 5


def _add(a: int | None, b: int | None) -> int | None:
    if b is None:
        return a
    return b if a is None else a + b


def answer_and_judge(
    question: Question,
    chunks: list[dict],
    generator: LLM,
    judge: LLM,
    *,
    generator_model: str | None = None,
    judge_model: str | None = None,
    top_chunks: int = TOP_CHUNKS_FOR_ANSWER,
) -> dict:
    """One row of generation results. Judge failures are recorded, never raised, so one bad
    reply cannot abort a long run."""
    used = chunks[:top_chunks]
    generated = generate_answer(generator, question.question, used, model=generator_model)

    row = {
        'id': question.id,
        'type': question.type,
        'language': question.language,
        'split': question.split,
        'answer': generated.text,
        'not_found': generated.not_found,
        'faithfulness': None,
        'correctness': None,
        'canary_leaked': None,
        'gen_input_tokens': generated.input_tokens,
        'gen_output_tokens': generated.output_tokens,
        'gen_seconds': generated.seconds,
        'judge_input_tokens': None,
        'judge_output_tokens': None,
        'judge_error': None,
    }
    if question.canary:
        row['canary_leaked'] = question.canary.lower() in generated.text.lower()

    if not question.answerable:
        return row  # abstention is a plain string check, nothing to judge

    if generated.not_found:
        row['correctness'] = 0.0  # refusing to answer an answerable question is incorrect
        return row

    try:
        faithfulness = judge_faithfulness(
            judge, question.question, used, generated.text, model=judge_model
        )
        correctness = judge_correctness(
            judge, question.question, question.reference_answer, generated.text, model=judge_model
        )
    except LLMError as exc:
        row['judge_error'] = str(exc)
        return row

    row['faithfulness'] = faithfulness.value
    row['correctness'] = correctness.value
    for outcome in (faithfulness, correctness):
        row['judge_input_tokens'] = _add(row['judge_input_tokens'], outcome.input_tokens)
        row['judge_output_tokens'] = _add(row['judge_output_tokens'], outcome.output_tokens)
    return row
