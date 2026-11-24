"""Get a JSON object out of an LLM reply, defensively.

Small local models often wrap JSON in markdown fences, add commentary, or forget a key.
`complete_json` tolerates the first two and retries (with a repair note) on the third.
"""

import json
import re
from dataclasses import dataclass

from talk_to_documents.llm import LLM, LLMError, strip_thinking

_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)


class LLMJSONError(LLMError):
    """The LLM did not return a usable JSON object, even after retries."""


@dataclass(frozen=True)
class JSONReply:
    data: dict
    input_tokens: int | None  # summed over all attempts (None if the server gave no counts)
    output_tokens: int | None
    seconds: float
    attempts: int


def extract_json_object(text: str) -> dict | None:
    """The first complete JSON object in `text`, or None.

    Removes thinking blocks and code fences, then scans for balanced braces while respecting
    strings and escapes (a greedy regex would break on braces inside string values).
    """
    text = strip_thinking(text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    position = 0
    while True:
        start = text.find('{', position)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break  # not valid JSON: look for the next opening brace
                    if isinstance(parsed, dict):
                        return parsed
                    break
        position = start + 1


def _add(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return value if total is None else total + value


def complete_json(
    llm: LLM,
    system: str | None,
    user: str,
    required_keys: list[str],
    max_tokens: int = 512,
    retries: int = 2,
    model: str | None = None,
) -> JSONReply:
    """Call the LLM until it returns a JSON object containing every required key.

    Retries add a note to the user message; repeating an identical prompt at temperature 0
    would only repeat the failure. Raises LLMJSONError after the last attempt.
    """
    repair_note = (
        '\n\nYour previous reply was not a valid JSON object with the required keys '
        f'({", ".join(required_keys)}). Reply with ONLY the JSON object, '
        'with no markdown fences and no commentary.'
    )
    input_tokens: int | None = None
    output_tokens: int | None = None
    seconds = 0.0
    last_text = ''

    for attempt in range(retries + 1):
        prompt = user if attempt == 0 else user + repair_note
        response = llm.complete(system, prompt, max_tokens=max_tokens, temperature=0.0, model=model)
        input_tokens = _add(input_tokens, response.input_tokens)
        output_tokens = _add(output_tokens, response.output_tokens)
        seconds += response.seconds
        last_text = response.text

        data = extract_json_object(response.text)
        if data is not None and all(key in data for key in required_keys):
            return JSONReply(data, input_tokens, output_tokens, seconds, attempt + 1)

    raise LLMJSONError(
        f'No valid JSON with keys {required_keys} after {retries + 1} attempts. '
        f'Last reply: {last_text[:300]!r}'
    )
