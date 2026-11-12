from talk_to_documents.llm import LLMError, strip_thinking


def test_strip_thinking_removes_closed_blocks():
    assert strip_thinking('<think>reasoning here</think>The answer.') == 'The answer.'
    assert strip_thinking('<think>a\nmultiline\nthought</think>\n\nAnswer') == 'Answer'
    assert strip_thinking('Before <think>x</think> after <think>y</think>!') == 'Before  after !'


def test_strip_thinking_leaves_plain_text_alone():
    assert strip_thinking('  just text  ') == 'just text'
    assert strip_thinking('') == ''


def test_unclosed_think_block_is_dropped():
    assert strip_thinking('Partial answer <think>still thinking and cut o') == 'Partial answer'
    assert strip_thinking('<think>only thinking') == ''


def test_llm_error_is_a_runtime_error():
    # the CLI shows RuntimeErrors as one red line instead of a traceback
    assert issubclass(LLMError, RuntimeError)
