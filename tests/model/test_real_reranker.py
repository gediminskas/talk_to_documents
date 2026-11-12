"""Runs the real reranker. Slow on first run (model download): make test-model."""

from talk_to_documents.retrieval.rerank import get_reranker


def test_english_relevant_passage_scores_higher():
    scores = get_reranker().score(
        'How do I reset my password?',
        [
            'To reset your password, open Settings and choose Forgot password.',
            'The cafeteria is open from eight to four on weekdays.',
        ],
    )
    assert scores[0] > scores[1]


def test_lithuanian_relevant_passage_scores_higher():
    scores = get_reranker().score(
        'Kaip atstatyti slaptažodį?',
        [
            'Norėdami atstatyti slaptažodį, nustatymuose pasirinkite „Pamiršau slaptažodį“.',
            'Šiandien graži saulėta diena, todėl einame pasivaikščioti.',
        ],
    )
    assert scores[0] > scores[1]


def test_scores_keep_the_input_order():
    reranker = get_reranker()
    relevant = 'Postgres is a relational database.'
    irrelevant = 'Bananas are yellow fruit.'
    forward = reranker.score('What is Postgres?', [relevant, irrelevant])
    backward = reranker.score('What is Postgres?', [irrelevant, relevant])
    assert forward[0] > forward[1]
    assert backward[1] > backward[0]
