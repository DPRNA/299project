from collections import defaultdict
from typing import List, Tuple
from documents import TransformedDocument


def count_terms(terms: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    for t in terms:
        counts[t] += 1
    return counts


def combine_term_scores(terms: list[str], score: dict[str, float]) -> float:
    return sum([score[term] for term in terms])


class BaseIndex:
    def __init__(self):
        self.id_to_term_counts: dict[str, dict[str, float]] = dict()

    def add_document(self, doc: TransformedDocument):
        pass

    def search(self, processed_query: list[str], number_of_results: int) -> list[str]:
        pass

    def lookup(self, term: str) -> List[Tuple[str, float]]:
        # Return list of tuples with doc ids and scores for given term
        return [(doc_id, counts.get(term, 0.0)) for doc_id, counts in self.id_to_term_counts.items()]


class Index(BaseIndex):
    def __init__(self):
        self.id_to_term_counts: dict[str, dict[str, float]] = dict()

    def add_document(self, doc: TransformedDocument):
        # Mapping from doc_ids to term counts in the corresponding document.
        self.id_to_term_counts[doc.doc_id] = count_terms(doc.terms)

    def search(self, processed_query: list[str], number_of_results: int) -> list[str]:
        scores = dict()
        for doc_id, counts in self.id_to_term_counts.items():
            score = combine_term_scores(processed_query, self.id_to_term_counts[doc_id])
            scores[doc_id] = score
        return sorted(self.id_to_term_counts.keys(), key=scores.get, reverse=True)[:number_of_results]
