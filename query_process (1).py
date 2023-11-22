import json
from collections import defaultdict
from typing import List, Tuple
from typing import Dict, Any

from documents import DocumentStore, DictDocumentStore, Document
from index import BaseIndex
from tokenizer import tokenize


def preprocess_query(query_str: str) -> list[str]:
    return tokenize(query_str)


class FullDocumentsOutputFormatter:
    def format_out(self, results: list[str], document_store: DocumentStore):
        output_string = ''
        for doc_id in results:
            doc = document_store.get_by_doc_id(doc_id)
            output_string += f'({doc.doc_id}) {doc.text}\n\n'
        return output_string


class DocIdsOnlyFormatter:
    def format_out(self, results: list[str], document_store: DocumentStore, unused_processed_query):
        return results


def format_out(results: list[str], document_store: DocumentStore, unused_processed_query) -> str:
    output_string = ''
    for doc_id in results:
        doc = document_store.get_by_doc_id(doc_id)
        output_string += f'({doc.doc_id}) {doc.text}\n\n'
    return output_string


class QueryProcess:
    def __init__(self, document_store: DocumentStore, index: BaseIndex, stopwords: set[str] = None,
                 output_formatter=FullDocumentsOutputFormatter()):
        self.document_store = document_store
        self.index = index
        self.stopwords = stopwords
        self.output_formatter = output_formatter

    # returns a dictionary of the terms along with their synonyms in a dictionary
    def read(self: str):
        thesaurus = dict()
        with open(self) as fp:
            for line in fp:
                record = json.loads(line)
                thesaurus[record['term']] = record['syns']
        return thesaurus

    def expandQueries(self, query: str, thesaurus: dict) -> dict:
        # Representation for the queries called 'querySyns'
        querySyns = {}
        terms = preprocess_query(query)
        # Iterate through each term and add synonyms to querySyns

        for term in terms:
            synonyms = thesaurus.get(term, [])
            # Add the term itself to the list of synonyms for completeness
            querySyns[term] = [term] + synonyms
        return querySyns

    def search(self, query: str, thesaurus: dict, number_of_results: int) -> str:
        # Expand query w/ synonyms
        expanded_query = self.expandQueries(query, thesaurus)

        # make dictionary to store scores for each doc
        doc_scores = defaultdict(float)

        # loop through each term and the synonyms in expanded query
        for term, synonyms in expanded_query.items():
            # look up doc_ids and scores for term and synonms
            term_entries = self.index.lookup(term)
            for synonym in synonyms:
                synonym_entries = self.index.lookup(synonym)
                # combaine entries for term and synonyms
                combined_entries = self.combine_entries(term_entries, synonym_entries)
                # update scores in doc_scores dict
                for doc_id, score in combined_entries:
                    doc_scores[doc_id] += score

        # Sort doc_ids based on their scores in descending order
        sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:number_of_results]

        # Return the formatted output
        return self.output_formatter.format_out(sorted_doc_ids, self.document_store)

    def combine_entries(self, entries1: List[Tuple[str, float]], entries2: List[Tuple[str, float]]) -> List[
        Tuple[str, float]]:
        # Combine entries from two sets while summing up scores for the same doc_id
        combined_entries = defaultdict(float)
        for doc_id, score in entries1 + entries2:
            combined_entries[doc_id] += score
        return list(combined_entries.items())
