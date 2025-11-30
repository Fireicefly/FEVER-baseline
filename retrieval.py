"""
Document Retrieval and Sentence Selection using TF-IDF (DrQA-style).

Implements the evidence retrieval component of the FEVER baseline.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import config


class DocumentRetriever:
    """
    Document retrieval using TF-IDF.

    Retrieves the top-k most relevant Wikipedia pages for a given claim.
    """

    def __init__(self, wiki_processor):
        """
        Initialize document retriever.

        Args:
            wiki_processor: WikipediaProcessor instance
        """
        self.wiki_processor = wiki_processor
        self.vectorizer = None
        self.doc_vectors = None
        self.page_ids = None

    def build_index(self):
        """Build TF-IDF index over all Wikipedia pages."""
        if not self.wiki_processor.page_texts:
            print("Warning: No Wikipedia pages loaded. Index not built.")
            return

        print("Building TF-IDF index for document retrieval...")

        self.page_ids = list(self.wiki_processor.page_texts.keys())
        page_texts = [self.wiki_processor.page_texts[pid] for pid in self.page_ids]

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        self.doc_vectors = self.vectorizer.fit_transform(page_texts)
        print(f"Index built for {len(self.page_ids)} documents.")

    def retrieve_top_k(self, claim: str, k: int = None) -> List[str]:
        """
        Retrieve top-k most relevant documents for a claim.

        Args:
            claim: Claim text
            k: Number of documents to retrieve (default from config)

        Returns:
            List of page IDs
        """
        if k is None:
            k = config.NUM_DOCS_RETRIEVED

        if self.vectorizer is None or self.doc_vectors is None:
            print("Warning: Index not built. Returning empty list.")
            return []

        claim_vector = self.vectorizer.transform([claim])
        similarities = cosine_similarity(claim_vector, self.doc_vectors)[0]

        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_pages = [self.page_ids[idx] for idx in top_k_indices]

        return top_k_pages


class SentenceSelector:
    """
    Sentence selection using TF-IDF.

    Selects the top-l most relevant sentences from retrieved documents.
    """

    def __init__(self, wiki_processor):
        """
        Initialize sentence selector.

        Args:
            wiki_processor: WikipediaProcessor instance
        """
        self.wiki_processor = wiki_processor

    def select_sentences(
        self,
        claim: str,
        page_ids: List[str],
        l: int = None
    ) -> List[Tuple[str, int, str]]:
        """
        Select top-l most relevant sentences from given pages.

        Args:
            claim: Claim text
            page_ids: List of page IDs to search
            l: Number of sentences to retrieve (default from config)

        Returns:
            List of (page_id, sentence_id, sentence_text) tuples
        """
        if l is None:
            l = config.NUM_SENTENCES_RETRIEVED

        all_sentences = []
        sentence_metadata = []

        # Collect all sentences from retrieved pages
        for page_id in page_ids:
            sentences = self.wiki_processor.get_page_sentences(page_id)
            for sent_dict in sentences:
                sent_id = sent_dict['id']
                sent_text = sent_dict['text']
                if sent_text.strip():
                    all_sentences.append(sent_text)
                    sentence_metadata.append((page_id, sent_id, sent_text))

        if not all_sentences:
            return []

        # Build TF-IDF vectors for sentences
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            sentence_vectors = vectorizer.fit_transform(all_sentences)
            claim_vector = vectorizer.transform([claim])

            # Compute similarities
            similarities = cosine_similarity(claim_vector, sentence_vectors)[0]

            # Get top-l sentences
            top_l_indices = np.argsort(similarities)[-l:][::-1]
            top_l_sentences = [sentence_metadata[idx] for idx in top_l_indices]

            return top_l_sentences

        except ValueError:
            # Handle case where vectorizer fails
            return sentence_metadata[:l]


class EvidenceRetrieval:
    """
    Combined evidence retrieval pipeline.

    Performs document retrieval followed by sentence selection.
    """

    def __init__(self, wiki_processor):
        """
        Initialize evidence retrieval pipeline.

        Args:
            wiki_processor: WikipediaProcessor instance
        """
        self.wiki_processor = wiki_processor
        self.doc_retriever = DocumentRetriever(wiki_processor)
        self.sent_selector = SentenceSelector(wiki_processor)

    def build_index(self):
        """Build document retrieval index."""
        self.doc_retriever.build_index()

    def retrieve_evidence(self, claim: str) -> List[Tuple[str, int, str]]:
        """
        Retrieve evidence for a claim.

        Args:
            claim: Claim text

        Returns:
            List of (page_id, sentence_id, sentence_text) tuples
        """
        # Step 1: Retrieve top-k documents
        top_docs = self.doc_retriever.retrieve_top_k(claim)

        # Step 2: Select top-l sentences from those documents
        top_sentences = self.sent_selector.select_sentences(claim, top_docs)

        return top_sentences


class OracleEvidenceRetrieval:
    """
    Oracle evidence retrieval for testing.

    Uses the gold evidence from the dataset instead of retrieving.
    """

    def __init__(self, wiki_processor):
        """
        Initialize oracle retrieval.

        Args:
            wiki_processor: WikipediaProcessor instance
        """
        self.wiki_processor = wiki_processor

    def retrieve_evidence(
        self,
        claim: str,
        gold_evidence: List[List[List]]
    ) -> List[Tuple[str, int, str]]:
        """
        Retrieve gold evidence sentences.

        Args:
            claim: Claim text (not used in oracle mode)
            gold_evidence: Gold evidence from dataset

        Returns:
            List of (page_id, sentence_id, sentence_text) tuples
        """
        evidence_sentences = []

        for evidence_set in gold_evidence:
            for evidence in evidence_set:
                if len(evidence) >= 4:
                    page_id = evidence[2]
                    sent_id = evidence[3]

                    if page_id and sent_id is not None:
                        sent_text = self.wiki_processor.get_sentence(page_id, sent_id)
                        if sent_text:
                            evidence_sentences.append((page_id, sent_id, sent_text))

        return evidence_sentences
