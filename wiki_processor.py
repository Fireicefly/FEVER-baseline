"""
Wikipedia page processor for FEVER.

Handles loading and indexing Wikipedia pages.
"""

import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import config


class WikipediaProcessor:
    """Process Wikipedia pages for document retrieval."""

    def __init__(self, wiki_dir: str = None):
        """
        Initialize Wikipedia processor.

        Args:
            wiki_dir: Directory containing wiki-pages (JSONL files)
        """
        self.wiki_dir = wiki_dir or config.WIKI_DIR
        self.pages = {}  # page_id -> page_data
        self.page_texts = {}  # page_id -> full_text
        self.page_sentences = {}  # page_id -> list of sentences

    def load_wikipedia_pages(self):
        """Load all Wikipedia pages from JSONL files."""
        if not os.path.exists(self.wiki_dir):
            print(f"Warning: Wikipedia directory {self.wiki_dir} not found.")
            print("Set DOWNLOAD_WIKIPEDIA=True in config.py to download.")
            return

        print("Loading Wikipedia pages...")

        wiki_files = []
        for root, dirs, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    wiki_files.append(os.path.join(root, file))

        for wiki_file in tqdm(wiki_files, desc="Processing wiki files"):
            with open(wiki_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        page = json.loads(line)
                        page_id = page.get('id', '')
                        self.pages[page_id] = page
                        self._process_page(page_id, page)

        print(f"Loaded {len(self.pages)} Wikipedia pages.")

    def _process_page(self, page_id: str, page: Dict):
        """
        Process a single Wikipedia page.

        Args:
            page_id: Page ID
            page: Page data dictionary
        """
        lines = page.get('lines', '')

        sentences = []
        full_text = []

        # Parse lines - they are tab-separated strings, not JSON
        # Format: sentence_id\tsentence_text[\tentity_text\tentity_link\t...]
        if isinstance(lines, str) and lines.strip():
            # Split by newlines to get individual line entries
            for line in lines.strip().split('\n'):
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            sent_id = int(parts[0])
                            sent_text = parts[1]  # Sentence is the second field

                            # Store sentence with ID
                            sentences.append({
                                'id': sent_id,
                                'text': sent_text
                            })
                            full_text.append(sent_text)
                        except (ValueError, IndexError):
                            # Skip malformed lines
                            continue

        self.page_sentences[page_id] = sentences
        self.page_texts[page_id] = ' '.join(full_text)

    def get_page_text(self, page_id: str) -> str:
        """Get full text of a page."""
        return self.page_texts.get(page_id, '')

    def get_page_sentences(self, page_id: str) -> List[Dict]:
        """Get sentences from a page."""
        return self.page_sentences.get(page_id, [])

    def get_sentence(self, page_id: str, sentence_id: int) -> str:
        """Get a specific sentence from a page."""
        sentences = self.get_page_sentences(page_id)
        for sent in sentences:
            if sent['id'] == sentence_id:
                return sent['text']
        return ''

    def page_exists(self, page_id: str) -> bool:
        """Check if a page exists."""
        return page_id in self.pages


class MockWikipediaProcessor(WikipediaProcessor):
    """
    Mock Wikipedia processor for when the full dump is not available.

    This assumes that the correct evidence is provided in the dataset.
    """

    def __init__(self):
        """Initialize mock processor."""
        super().__init__()
        self.mock_mode = True
        print("\n" + "="*80)
        print("Running in MOCK mode (Wikipedia not downloaded)")
        print("Assuming correct evidence is provided in the dataset.")
        print("="*80 + "\n")

    def load_wikipedia_pages(self):
        """No-op for mock mode."""
        pass

    def get_page_text(self, page_id: str) -> str:
        """Return a mock page text."""
        return f"Mock content for page: {page_id}"

    def get_page_sentences(self, page_id: str) -> List[Dict]:
        """Return mock sentences."""
        return [
            {'id': i, 'text': f"Mock sentence {i} from {page_id}"}
            for i in range(10)
        ]

    def get_sentence(self, page_id: str, sentence_id: int) -> str:
        """Return a mock sentence."""
        return f"Mock sentence {sentence_id} from page {page_id}"

    def page_exists(self, page_id: str) -> bool:
        """Always return True in mock mode."""
        return True
