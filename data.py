import os
import csv
import json
from typing import Dict, Iterable, List, Optional, Tuple


class MindDataset:
    """
    Data access layer for MIND-small.

    Exposes:
    - iter_docs(split): yield (doc_id, text, meta)
    - iter_sessions(split): yield (user_id, session_id, history_ids, candidate_ids, labels)
    - get_doc(doc_id): return (text, meta)
    """

    def __init__(self, root_dir: str, train_subdir: str = "data/MINDsmall_train", dev_subdir: str = "data/MINDsmall_dev") -> None:
        self.root_dir = root_dir
        self.split_dirs = {
            "train": os.path.join(root_dir, train_subdir),
            "dev": os.path.join(root_dir, dev_subdir),
            "valid": os.path.join(root_dir, dev_subdir),
        }

        # Lazy-loaded caches
        self._news_cache: Dict[str, Dict[str, dict]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_split_dir(self, split: str) -> str:
        if split not in self.split_dirs:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self.split_dirs.keys())}")
        split_dir = self.split_dirs[split]
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        return split_dir

    def _load_news(self, split: str) -> Dict[str, dict]:
        """
        Load news.tsv for the given split and cache it.

        news.tsv columns (MIND-small):
        0: news_id
        1: category
        2: subcategory
        3: title
        4: abstract
        5: url
        6: title_entities (JSON)
        7: abstract_entities (JSON)
        """
        if split in self._news_cache:
            return self._news_cache[split]

        split_dir = self._get_split_dir(split)
        news_path = os.path.join(split_dir, "news.tsv")
        news: Dict[str, dict] = {}

        with open(news_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                # Some lines might be shorter if entities are missing
                row = row + [""] * (8 - len(row))
                news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = row[:8]
                text = self._clean_text(f"{title} {abstract}".strip())

                meta = {
                    "news_id": news_id,
                    "category": category,
                    "subcategory": subcategory,
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "title_entities": self._safe_json_loads(title_entities),
                    "abstract_entities": self._safe_json_loads(abstract_entities),
                }
                news[news_id] = {"text": text, "meta": meta}

        self._news_cache[split] = news
        return news

    @staticmethod
    def _safe_json_loads(s: str):
        s = s.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # In case of malformed JSON, just return raw string for debugging
            return s

    @staticmethod
    def _clean_text(text: str) -> str:
        # Minimal cleaning: strip whitespace, collapse spaces.
        # More advanced cleaning can be added if needed.
        return " ".join(text.split())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_docs(self, split: str) -> Iterable[Tuple[str, str, dict]]:
        """
        Iterate over documents in the given split.

        Yields:
            (doc_id, text, meta)
        """
        news = self._load_news(split)
        for doc_id, obj in news.items():
            yield doc_id, obj["text"], obj["meta"]

    def get_doc(self, doc_id: str, split: str = "train") -> Optional[Tuple[str, dict]]:
        """
        Get a document by id from the given split.

        Returns:
            (text, meta) or None if not found.
        """
        news = self._load_news(split)
        if doc_id not in news:
            return None
        obj = news[doc_id]
        return obj["text"], obj["meta"]

    def iter_sessions(
        self, split: str
    ) -> Iterable[Tuple[str, str, List[str], List[str], List[int]]]:
        """
        Iterate over user sessions for the given split.

        Yields:
            (user_id, session_id, history_ids, candidate_doc_ids, labels)
        """
        split_dir = self._get_split_dir(split)
        behaviors_path = os.path.join(split_dir, "behaviors.tsv")

        with open(behaviors_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                # behaviors.tsv columns:
                # 0: impression_id
                # 1: user_id
                # 2: time
                # 3: history (space separated news_ids or empty)
                # 4: impressions (space separated 'news_id-label')
                row = row + [""] * (5 - len(row))
                impression_id, user_id, _, history, impressions = row[:5]

                history_ids = history.strip().split() if history.strip() else []

                cand_ids: List[str] = []
                labels: List[int] = []
                impressions = impressions.strip().split()
                for imp in impressions:
                    # Example: N55689-1 or N35729-0
                    if "-" not in imp:
                        continue
                    nid, label_str = imp.rsplit("-", 1)
                    try:
                        label = int(label_str)
                    except ValueError:
                        label = 0
                    cand_ids.append(nid)
                    labels.append(label)

                yield user_id, impression_id, history_ids, cand_ids, labels


__all__ = ["MindDataset"]


