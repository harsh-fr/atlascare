"""
repositories/kb_repository.py
==============================
Knowledge Base data persistence layer.

Responsibility
--------------
  Owns all read access to kb_articles.json.
  KB is read-only at runtime — articles are authored offline and
  loaded at startup. No write methods are exposed.

Design principles
-----------------
- Two indexes for fast lookup: by article_id and by tag.
- Tag index enables O(1) set lookup per tag during search.
- Returns plain dicts (shallow copies) — no internal state leaks.
- Path overridable via env var for test isolation.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "kb_articles.json"
)


class KbRepository:
    """
    JSON-backed read-only repository for KB articles.

    Maintains two in-memory indexes:
      _articles_by_id  : article_id → article dict
      _articles_by_tag : tag        → set of article_ids
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._path = os.path.abspath(
            data_path or os.getenv("KB_DATA_PATH", _DEFAULT_DATA_PATH)
        )
        self._articles_by_id:  dict[str, dict[str, Any]] = {}
        self._articles_by_tag: dict[str, set[str]]       = {}
        self._load()
        logger.debug(
            "KbRepository loaded | path=%s | articles=%d | tags=%d",
            self._path,
            len(self._articles_by_id),
            len(self._articles_by_tag),
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def find_by_id(self, article_id: str) -> dict[str, Any] | None:
        """Return article dict or None if not found."""
        article = self._articles_by_id.get(article_id)
        return dict(article) if article is not None else None

    def find_by_tags(self, tags: list[str]) -> list[dict[str, Any]]:
        """
        Return all articles that match ANY of the provided tags.

        Uses the tag index for efficient lookup — does not scan
        all articles linearly.

        Parameters
        ----------
        tags : list of tag strings (case-insensitive)

        Returns
        -------
        Deduplicated list of article dicts. Order is not guaranteed
        here — KbTool handles ranking by match score.
        """
        if not tags:
            return []

        matched_ids: set[str] = set()
        for tag in tags:
            normalised = tag.lower().strip()
            matched_ids |= self._articles_by_tag.get(normalised, set())

        return [
            dict(self._articles_by_id[aid])
            for aid in matched_ids
            if aid in self._articles_by_id
        ]

    def get_all_articles(self) -> list[dict[str, Any]]:
        """Return all articles as a list of dicts."""
        return [dict(a) for a in self._articles_by_id.values()]

    def list_all_tags(self) -> list[str]:
        """Return sorted list of all unique tags in the KB."""
        return sorted(self._articles_by_tag.keys())

    def article_count(self) -> int:
        """Return total number of articles loaded."""
        return len(self._articles_by_id)

    # ------------------------------------------------------------------
    # Private — load
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """
        Load articles from the JSON file and build both indexes.
        Starts empty if the file does not exist.
        """
        if not os.path.exists(self._path):
            logger.warning(
                "KB data file not found at '%s' — starting empty.",
                self._path,
            )
            self._articles_by_id  = {}
            self._articles_by_tag = {}
            return

        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        articles_list: list[dict] = raw.get("articles", [])

        self._articles_by_id = {
            a["article_id"]: a for a in articles_list
        }

        # Build tag → set[article_id] index
        self._articles_by_tag = {}
        for article in articles_list:
            for tag in article.get("tags", []):
                normalised = tag.lower().strip()
                if normalised not in self._articles_by_tag:
                    self._articles_by_tag[normalised] = set()
                self._articles_by_tag[normalised].add(article["article_id"])

        logger.debug(
            "KbRepository index built | articles=%d | unique_tags=%d",
            len(self._articles_by_id),
            len(self._articles_by_tag),
        )