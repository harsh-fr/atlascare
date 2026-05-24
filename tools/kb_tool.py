"""
tools/kb_tool.py
================
Knowledge Base (KB) integration tool.

Responsibility
--------------
  Exposes typed, async methods the Executor calls to retrieve
  policy articles and FAQ content:
    - search()      : find articles by tags
    - get_article() : fetch a single article by ID

Design principles
-----------------
- KB content grounds LLM responses in actual policy — never invented.
- Tag-based search is deterministic; results are ranked by relevance
  (number of matching tags), not by LLM scoring.
- Articles are returned with their full content so the ResponseBuilder
  can pass verified policy text to Gemini as grounding context.
- Returns plain dicts only — no repository objects leak out.
"""

import logging
from typing import Any

from repositories.kb_repository import KbRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class KbError(Exception):
    """Base error for all KB tool failures."""

class ArticleNotFoundError(KbError):
    """Article ID does not exist in the knowledge base."""


# ---------------------------------------------------------------------------
# KbTool
# ---------------------------------------------------------------------------
class KbTool:
    """
    Typed async interface to the Knowledge Base.

    Backed today by JSON repository; stable interface supports
    migration to a vector store or enterprise KB API.
    """

    def __init__(self) -> None:
        self._kb_repo = KbRepository()
        logger.debug("KbTool initialised.")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    async def search(
        self,
        tags: list[str],
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for KB articles that match any of the provided tags.

        Results are ranked by number of matching tags (descending),
        then by last_updated (newest first) as a tiebreaker.

        Parameters
        ----------
        tags        : list of tag strings to match against article tags
        max_results : maximum number of articles to return (default 5)

        Returns
        -------
        List of article dicts, best match first.
        Each dict includes: article_id, title, tags, content,
                            last_updated, applies_to, match_score.
        """
        logger.debug(
            "KbTool.search | tags=%s | max_results=%d",
            tags,
            max_results,
        )

        if not tags:
            logger.warning("KbTool.search called with empty tags list.")
            return []

        # Normalise tags to lowercase for case-insensitive matching
        query_tags = {t.lower().strip() for t in tags if t}

        all_articles = self._kb_repo.get_all_articles()

        # Score each article by number of matching tags
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for article in all_articles:
            article_tags = {
                t.lower().strip()
                for t in article.get("tags", [])
            }
            match_score = len(query_tags & article_tags)
            if match_score > 0:
                scored.append((
                    match_score,
                    article.get("last_updated", ""),
                    article,
                ))

        # Sort: most matching tags first, then newest article first
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        results = []
        for score, _, article in scored[:max_results]:
            enriched = dict(article)
            enriched["match_score"] = score
            results.append(enriched)

        logger.info(
            "KbTool.search | tags=%s | matched=%d | returned=%d",
            tags,
            len(scored),
            len(results),
        )

        return results

    # ------------------------------------------------------------------
    # get_article
    # ------------------------------------------------------------------
    async def get_article(self, article_id: str) -> dict[str, Any]:
        """
        Fetch a single KB article by its ID.

        Parameters
        ----------
        article_id : e.g. "KB-001"

        Returns
        -------
        dict matching the kb_articles.json schema.

        Raises
        ------
        ArticleNotFoundError  if article_id does not exist.
        """
        logger.debug("KbTool.get_article | article_id=%s", article_id)

        article = self._kb_repo.find_by_id(article_id)
        if article is None:
            raise ArticleNotFoundError(
                f"KB article '{article_id}' not found."
            )

        return article