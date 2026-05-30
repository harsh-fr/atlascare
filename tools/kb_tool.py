import logging
from typing import Any

from repositories.kb_repository import KbRepository

logger = logging.getLogger(__name__)


class KbError(Exception):
    pass

class ArticleNotFoundError(KbError):
    pass


def rank_articles(
    query_tags: set[str],
    candidates: list[dict[str, Any]],
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Rank KB articles for a tag query. Pure + deterministic.

    Order:
      1. match_score  — number of query tags the article carries (primary).
      2. precision    — share of the article's OWN tags that matched, so a focused
                        article genuinely ABOUT the query beats one that merely
                        brushes it on a single shared tag (e.g. the general 'Refund
                        Policy' is not crowded out by the COD-specific article when
                        both match 'refund'). This is the relevance tie-break.
      3. article_id   — stable, deterministic final tie-break.
    Edit-recency ('last_updated') is intentionally NOT used: it is not a relevance
    signal and let an unrelated, more-recently-edited article jump the queue.
    """
    scored: list[tuple[int, float, str, dict[str, Any]]] = []
    for article in candidates:
        article_tags = {t.lower().strip() for t in article.get("tags", [])}
        match_score = len(query_tags & article_tags)
        if match_score > 0:
            precision = match_score / len(article_tags) if article_tags else 0.0
            scored.append((match_score, precision, article.get("article_id", ""), article))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

    results: list[dict[str, Any]] = []
    for score, _precision, _aid, article in scored[:max_results]:
        enriched = dict(article)
        enriched["match_score"] = score
        results.append(enriched)
    return results


class KbTool:
    def __init__(self) -> None:
        self._kb_repo = KbRepository()
        logger.debug("KbTool initialised.")

    async def search(self, tags: list[str], max_results: int = 5) -> list[dict[str, Any]]:
        logger.debug("KbTool.search | tags=%s | max_results=%d", tags, max_results)
        if not tags:
            logger.warning("KbTool.search called with empty tags list.")
            return []

        query_tags = {t.lower().strip() for t in tags if t}
        # Use the repository's tag index to fetch only candidate articles that
        # match at least one tag, then rank them below — instead of scanning every
        # article. (The result set is identical.)
        candidates = self._kb_repo.find_by_tags(tags)
        results = rank_articles(query_tags, candidates, max_results)

        logger.info(
            "KbTool.search | tags=%s | returned=%d", tags, len(results),
        )
        return results

    async def get_article(self, article_id: str) -> dict[str, Any]:
        logger.debug("KbTool.get_article | article_id=%s", article_id)
        article = self._kb_repo.find_by_id(article_id)
        if article is None:
            raise ArticleNotFoundError(f"KB article '{article_id}' not found.")
        return article
