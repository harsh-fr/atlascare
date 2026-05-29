import logging
from typing import Any

from repositories.kb_repository import KbRepository

logger = logging.getLogger(__name__)


class KbError(Exception):
    pass

class ArticleNotFoundError(KbError):
    pass


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
        # match at least one tag, then rank them by match score below — instead
        # of scanning every article. (The result set is identical.)
        candidates = self._kb_repo.find_by_tags(tags)

        scored: list[tuple[int, str, dict[str, Any]]] = []
        for article in candidates:
            article_tags = {t.lower().strip() for t in article.get("tags", [])}
            match_score  = len(query_tags & article_tags)
            if match_score > 0:
                scored.append((match_score, article.get("last_updated", ""), article))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        results = []
        for score, _, article in scored[:max_results]:
            enriched = dict(article)
            enriched["match_score"] = score
            results.append(enriched)

        logger.info(
            "KbTool.search | tags=%s | matched=%d | returned=%d",
            tags, len(scored), len(results),
        )
        return results

    async def get_article(self, article_id: str) -> dict[str, Any]:
        logger.debug("KbTool.get_article | article_id=%s", article_id)
        article = self._kb_repo.find_by_id(article_id)
        if article is None:
            raise ArticleNotFoundError(f"KB article '{article_id}' not found.")
        return article
