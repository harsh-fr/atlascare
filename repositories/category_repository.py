"""
repositories/category_repository.py
===================================
Read-access to the two DERIVED policy-category files produced by
`data/derive_support_files.py`:

  product_categories.json  {categories: [...], products: {PID: {name, category}}}
  category_policies.json   {category: [{article_id, title, tags}, ...]}

Both are pure functions of the canonical inputs (kb_articles.applies_to defines
the vocab; a deterministic keyword cross-check assigns each product a category;
applies_to is inverted into category->policies). Nothing here is hardcoded policy.

Loading is LAZY and DEFENSIVE: the files may not exist yet at import time (they
are generated in the app's startup bootstrap, and may be absent in a bare test
env). When absent, every accessor returns empty so callers degrade gracefully to
tag-only KB retrieval — never an error.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PRODUCTS = os.path.join(_HERE, "..", "data", "product_categories.json")
_DEFAULT_POLICIES = os.path.join(_HERE, "..", "data", "category_policies.json")


class CategoryRepository:
    def __init__(self, products_path: str | None = None, policies_path: str | None = None) -> None:
        # Explicit overrides win; otherwise the path is resolved from the env var at
        # LOAD time (not here), so an env set after construction — e.g. a test's
        # monkeypatch, or the startup bootstrap — is honoured on (re)load.
        self._products_override = products_path
        self._policies_override = policies_path
        self._lock = threading.Lock()
        self._loaded = False
        self._product_cat: dict[str, str] = {}
        self._policies: dict[str, list[dict[str, Any]]] = {}
        self._categories: list[str] = []

    @property
    def _products_path(self) -> str:
        return self._products_override or os.getenv("PRODUCT_CATEGORIES_PATH", _DEFAULT_PRODUCTS)

    @property
    def _policies_path(self) -> str:
        return self._policies_override or os.getenv("CATEGORY_POLICIES_PATH", _DEFAULT_POLICIES)

    # -- loading ----------------------------------------------------------
    @staticmethod
    def _read(path: str) -> Any | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            logger.warning(
                "Category file not found: %s — category-aware policy lookup disabled "
                "until it is derived.", path,
            )
            return None
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Could not read category file %s: %s", path, exc)
            return None

    def _ensure(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            prods = self._read(self._products_path)
            if isinstance(prods, dict):
                self._categories = [c for c in prods.get("categories", []) if isinstance(c, str)]
                for pid, info in (prods.get("products") or {}).items():
                    if isinstance(info, dict) and info.get("category"):
                        self._product_cat[pid] = info["category"]
            pol = self._read(self._policies_path)
            if isinstance(pol, dict):
                self._policies = {k: v for k, v in pol.items() if isinstance(v, list)}
            self._loaded = True

    def reload(self) -> None:
        """Drop the cache (e.g. after the startup bootstrap regenerates the files)."""
        with self._lock:
            self._loaded = False
            self._product_cat = {}
            self._policies = {}
            self._categories = []
        self._ensure()

    # -- accessors --------------------------------------------------------
    @property
    def categories(self) -> list[str]:
        self._ensure()
        return list(self._categories)

    def category_for_product(self, product_id: str) -> str | None:
        self._ensure()
        return self._product_cat.get(product_id)

    def categories_for_order(self, order: dict[str, Any]) -> list[str]:
        """Distinct categories of an order's line items, preserving first-seen order."""
        self._ensure()
        cats: list[str] = []
        for item in (order.get("items") or []):
            cat = self._product_cat.get(item.get("product_id"))
            if cat and cat not in cats:
                cats.append(cat)
        return cats

    def policies_for_category(self, category: str) -> list[dict[str, Any]]:
        self._ensure()
        return list(self._policies.get(category, []))

    def tags_for_categories(self, categories: list[str]) -> set[str]:
        """Union of policy tags applicable to the given categories."""
        self._ensure()
        tags: set[str] = set()
        for cat in categories:
            for entry in self._policies.get(cat, []):
                tags.update(entry.get("tags", []))
        return tags
