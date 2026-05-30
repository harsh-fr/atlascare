"""
data/derive_support_files.py
============================
Derive the support files AtlasCare needs at runtime FROM the four canonical
data files supplied by the operator/evaluator. The canonical files are treated
as READ-ONLY inputs and are never modified:

  Inputs  (strict schema — see example_schema/schema_*.json):
    crm_cases.json       customers[] + cases[]
    orders.json          orders[]
    kb_articles.json     articles[]
    payment_config.json  refund limit / methods / sla / behaviour

  Derived outputs (generated here):
    users.json            one auth user per CRM customer
    sessions.json         one session_id -> customer_id mapping per customer
    refunds.json          runtime ledger — created empty if missing, NEVER overwritten
    order_audit_log.json  runtime ledger — created empty if missing, NEVER overwritten

`business_rules.json` is intentionally NOT generated: no code reads it, and the
real policy is sourced from payment_config.json + agent.guardrails + the agent
prompt (see memory: policy-data-sourcing).

Design
------
- Single source of truth: every derived record is a pure function of the
  canonical inputs, so a swapped-in data folder produces matching support files.
- Idempotent: `ensure_support_files()` only fills in MISSING files (use
  force=True to regenerate users/sessions). refunds.json is never clobbered
  because it accumulates runtime state.
- Defensive: missing or malformed canonical data is reported, not fatal — the
  app still boots (with empty support files) so the failure mode is visible
  rather than a hard crash.
- Path resolution mirrors the repositories exactly (same env vars), so the
  deriver always writes where the repos read.

Usage
-----
  python -m data.derive_support_files            # fill missing support files
  python -m data.derive_support_files --force    # regenerate users + sessions
  python -m data.derive_support_files --data-dir ./data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derivation constants
# ---------------------------------------------------------------------------
# All derived users share this default password (SHA-256, unsalted — matches
# services/auth_service._hash). Demo/eval system; override per-user via the
# /auth/register flow at runtime.
_DEFAULT_PASSWORD = os.getenv("DERIVED_DEFAULT_PASSWORD", "password")
_FIXED_CREATED_AT = "2025-01-01T00:00:00Z"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _HERE
_SCHEMA_DIR = os.path.join(_HERE, "..", "example_schema")

# env var -> default filename, mirroring each repository's path resolution.
_CANONICAL_PATHS = {
    "crm_cases":      ("CRM_DATA_PATH",       "crm_cases.json"),
    "orders":         ("ORDERS_DATA_PATH",    "orders.json"),
    "kb_articles":    ("KB_DATA_PATH",        "kb_articles.json"),
    "payment_config": ("PAYMENT_CONFIG_PATH", "payment_config.json"),
}
_DERIVED_PATHS = {
    "users":    ("USERS_DATA_PATH",    "users.json"),
    "sessions": ("SESSIONS_DATA_PATH", "sessions.json"),
    # Category vocab is read from kb_articles.applies_to; product->category is a
    # deterministic keyword cross-check; category->policies inverts applies_to.
    "product_categories": ("PRODUCT_CATEGORIES_PATH", "product_categories.json"),
    "category_policies":  ("CATEGORY_POLICIES_PATH",  "category_policies.json"),
}
# Runtime append-only ledgers: created empty when missing, NEVER overwritten.
# (env var, default filename, JSON root key)
_RUNTIME_LEDGERS = {
    "refunds":         ("REFUNDS_DATA_PATH", "refunds.json",         "refunds"),
    "order_audit_log": ("AUDIT_LOG_PATH",    "order_audit_log.json", "events"),
}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _resolve_path(env_var: str, filename: str, data_dir: str) -> str:
    """Resolve a file path the same way the repositories do: explicit env var
    wins, else <data_dir>/<filename>."""
    return os.path.abspath(os.getenv(env_var) or os.path.join(data_dir, filename))


def _load_json(path: str) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not read '%s': %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Validation — dependency-free checker for the draft-07 subset our schemas use
# (required, type, enum, pattern, const, minimum, minItems, properties, items).
# ---------------------------------------------------------------------------
_MAX_ERRORS = 25

# JSON Schema type name -> python type predicate. bool is excluded from the
# numeric types because Python treats bool as an int.
_TYPE_PREDICATES: dict[str, Any] = {
    "object":  lambda v: isinstance(v, dict),
    "array":   lambda v: isinstance(v, list),
    "string":  lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null":    lambda v: v is None,
}


def _type_ok(value: Any, type_spec: Any) -> bool:
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    return any(_TYPE_PREDICATES.get(t, lambda _v: True)(value) for t in types)


def _validate(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    if len(errors) >= _MAX_ERRORS:
        return
    if "type" in schema and not _type_ok(value, schema["type"]):
        errors.append(f"{path or '<root>'}: expected type {schema['type']}, got {type(value).__name__}")
        return  # further checks assume the right type

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path or '<root>'}: must equal {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path or '<root>'}: {value!r} not in allowed {schema['enum']}")
    if isinstance(value, str) and "pattern" in schema:
        if not re.search(schema["pattern"], value):
            errors.append(f"{path or '<root>'}: {value!r} does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path or '<root>'}: {value} below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path or '<root>'}: {value} above maximum {schema['maximum']}")

    if isinstance(value, dict):
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"{path or '<root>'}: missing required field '{field}'")
        props = schema.get("properties", {})
        for name, subschema in props.items():
            if name in value and isinstance(subschema, dict):
                _validate(value[name], subschema, f"{path}/{name}" if path else name, errors)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path or '<root>'}: has {len(value)} items, fewer than minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]", errors)


def validate_against_schema(key: str, data: Any) -> list[str]:
    """Return a list of human-readable schema violations for one canonical file.
    Empty list = valid. Never raises. Capped at _MAX_ERRORS entries."""
    schema = _load_json(os.path.join(_SCHEMA_DIR, f"schema_{key}.json"))
    if not isinstance(schema, dict):
        return []
    # Strip our annotation-only keys that aren't part of the JSON Schema vocab.
    schema = {k: v for k, v in schema.items() if not k.startswith("_")}
    errors: list[str] = []
    _validate(data, schema, "", errors)
    return errors


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def _username_for(customer: dict[str, Any]) -> str | None:
    """Derive a login username from a customer record.
    Prefers the email local-part first token (priya.sharma@... -> 'priya'),
    falls back to the name, then the customer_id. Returns None if nothing usable."""
    email = (customer.get("email") or "").strip().lower()
    if "@" in email:
        local = email.split("@", 1)[0]
        first = local.split(".")[0].strip()
        candidate = first or local
        if candidate:
            return candidate
    name = (customer.get("name") or "").strip().lower()
    if name:
        slug = re.sub(r"[^a-z0-9]+", "", name.split()[0])
        if slug:
            return slug
    cid = (customer.get("customer_id") or "").strip().lower().replace("-", "")
    return cid or None


def _user_id_for(customer_id: str, index: int) -> str:
    """CUST-001 -> USER-001; otherwise fall back to a stable sequential id."""
    m = re.search(r"(\d{3,})$", customer_id or "")
    return f"USER-{m.group(1)}" if m else f"USER-{index:03d}"


def _session_id_for(customer_id: str) -> str:
    """CUST-001 -> 'sess-cust001' (resolvable by SessionStore's explicit map AND
    its embedded-pattern extractor)."""
    return "sess-" + (customer_id or "").lower().replace("-", "")


def derive_users(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One auth user per customer. Deterministic; de-duplicates usernames."""
    users: list[dict[str, Any]] = []
    seen: set[str] = set()
    pw_hash = _hash_password(_DEFAULT_PASSWORD)
    for index, customer in enumerate(customers, start=1):
        customer_id = (customer.get("customer_id") or "").strip()
        if not customer_id:
            logger.warning("Skipping customer with no customer_id: %r", customer)
            continue
        username = _username_for(customer)
        if not username:
            logger.warning("Could not derive username for %s — skipping", customer_id)
            continue
        if username in seen:
            # Collision — disambiguate with the customer number.
            suffix = customer_id.lower().replace("-", "").replace("cust", "")
            username = f"{username}{suffix}"
        seen.add(username)
        users.append({
            "user_id":       _user_id_for(customer_id, index),
            "username":      username,
            "email":         (customer.get("email") or "").strip(),
            "password_hash": pw_hash,
            "customer_id":   customer_id,
            "created_at":    _FIXED_CREATED_AT,
        })
    return users


def derive_sessions(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One stable session_id -> customer_id mapping per customer."""
    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for customer in customers:
        customer_id = (customer.get("customer_id") or "").strip()
        if not customer_id or customer_id in seen:
            continue
        seen.add(customer_id)
        sessions.append({
            "session_id":  _session_id_for(customer_id),
            "customer_id": customer_id,
        })
    return sessions


# ---------------------------------------------------------------------------
# Policy category mapping (deterministic — no LLM, no network)
# ---------------------------------------------------------------------------
# The category VOCABULARY is read from kb_articles.applies_to (never hardcoded
# here). Each product is classified by a deterministic cross-check against a broad
# keyword "bucket" below: scan the product's id + name, the longest matching
# keyword wins, ties break by the order categories appear in the vocab. Unmatched
# products fall back to a configurable default and are reported so the bucket can
# be extended. The bucket is classifier knowledge, NOT policy — policy text stays
# in kb_articles.json / payment_config.json.
_CATEGORY_KEYWORD_BUCKET: dict[str, list[str]] = {
    "electronics": [
        "laptop", "macbook", "notebook", "ultrabook", "desktop", "computer",
        "phone", "smartphone", "galaxy", "iphone", "pixel", "oneplus", "redmi",
        "tablet", "ipad", "tab", "kindle", "e-reader",
        "tv", "television", "bravia", "oled", "qled", "led tv", "monitor", "display",
        "headphone", "earphone", "earbud", "airpod", "airpods", "speaker", "soundbar",
        "smartwatch", "apple watch", "fitbit", "fitness band", "smart band",
        "camera", "gopro", "dslr", "lens", "drone",
        "usb", "hub", "dock", "charger", "adapter", "powerbank", "power bank", "cable",
        "router", "modem", "console", "playstation", "xbox", "ssd", "pendrive",
        "keyboard", "mouse", "webcam", "printer", "projector", "soundbox",
    ],
    "apparel": [
        "kurta", "kurti", "saree", "sari", "dupatta", "palazzo", "stole", "lehenga",
        "salwar", "shirt", "t-shirt", "tshirt", "tee", "top", "blouse", "jeans",
        "trouser", "pant", "pants", "shorts", "dress", "skirt", "jacket", "coat",
        "sweater", "hoodie", "ethnic", "saree blouse", "leggings",
        "shoe", "sneaker", "sandal", "jutti", "juttis", "heel", "boot", "footwear",
        "slipper", "loafer", "flip flop",
        "anklet", "earring", "necklace", "bracelet", "jewellery", "jewelry", "ring",
        "bangle", "watch", "analog watch", "scarf", "belt", "handbag", "wallet",
        "cap", "hat", "sunglasses", "tie",
        "perfume", "fragrance", "cologne", "deodorant", "eau de",
    ],
    "home_goods": [
        "ac", "air conditioner", "conditioner", "refrigerator", "fridge", "freezer",
        "washing machine", "washer", "dryer", "dishwasher", "microwave", "oven",
        "air purifier", "purifier", "fan", "ceiling fan", "cooler", "heater", "geyser",
        "iron", "steam iron", "vacuum", "stabilizer", "voltage", "inverter",
        "chair", "table", "desk", "sofa", "bed", "mattress", "shelf", "rack",
        "organizer", "wardrobe", "cabinet",
        "cover", "stand", "mount", "laundry", "mesh bag", "curtain", "cushion",
        "bedsheet", "pillow", "blanket", "towel", "doormat", "carpet", "rug",
        "cookware", "utensil", "pan", "pot", "kettle", "mixer", "grinder", "blender",
        "toaster", "furniture", "lamp", "bulb", "tube light",
    ],
}

_DEFAULT_FALLBACK_CATEGORY = "home_goods"


def _categories_from_kb(articles: list[dict[str, Any]]) -> list[str]:
    """The category vocabulary, derived purely from kb_articles.applies_to.
    Sorted + de-duplicated so the order is stable across runs."""
    cats: set[str] = set()
    for a in articles:
        for c in a.get("applies_to", []) or []:
            if isinstance(c, str) and c.strip():
                cats.add(c.strip())
    return sorted(cats)


def _fallback_category(categories: list[str]) -> str | None:
    """Category for products the keyword bucket doesn't match. Configurable via
    PRODUCT_FALLBACK_CATEGORY; else the default if present in the vocab; else the
    first category in the vocab; else None (empty vocab)."""
    env = (os.getenv("PRODUCT_FALLBACK_CATEGORY") or "").strip()
    if env in categories:
        return env
    if _DEFAULT_FALLBACK_CATEGORY in categories:
        return _DEFAULT_FALLBACK_CATEGORY
    return categories[0] if categories else None


def _normalise(text: str) -> tuple[set[str], str]:
    """Return (token_set_with_singularised_forms, normalised_joined_text) for a
    product string, so keyword matching is whole-word and plural-tolerant (so
    'Juttis' matches 'jutti', 'Sneakers' matches 'sneaker')."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    token_set: set[str] = set()
    for t in tokens:
        token_set.add(t)
        if len(t) > 3 and t.endswith("s"):
            token_set.add(t[:-1])
    return token_set, " ".join(tokens)


def _keyword_matches(keyword: str, tokens: set[str], joined: str) -> bool:
    if " " in keyword:                       # multi-word phrase
        return keyword in joined
    base = keyword[:-1] if len(keyword) > 3 and keyword.endswith("s") else keyword
    return keyword in tokens or base in tokens


def classify_product(text: str, categories: list[str]) -> tuple[str | None, str]:
    """Deterministically classify one product (id + name) into a category from the
    vocab via the keyword bucket. Returns (category_or_None, winning_keyword).
    Longest matching keyword wins; only categories in the live vocab are eligible."""
    tokens, joined = _normalise(text)
    best_cat: str | None = None
    best_kw = ""
    for category in categories:                      # vocab order = tie-break
        for keyword in _CATEGORY_KEYWORD_BUCKET.get(category, []):
            if len(keyword) > len(best_kw) and _keyword_matches(keyword, tokens, joined):
                best_cat, best_kw = category, keyword
    return best_cat, best_kw


def classify_products(
    products: dict[str, str],
    categories: list[str],
    existing: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Classify a {product_id: name} map into {product_id: {name, category}}.
    Re-uses any `existing` entry whose category is still in the vocab (cache), so
    only genuinely new/changed products are re-classified. Returns (mapping,
    unmatched_ids) where unmatched fell back to the default category."""
    existing = existing or {}
    mapping: dict[str, dict[str, str]] = {}
    unmatched: list[str] = []
    fallback = _fallback_category(categories)
    for pid, name in sorted(products.items()):
        prev = existing.get(pid)
        if isinstance(prev, dict) and prev.get("category") in categories:
            mapping[pid] = {"name": name or prev.get("name", ""), "category": prev["category"]}
            continue
        category, _kw = classify_product(f"{pid} {name}", categories)
        if category is None:
            category = fallback
            if category is not None:
                unmatched.append(pid)
        if category is not None:
            mapping[pid] = {"name": name, "category": category}
    return mapping, unmatched


def derive_category_policies(
    articles: list[dict[str, Any]],
    categories: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Invert kb_articles.applies_to into {category: [{article_id, title, tags}]}.
    Pure derivation — the policy text itself stays in kb_articles.json."""
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in categories}
    for a in articles:
        entry = {
            "article_id": a.get("article_id"),
            "title":      a.get("title"),
            "tags":       a.get("tags", []),
        }
        for c in a.get("applies_to", []) or []:
            if c in out:
                out[c].append(entry)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def ensure_support_files(
    data_dir: str | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Generate missing derived support files from the canonical inputs.

    Parameters
    ----------
    data_dir : base directory for default paths (used only when the per-file
               env var is unset). Defaults to the data/ package directory.
    force    : regenerate users.json and sessions.json even if they exist.
               refunds.json is never overwritten (runtime ledger).

    Returns
    -------
    A report dict: {"generated": [...], "skipped": [...], "errors": [...],
                    "warnings": [...]}.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    report: dict[str, list[str]] = {
        "generated": [], "skipped": [], "errors": [], "warnings": [],
    }

    # --- Load + validate canonical inputs -------------------------------
    crm_path = _resolve_path(*_CANONICAL_PATHS["crm_cases"], data_dir)
    crm = _load_json(crm_path)
    if crm is None:
        report["errors"].append(
            f"crm_cases not found/readable at '{crm_path}' — cannot derive users/sessions."
        )
        customers = []
    else:
        for violation in validate_against_schema("crm_cases", crm):
            report["warnings"].append(f"crm_cases schema: {violation}")
        customers = crm.get("customers", []) if isinstance(crm, dict) else []
        if not customers:
            report["warnings"].append("crm_cases has no customers — users/sessions will be empty.")

    # Validate the other canonical files too (report-only; not required to derive).
    # Retain the payloads — orders + kb_articles feed the category derivations below.
    canonical: dict[str, Any] = {}
    for key in ("orders", "kb_articles", "payment_config"):
        path = _resolve_path(*_CANONICAL_PATHS[key], data_dir)
        payload = _load_json(path)
        canonical[key] = payload
        if payload is None:
            report["warnings"].append(f"{key} not found/readable at '{path}'.")
            continue
        for violation in validate_against_schema(key, payload):
            report["warnings"].append(f"{key} schema: {violation}")

    # --- users.json ------------------------------------------------------
    users_path = _resolve_path(*_DERIVED_PATHS["users"], data_dir)
    if force or not os.path.exists(users_path):
        users = derive_users(customers)
        _write_json(users_path, {"users": users})
        report["generated"].append(f"users.json ({len(users)} users) -> {users_path}")
    else:
        report["skipped"].append(f"users.json exists -> {users_path}")

    # --- sessions.json ---------------------------------------------------
    sessions_path = _resolve_path(*_DERIVED_PATHS["sessions"], data_dir)
    if force or not os.path.exists(sessions_path):
        sessions = derive_sessions(customers)
        _write_json(sessions_path, {"sessions": sessions})
        report["generated"].append(f"sessions.json ({len(sessions)} sessions) -> {sessions_path}")
    else:
        report["skipped"].append(f"sessions.json exists -> {sessions_path}")

    # --- runtime ledgers (create empty only if missing — never clobber) --
    for env_var, filename, root_key in _RUNTIME_LEDGERS.values():
        path = _resolve_path(env_var, filename, data_dir)
        if not os.path.exists(path):
            _write_json(path, {root_key: []})
            report["generated"].append(f"{filename} (empty) -> {path}")
        else:
            report["skipped"].append(f"{filename} exists (runtime data preserved) -> {path}")

    # --- policy category derivations (deterministic, from kb + orders) ---
    kb = canonical.get("kb_articles")
    articles = kb.get("articles", []) if isinstance(kb, dict) else []
    categories = _categories_from_kb(articles)
    if not categories:
        report["warnings"].append(
            "kb_articles has no applies_to categories — skipping category derivations."
        )
    else:
        # category_policies.json — pure inversion of applies_to; regenerate each run.
        cat_pol_path = _resolve_path(*_DERIVED_PATHS["category_policies"], data_dir)
        _write_json(cat_pol_path, derive_category_policies(articles, categories))
        report["generated"].append(
            f"category_policies.json ({len(categories)} categories: "
            f"{', '.join(categories)}) -> {cat_pol_path}"
        )

        # product_categories.json — deterministic classification, cached/incremental.
        orders_payload = canonical.get("orders")
        orders = orders_payload.get("orders", []) if isinstance(orders_payload, dict) else []
        products: dict[str, str] = {}
        for o in orders:
            for it in o.get("items", []) if isinstance(o, dict) else []:
                pid = it.get("product_id")
                if pid:
                    products.setdefault(pid, it.get("name", ""))
        pc_path = _resolve_path(*_DERIVED_PATHS["product_categories"], data_dir)
        existing = None
        if not force:
            prev = _load_json(pc_path)
            if isinstance(prev, dict):
                existing = prev.get("products")
        mapping, unmatched = classify_products(products, categories, existing)
        _write_json(pc_path, {"categories": categories, "products": mapping})
        report["generated"].append(
            f"product_categories.json ({len(mapping)} products, "
            f"{len(unmatched)} fallback) -> {pc_path}"
        )
        for pid in unmatched:
            report["warnings"].append(
                f"product {pid} ({products.get(pid, '')!r}) matched no keyword — "
                f"assigned fallback category; extend _CATEGORY_KEYWORD_BUCKET."
            )

    return report


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _print_report(report: dict[str, list[str]]) -> None:
    for line in report["generated"]:
        print(f"  [GEN]  {line}")
    for line in report["skipped"]:
        print(f"  [SKIP] {line}")
    for line in report["warnings"]:
        print(f"  [WARN] {line}")
    for line in report["errors"]:
        print(f"  [ERR]  {line}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Derive AtlasCare support files from the 4 canonical files.")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="Data directory (default: ./data)")
    parser.add_argument("--force", action="store_true", help="Regenerate users.json and sessions.json even if present")
    args = parser.parse_args()

    print("AtlasCare Support-File Deriver")
    print("=" * 40)
    rpt = ensure_support_files(args.data_dir, force=args.force)
    _print_report(rpt)
    print("\nDone." if not rpt["errors"] else "\nCompleted with errors.")
