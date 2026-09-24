from __future__ import annotations

import requests
import time
import random
import json
import os
import re
import sqlite3
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable system proxy — macOS proxy (e.g. Clash on :7890) can block NCBI/PMC API calls.
for _var in ("NO_PROXY", "no_proxy"):
    if not os.environ.get(_var):
        os.environ[_var] = "*"
# Force offline mode for HuggingFace — models are already cached locally.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    if not os.environ.get(_var):
        os.environ[_var] = "1"

# =============================================================================
# Configuration
# =============================================================================

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
BMJ_BEST_PRACTICE_BASE_URL = "https://bestpractice.bmj.com"

# Rate limits (requests per second)
PUBMED_RATE_LIMIT = 0.11  # ~9 req/sec with NCBI API key (max 10/s)
EUROPE_PMC_RATE_LIMIT = 0.1  # ~10 requests/sec
SEMANTIC_SCHOLAR_RATE_LIMIT = 0.02  # generous, actual limit is higher
BMJ_BEST_PRACTICE_RATE_LIMIT = 0.5  # be polite to BMJ web servers
PUBMED_TIMEOUT = 30  # seconds per request (high timeout for unstable network)
BMJ_BEST_PRACTICE_TIMEOUT = 15

# Determine project root (scripts/ -> parent -> project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DB_PATH = os.environ.get(
    "RETRIEVAL_CACHE_DB_PATH",
    os.path.join(_PROJECT_ROOT, "cache", "retrieval_cache.db")
)
BMJ_BEST_PRACTICE_CACHE_TTL_HOURS = int(os.environ.get("BMJ_BEST_PRACTICE_CACHE_TTL_HOURS", "168"))

# LLM Configuration for retrieval enhancement
LLM_PROVIDER = (os.environ.get("RETRIEVAL_LLM_PROVIDER", "") or os.environ.get("LLM_PROVIDER", "deepseek-flash")).lower()
LLM_API_KEY = os.environ.get("RETRIEVAL_LLM_API_KEY", "") or os.environ.get("DEEP_SEEK_API", "") or os.environ.get("DEEPSEEK_API_KEY", "") or ""
LLM_MODEL = os.environ.get("RETRIEVAL_LLM_MODEL", "") or os.environ.get("LLM_MODEL", "") or "deepseek-flash"
LLM_API_BASE = os.environ.get("RETRIEVAL_LLM_API_BASE", "") or os.environ.get("LLM_API_BASE", "") or "https://api.deepseek.com/v1"
ENABLE_LLM_SYNTHESIS = os.environ.get("ENABLE_LLM_SYNTHESIS", "false").lower() == "true"
ENABLE_LLM_KEYWORDS = os.environ.get("ENABLE_LLM_KEYWORDS", "true").lower() == "true"
LLM_RATE_LIMIT = 0.5  # seconds between LLM calls

# PubMed / NCBI configuration
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "") or os.environ.get("EUTILS_API_KEY", "")


# =============================================================================
# Structured Retrieval Configuration
# =============================================================================

# Base weights for strategy source in ranking
STRATEGY_WEIGHTS = {
    "精准匹配": 10,
    "治疗证据": 5,
    "鉴别预后": 3,
    "风险因素导向": 4,  # NEW: risk-factor-guided strategy
    "legacy": 8,
}

# Local model optimized max results per strategy
# For local models (e.g., DNZS/大内助手) needing knowledge supplementation:
# - Increase max_results to improve recall (more candidates for filtering)
# - Rely on post-retrieval filtering (perplexity filter + hard cap) to control noise
STRATEGY_MAX_RESULTS_LOCAL = {
    "精准匹配": 5,  # Increased from 3 to improve recall
    "治疗证据": 3,  # Increased from 2
    "鉴别预后": 3,  # Increased from 2
    "风险因素导向": 5,  # NEW: risk-factor-guided strategy
    "legacy": 5,    # Increased from 3
}

# Hard cap on total sources for local models
# For local models needing knowledge supplementation:
# - Keep cap at 5 to prevent context overload (per deep analysis: k=1-3 optimal for strong models)
# - But allow more candidates through filtering (via increased STRATEGY_MAX_RESULTS_LOCAL)
MAX_TOTAL_SOURCES_LOCAL = 5


# =============================================================================
# StatPearls / NCBI Bookshelf Client
# =============================================================================

@dataclass
class StatPearlsArticle:
    """Structured content from a StatPearls article on NCBI Bookshelf."""
    nbk_id: str
    title: str
    url: str
    introduction: str = ""
    differential_diagnosis: str = ""
    history_physical: str = ""
    evaluation: str = ""
    etiology: str = ""
    epidemiology: str = ""

    def to_text(self, max_chars: int = 1200) -> str:
        """Format article as concise evidence text for prompts."""
        parts = [f"StatPearls: {self.title}", f"URL: {self.url}"]
        if self.introduction:
            parts.append(f"Introduction: {self.introduction}")
        if self.history_physical:
            parts.append(f"History & Physical: {self.history_physical}")
        if self.evaluation:
            parts.append(f"Evaluation: {self.evaluation}")
        if self.differential_diagnosis:
            parts.append(f"Differential Diagnosis: {self.differential_diagnosis}")
        text = "\n".join(parts)
        return text[:max_chars]


class StatPearlsClient:
    """
    Lightweight client for StatPearls articles on NCBI Bookshelf.

    - Uses NCBI Bookshelf E-utilities (esearch/esummary) to find StatPearls
      articles by title.
    - Fetches article HTML and extracts key sections.
    - Caches results to a local SQLite DB.
    """

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or os.path.join(
            _PROJECT_ROOT, "cache", "statpearls_cache.db"
        )
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._init_cache()

    def _init_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS statpearls_cache (
                    query TEXT PRIMARY KEY,
                    nbk_id TEXT,
                    title TEXT,
                    url TEXT,
                    introduction TEXT,
                    differential_diagnosis TEXT,
                    history_physical TEXT,
                    evaluation TEXT,
                    etiology TEXT,
                    epidemiology TEXT,
                    fetched_at REAL
                )
                """
            )
            conn.commit()

    def _rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < PUBMED_RATE_LIMIT:
                time.sleep(PUBMED_RATE_LIMIT - elapsed)
            self._last_request = time.time()

    def _get_cached(self, query: str) -> Optional[StatPearlsArticle]:
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.cursor()
                ttl_seconds = 168 * 3600  # 1 week
                cursor.execute(
                    """
                    SELECT nbk_id, title, url, introduction, differential_diagnosis,
                           history_physical, evaluation, etiology, epidemiology
                    FROM statpearls_cache
                    WHERE query = ? AND fetched_at > ?
                    """,
                    (query.lower().strip(), time.time() - ttl_seconds),
                )
                row = cursor.fetchone()
                if row:
                    return StatPearlsArticle(
                        nbk_id=row[0] or "",
                        title=row[1] or "",
                        url=row[2] or "",
                        introduction=row[3] or "",
                        differential_diagnosis=row[4] or "",
                        history_physical=row[5] or "",
                        evaluation=row[6] or "",
                        etiology=row[7] or "",
                        epidemiology=row[8] or "",
                    )
        except Exception as e:
            print(f"[StatPearls] Cache read error: {e}")
        return None

    def _set_cached(self, query: str, article: StatPearlsArticle):
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO statpearls_cache
                    (query, nbk_id, title, url, introduction, differential_diagnosis,
                     history_physical, evaluation, etiology, epidemiology, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query.lower().strip(),
                        article.nbk_id,
                        article.title,
                        article.url,
                        article.introduction,
                        article.differential_diagnosis,
                        article.history_physical,
                        article.evaluation,
                        article.etiology,
                        article.epidemiology,
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"[StatPearls] Cache write error: {e}")

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search StatPearls articles by title using NCBI Bookshelf.

        Tries an exact title match first; if that yields nothing, falls back
        to a broader StatPearls search so shorter core terms still find
        relevant articles.
        """
        if not query or len(query.strip()) < 3:
            return []

        def _do_search(term: str) -> List[Dict]:
            self._rate_limit()
            url = f"{PUBMED_BASE_URL}/esearch.fcgi"
            params = {
                "db": "books",
                "term": term,
                "retmax": max_results,
                "retmode": "json",
            }
            if NCBI_API_KEY:
                params["api_key"] = NCBI_API_KEY
            try:
                resp = requests.get(url, params=params, timeout=PUBMED_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                idlist = data.get("esearchresult", {}).get("idlist", [])
            except Exception as e:
                print(f"[StatPearls] Search failed for '{query}' (term={term}): {e}")
                return []

            if not idlist:
                return []

            summaries = self._fetch_summaries(idlist)
            results = []
            for item in summaries:
                nbk_id = item.get("accessionid", "")
                if nbk_id:
                    results.append({
                        "uid": item.get("uid", ""),
                        "nbk_id": nbk_id,
                        "title": item.get("title", ""),
                        "url": f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/",
                    })
            return results

        clean = query.strip().replace('"', '')

        # Strategy 1: exact title match within StatPearls
        results = _do_search(f'"{clean}"[Title] AND StatPearls[Book]')
        if results:
            return results

        # Strategy 2: broad StatPearls keyword search (core term anywhere)
        results = _do_search(f'{clean}[Title/Abstract] AND StatPearls[Book]')
        if results:
            return results

        print(f"[StatPearls] No search results for '{query}'")
        return []

    def _fetch_summaries(self, uids: List[str]) -> List[Dict]:
        """Fetch book summaries by UID."""
        if not uids:
            return []
        self._rate_limit()
        url = f"{PUBMED_BASE_URL}/esummary.fcgi"
        params = {
            "db": "books",
            "id": ",".join(str(u) for u in uids),
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        try:
            resp = requests.get(url, params=params, timeout=PUBMED_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for uid in uids:
                summary = data.get("result", {}).get(str(uid), {})
                if summary:
                    results.append(summary)
            return results
        except Exception as e:
            print(f"[StatPearls] Summary fetch failed: {e}")
            return []

    def fetch_article(self, nbk_id: str, title: str = "", url: str = "") -> Optional[StatPearlsArticle]:
        """Fetch and parse a StatPearls article HTML page."""
        if not nbk_id:
            return None

        self._rate_limit()
        if not url:
            url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/"

        try:
            session = requests.Session()
            retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retry))
            resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"[StatPearls] Fetch failed for {nbk_id}: {e}")
            return None

        parsed = self._parse_article_html(resp.text)
        if not title:
            title_match = re.search(r'<title[^>]*>(.*?)</title>', resp.text, re.IGNORECASE | re.DOTALL)
            title = re.sub(r'\s*-\s*NCBI Bookshelf.*', '', title_match.group(1).strip()) if title_match else ""
            title = re.sub(r'<[^>]+>', '', title)

        return StatPearlsArticle(
            nbk_id=nbk_id,
            title=title or parsed.get("title", ""),
            url=url,
            introduction=parsed.get("introduction", ""),
            differential_diagnosis=parsed.get("differential_diagnosis", ""),
            history_physical=parsed.get("history_physical", ""),
            evaluation=parsed.get("evaluation", ""),
            etiology=parsed.get("etiology", ""),
            epidemiology=parsed.get("epidemiology", ""),
        )

    def _parse_article_html(self, html: str) -> Dict:
        """Extract key sections from a StatPearls HTML page."""
        result = {
            "title": "",
            "introduction": "",
            "differential_diagnosis": "",
            "history_physical": "",
            "evaluation": "",
            "etiology": "",
            "epidemiology": "",
        }

        # Try BeautifulSoup if available
        soup = None
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            pass

        text = soup.get_text("\n") if soup else re.sub(r'<[^>]+>', '\n', html)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        def extract_section(name: str) -> str:
            # Match section heading and take until next major heading
            pattern = rf'(?:^|\n)\s*{re.escape(name)}\s*\n(.*?)(?=(?:^|\n)\s*(?:Introduction|Etiology|Epidemiology|Pathophysiology|History and Physical|Physical Examination|Evaluation|Differential Diagnosis|Treatment|Management|Prognosis|Complications|Pearls and Pitfalls|Enhancing Healthcare Team Outcomes|References)\s*\n|\Z)'
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                raw = m.group(1).strip()
                # Remove reference markers like [44]
                raw = re.sub(r'\[\d+\]', '', raw)
                return raw[:1200]
            return ""

        result["introduction"] = extract_section("Introduction")
        result["etiology"] = extract_section("Etiology")
        result["epidemiology"] = extract_section("Epidemiology")
        result["history_physical"] = extract_section("History and Physical")
        if not result["history_physical"]:
            result["history_physical"] = extract_section("Physical Examination")
        result["evaluation"] = extract_section("Evaluation")
        result["differential_diagnosis"] = extract_section("Differential Diagnosis")

        return result

    def lookup(self, query: str) -> Optional[StatPearlsArticle]:
        """Search StatPearls and fetch the top article."""
        if not query or len(query.strip()) < 3:
            return None

        cached = self._get_cached(query)
        if cached:
            print(f"[StatPearls] Cache hit for '{query}' -> {cached.title}")
            return cached

        results = self.search(query, max_results=1)
        if not results:
            print(f"[StatPearls] No search results for '{query}'")
            return None

        top = results[0]
        print(f"[StatPearls] Searching '{query}' -> {top['nbk_id']} ({top['title']})")
        article = self.fetch_article(top["nbk_id"], title=top["title"], url=top["url"])
        if article:
            self._set_cached(query, article)
        return article

# Clinical Queries filters for PubMed (higher specificity for local models)
CLINICAL_QUERIES_FILTER = {
    "broad": "Diagnosis/Broad[filter]",
    "narrow": "Diagnosis/Narrow[filter]",
}

# Evidence level scoring for publication types
EVIDENCE_LEVEL_SCORES = {
    "P0_case": {
        "case report": 8, "case series": 8, "case study": 6,
        "clinical report": 4, "clinical observation": 3,
    },
    "P1_study": {
        "cohort": 5, "retrospective": 5, "prospective": 5,
        "comparative study": 4, "clinical trial": 6,
        "controlled trial": 5, "follow-up": 3,
    },
    "P2_review": {
        "review": -4, "systematic review": -4, "meta-analysis": -4,
        "guideline": -3, "consensus": -3, "practice guideline": -3,
    },
}

# Rare/differential diagnosis signals
RARE_SIGNALS = {
    "rare": 5, "unusual": 5, "atypical": 5, "unexpected": 4,
    "differential diagnosis": 6, "diagnostic challenge": 5,
    "mimic": 4, "masquerade": 4, "overlap": 3,
}

PENALTY_SIGNALS = {
    "review": -3, "systematic review": -4, "meta-analysis": -4,
    "guideline": -3, "consensus": -3, "epidemiology": -2,
    "prevalence": -2, "incidence": -2,
}


@dataclass
class BMJBestPracticeTopic:
    """Structured public-overview content from BMJ Best Practice."""
    topic_id: str
    title: str
    url: str
    summary: str = ""
    definition: str = ""
    key_diagnostic_factors: List[str] = field(default_factory=list)
    differentials: List[str] = field(default_factory=list)
    investigations: List[str] = field(default_factory=list)
    treatment_algorithm: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None

    def to_text(self, max_chars: int = 1500) -> str:
        """Format topic as concise evidence text for prompts."""
        parts = [f"BMJ Best Practice: {self.title}", f"URL: {self.url}"]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.definition:
            parts.append(f"Definition: {self.definition}")
        if self.key_diagnostic_factors:
            parts.append("Key diagnostic factors: " + "; ".join(self.key_diagnostic_factors))
        if self.differentials:
            parts.append("Differentials: " + "; ".join(self.differentials))
        if self.investigations:
            parts.append("Investigations: " + "; ".join(self.investigations))
        text = "\n".join(parts)
        return text[:max_chars]


class BMJBestPracticeClient:
    """
    Lightweight client for BMJ Best Practice public overview pages.

    - Uses BMJ's search endpoint to find topic IDs.
    - Parses freely accessible overview content (summary, definition,
      key diagnostic factors, differentials, investigations).
    - Falls back gracefully when content is paywalled or pages change.
    - Caches results to a local SQLite DB to avoid repeated scraping.
    """

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or os.path.join(
            _PROJECT_ROOT, "cache", "bmj_best_practice_cache.db"
        )
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._init_cache()

    def _init_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bmj_cache (
                    query TEXT PRIMARY KEY,
                    topic_id TEXT,
                    title TEXT,
                    url TEXT,
                    summary TEXT,
                    definition TEXT,
                    key_diagnostic_factors TEXT,
                    differentials TEXT,
                    investigations TEXT,
                    treatment_algorithm TEXT,
                    fetched_at REAL
                )
                """
            )
            conn.commit()

    def _rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < BMJ_BEST_PRACTICE_RATE_LIMIT:
                time.sleep(BMJ_BEST_PRACTICE_RATE_LIMIT - elapsed)
            self._last_request = time.time()

    def _get_cached(self, query: str) -> Optional[BMJBestPracticeTopic]:
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.cursor()
                ttl_seconds = BMJ_BEST_PRACTICE_CACHE_TTL_HOURS * 3600
                cursor.execute(
                    """
                    SELECT topic_id, title, url, summary, definition,
                           key_diagnostic_factors, differentials, investigations,
                           treatment_algorithm
                    FROM bmj_cache
                    WHERE query = ? AND fetched_at > ?
                    """,
                    (query.lower().strip(), time.time() - ttl_seconds),
                )
                row = cursor.fetchone()
                if row:
                    return BMJBestPracticeTopic(
                        topic_id=row[0] or "",
                        title=row[1] or "",
                        url=row[2] or "",
                        summary=row[3] or "",
                        definition=row[4] or "",
                        key_diagnostic_factors=_parse_json_list(row[5]),
                        differentials=_parse_json_list(row[6]),
                        investigations=_parse_json_list(row[7]),
                        treatment_algorithm=_parse_json_list(row[8]),
                    )
        except Exception as e:
            print(f"[BMJ] Cache read error: {e}")
        return None

    def _set_cached(self, query: str, topic: BMJBestPracticeTopic):
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bmj_cache
                    (query, topic_id, title, url, summary, definition,
                     key_diagnostic_factors, differentials, investigations,
                     treatment_algorithm, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query.lower().strip(),
                        topic.topic_id,
                        topic.title,
                        topic.url,
                        topic.summary,
                        topic.definition,
                        json.dumps(topic.key_diagnostic_factors, ensure_ascii=False),
                        json.dumps(topic.differentials, ensure_ascii=False),
                        json.dumps(topic.investigations, ensure_ascii=False),
                        json.dumps(topic.treatment_algorithm, ensure_ascii=False),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"[BMJ] Cache write error: {e}")

    def search_topics(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search BMJ Best Practice via DuckDuckGo and return candidate topic dicts."""
        if not query or len(query.strip()) < 3:
            return []

        self._rate_limit()
        from urllib.parse import quote_plus
        search_query = f"{query.strip()} bestpractice.bmj.com"
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            session = requests.Session()
            retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retry))
            resp = session.get(
                search_url, headers=headers,
                timeout=BMJ_BEST_PRACTICE_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[BMJ] DuckDuckGo search failed for '{query}': {e}")
            return []

        html = resp.text
        results = []
        # DuckDuckGo redirects through /l/?uddg=...
        link_pattern = re.compile(
            r'class="result__a" href="([^"]+)"',
            re.IGNORECASE,
        )
        seen = set()
        for m in link_pattern.finditer(html):
            href = m.group(1)
            # Extract actual URL from uddg parameter
            uddg_match = re.search(r'uddg=([^&]+)', href)
            if not uddg_match:
                continue
            try:
                from urllib.parse import unquote
                actual_url = unquote(uddg_match.group(1))
            except Exception:
                continue
            if not actual_url.startswith("https://bestpractice.bmj.com/topics/"):
                continue
            # Parse topic id from URL path (may have trailing /management-approach or query string)
            # URL format: https://bestpractice.bmj.com/topics/{lang}/{topic_id}[/{subpage}]
            from urllib.parse import urlparse
            parsed = urlparse(actual_url)
            path = parsed.path.rstrip('/')
            parts = path.split('/')
            # path starts with leading '' then 'topics', 'lang', 'id'
            if len(parts) < 4 or parts[1] != "topics":
                continue
            topic_id = parts[-1]
            lang = parts[2] if len(parts) >= 3 else "en-gb"
            if not topic_id.isdigit():
                continue
            key = topic_id
            if key in seen:
                continue
            seen.add(key)
            # Build canonical topic URL without subpage
            canonical_url = f"{BMJ_BEST_PRACTICE_BASE_URL}/topics/{lang}/{topic_id}"
            results.append({"topic_id": topic_id, "title": "", "url": canonical_url})
            if len(results) >= max_results:
                break

        return results

    def fetch_overview(self, topic_id: str, title: str = "", url: str = "") -> Optional[BMJBestPracticeTopic]:
        """Fetch and parse the public overview page of a BMJ topic."""
        if not topic_id:
            return None

        self._rate_limit()
        if not url:
            url = f"{BMJ_BEST_PRACTICE_BASE_URL}/topics/en-gb/{topic_id}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=BMJ_BEST_PRACTICE_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            print(f"[BMJ] Fetch overview failed for {topic_id}: {e}")
            return None

        html = resp.text
        # Use simple regex-based extraction. We intentionally avoid lxml/bs4
        # dependency; if available we can optionally use BeautifulSoup.
        try:
            parsed = self._parse_overview_html(html)
        except Exception as e:
            print(f"[BMJ] Parse overview failed for {topic_id}: {e}")
            parsed = {}

        if not title:
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            title = re.sub(r'\s*\|\s*BMJ Best Practice.*', '', title_match.group(1).strip()) if title_match else ""
            title = re.sub(r'<[^>]+>', '', title)

        return BMJBestPracticeTopic(
            topic_id=topic_id,
            title=title or parsed.get("title", ""),
            url=url,
            summary=parsed.get("summary", ""),
            definition=parsed.get("definition", ""),
            key_diagnostic_factors=parsed.get("key_diagnostic_factors", []),
            differentials=parsed.get("differentials", []),
            investigations=parsed.get("investigations", []),
            treatment_algorithm=parsed.get("treatment_algorithm", []),
        )

    def _parse_overview_html(self, html: str) -> Dict:
        """Extract public overview fields from BMJ HTML."""
        result = {
            "title": "",
            "summary": "",
            "definition": "",
            "key_diagnostic_factors": [],
            "differentials": [],
            "investigations": [],
            "treatment_algorithm": [],
        }

        # Try BeautifulSoup if available for more robust parsing
        soup = None
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            pass

        text = soup.get_text("\n") if soup else re.sub(r'<[^>]+>', '\n', html)
        # Collapse whitespace and remove unicode arrows/collapsible icons
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[-]', '', text)

        def _extract_section(name: str, next_name: Optional[str] = None) -> str:
            """Extract text between two section headings (case-insensitive)."""
            pattern = rf'(?:^|\n)\s*{re.escape(name)}\s*\n(.*?)(?=(?:^|\n)\s*(?:{re.escape(next_name) if next_name else ""}|Summary|Definition|History and exam|Diagnosis|Investigations|Differentials|Treatment algorithm|Management|Follow up|Resources|Guidelines|References|Contributors)\s*\n|\Z)'
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                raw = m.group(1).strip()
                # Truncate if extremely long
                return raw[:1200]
            return ""

        # Summary is usually right after the title near the top of the page.
        # The public overview repeats the section nav; locate the second "Summary"
        # heading and take the paragraph that follows it.
        summary = ""
        summary_match = re.search(
            r'(?:^|\n)\s*Summary\s*\n.*?(?:^|\n)\s*Summary\s*\n\s*(.*?)(?=(?:^|\n)\s*(?:Definition|Theory|History and exam|Diagnosis|Investigations|Differentials|Treatment algorithm|Management|Follow up|Resources|Guidelines|References|Contributors|Evidence last reviewed|Topic last updated)\s*\n|\Z)',
            text, re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            summary = summary_match.group(1).strip()[:1200]
        if not summary:
            for anchor in ["Definition", "History and exam", "Diagnosis"]:
                section = _extract_section("Summary", anchor)
                if section:
                    summary = section
                    break
        if not summary:
            # Fallback: first chunk of text after stripping nav boilerplate
            lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
            if lines:
                summary = lines[0][:800]
        result["summary"] = summary

        result["definition"] = _extract_section("Definition", "History and exam")

        # Key diagnostic factors are often under "History and exam" or "Diagnosis" sections.
        hx_section = _extract_section("History and exam", "Investigations")
        if not hx_section:
            hx_section = _extract_section("Diagnosis", "Investigations")
        if hx_section:
            # Look for bulleted lists or "Key diagnostic factors" subsection
            kdf_match = re.search(
                r'Key diagnostic factors[\s:]*(.+?)(?=Other diagnostic factors|Risk factors|Essential|Investigations|\Z)',
                hx_section, re.IGNORECASE | re.DOTALL,
            )
            if kdf_match:
                raw = kdf_match.group(1)
            else:
                raw = hx_section
            # Split into bullets/numbered items or sentences
            items = re.split(r'\n|(?<=\.)\s+', raw)
            for item in items:
                item = re.sub(r'^[-•\d+\.]+\s*', '', item).strip()
                if item and len(item) > 5 and len(item) < 200:
                    result["key_diagnostic_factors"].append(item)
            result["key_diagnostic_factors"] = result["key_diagnostic_factors"][:8]

        # Differentials
        diff_section = _extract_section("Differentials", "Investigations")
        if diff_section:
            items = re.split(r'\n', diff_section)
            for item in items:
                item = re.sub(r'^[-•\d+\.]+\s*', '', item).strip()
                # Keep relatively short differential names
                if item and 5 < len(item) < 100:
                    result["differentials"].append(item)
            result["differentials"] = result["differentials"][:10]

        # Investigations
        inv_section = _extract_section("Investigations", "Differentials")
        if inv_section:
            items = re.split(r'\n', inv_section)
            for item in items:
                item = re.sub(r'^[-•\d+\.]+\s*', '', item).strip()
                if item and len(item) > 5 and len(item) < 150:
                    result["investigations"].append(item)
            result["investigations"] = result["investigations"][:8]

        # Treatment algorithm headings
        tx_section = _extract_section("Treatment algorithm", "Follow up")
        if tx_section:
            items = re.split(r'\n', tx_section)
            for item in items:
                item = re.sub(r'^[-•\d+\.]+\s*', '', item).strip()
                if item and len(item) > 5 and len(item) < 120:
                    result["treatment_algorithm"].append(item)
            result["treatment_algorithm"] = result["treatment_algorithm"][:6]

        return result

    def lookup(self, query: str) -> Optional[BMJBestPracticeTopic]:
        """
        Convenience method: search BMJ Best Practice for a query and fetch the
        top candidate's public overview. Returns None if nothing is found.
        """
        if not query or len(query.strip()) < 3:
            return None

        cached = self._get_cached(query)
        if cached:
            print(f"[BMJ] Cache hit for '{query}' -> {cached.title}")
            return cached

        results = self.search_topics(query, max_results=1)
        if not results:
            print(f"[BMJ] No search results for '{query}'")
            return None

        top = results[0]
        print(f"[BMJ] Searching '{query}' -> topic {top['topic_id']} ({top['title']})")
        topic = self.fetch_overview(top["topic_id"], title=top["title"], url=top["url"])
        if topic:
            self._set_cached(query, topic)
        return topic


def _parse_json_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []


# =============================================================================
# Generic Medical Term Expansion (NOT disease-specific)
# =============================================================================

# Common spelling variants in medical literature (US vs UK, etc.)
SPELLING_VARIANTS = {
    # US -> UK
    "edema": "oedema",
    "estrogen": "oestrogen",
    "pediatric": "paediatric",
    "anemia": "anaemia",
    "hemoglobin": "haemoglobin",
    "leukocyte": "leucocyte",
    "diarrhea": "diarrhoea",
    "esophageal": "oesophageal",
    "fetus": "foetus",
    # Common Latin/Greek variants
    "tumor": "tumour",
    "behavior": "behaviour",
    "color": "colour",
}

# Common medical suffixes that have singular/plural forms
_MEDICAL_SUFFIX_PLURALS = {
    "itis": "itides",  # e.g., proctitis -> proctitides (rare, but exists)
    "oma": "omata",   # e.g., granuloma -> granulomata
    "on": "a",        # e.g., criterion -> criteria (not medical but common)
    "us": "i",        # e.g., focus -> foci
    "um": "a",        # e.g., diverticulum -> diverticula
}

# Common word endings that indicate pluralization
_COMMON_PLURALS = {
    "s": "", "es": "",  # Remove trailing s/es for singular forms
}


def _generate_spelling_variants(term: str) -> List[str]:
    """Generate spelling variants of a medical term (US/UK, etc.).

    This is purely structural — it handles common orthographic differences
    in medical terminology without knowing anything about specific diseases.
    """
    variants = [term]
    term_lower = term.lower()

    # Check for known spelling variants
    for us_form, uk_form in SPELLING_VARIANTS.items():
        if us_form in term_lower:
            variants.append(term.replace(us_form, uk_form))
        elif uk_form in term_lower:
            variants.append(term.replace(uk_form, us_form))

    # Remove duplicates
    return list(dict.fromkeys(variants))


def _expand_query_with_variants(query: str) -> str:
    """Expand a PubMed query by adding OR-connected spelling variants.

    Only expands terms that have known spelling variants.
    Does NOT add disease-specific synonyms.

    Example:
      '("edema"[Title/Abstract]) AND ("proctitis"[Title/Abstract])'
      -> '(("edema"[Title/Abstract]) OR ("oedema"[Title/Abstract])) AND ("proctitis"[Title/Abstract])'
    """
    if not query or len(query) < 10:
        return query

    # Find all quoted terms in the query
    quoted_terms = re.findall(r'"([^"]+)"(?:\[Title/Abstract\])?', query)

    if not quoted_terms:
        return query

    modified_query = query
    for term in quoted_terms:
        variants = _generate_spelling_variants(term)
        if len(variants) > 1:
            # Build OR-connected variant string
            if '[Title/Abstract]' in query:
                variant_parts = [f'"{v}"[Title/Abstract]' for v in variants]
            else:
                variant_parts = [f'"{v}"' for v in variants]

            or_string = " OR ".join(variant_parts)
            original_pattern = f'"{term}"[Title/Abstract]' if '[Title/Abstract]' in query else f'"{term}"'

            # Replace only the first occurrence to avoid double replacement
            if original_pattern in modified_query:
                modified_query = modified_query.replace(original_pattern, f"({or_string})", 1)

    return modified_query


# =============================================================================
# LLM Helper
# =============================================================================

_last_llm_call = 0


def _call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
    """Lightweight LLM call for retrieval enhancement with retry."""
    global _last_llm_call

    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set for retrieval enhancement")

    def _rate_limit():
        global _last_llm_call
        elapsed = time.time() - _last_llm_call
        if elapsed < LLM_RATE_LIMIT:
            time.sleep(LLM_RATE_LIMIT - elapsed)
        _last_llm_call = time.time()

    if LLM_PROVIDER in ("openai", "deepseek-flash"):
        if LLM_PROVIDER == "deepseek-flash":
            base = LLM_API_BASE.rstrip("/") if LLM_API_BASE else "https://api.deepseek.com/v1"
        else:
            base = LLM_API_BASE.rstrip("/") if LLM_API_BASE else "https://api.openai.com/v1"
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"}
        }

        last_exception = None
        for attempt in range(3):
            _rate_limit()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data or not data["choices"]:
                    print(f"[LLM Warning] Empty choices in response (attempt {attempt + 1}/3)")
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                msg = data["choices"][0]["message"]
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning_content", "")
                if not content and reasoning:
                    content = reasoning
                if not content.strip():
                    print(f"[LLM Warning] Empty content in response (attempt {attempt + 1}/3)")
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                return content.strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Retry {attempt + 1}/3] Timeout/Connection error, waiting {wait:.1f}s...")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 500, 502, 503):
                    last_exception = e
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[Retry {attempt + 1}/3] HTTP {e.response.status_code}, waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
        raise last_exception

    elif LLM_PROVIDER == "llamacpp":
        base = (os.environ.get("LLAMACPP_API_BASE", "") or "http://127.0.0.1:8080/v1").rstrip("/")
        model = os.environ.get("LLAMACPP_MODEL", "") or "local-model"
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exception = None
        for attempt in range(3):
            _rate_limit()
            try:
                # Serialize with run_inference's process-wide llama.cpp lock:
                # the local server is single-slot and rejects concurrent
                # requests with 503. Lazy import avoids a circular import
                # (run_inference imports this module at load time).
                from run_inference import _LLAMACPP_LOCK
                with _LLAMACPP_LOCK:
                    resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data or not data["choices"]:
                    print(f"[LLM Warning] Empty choices in response (attempt {attempt + 1}/3)")
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                content = data["choices"][0]["message"].get("content", "") or ""
                if not content.strip():
                    print(f"[LLM Warning] Empty content in response (attempt {attempt + 1}/3)")
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                return content.strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Retry {attempt + 1}/3] Timeout/Connection error, waiting {wait:.1f}s...")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 500, 502, 503):
                    last_exception = e
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[Retry {attempt + 1}/3] HTTP {e.response.status_code}, waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
        raise last_exception

    elif LLM_PROVIDER == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": LLM_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": LLM_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        last_exception = None
        for attempt in range(3):
            _rate_limit()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()["content"][0]["text"].strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Retry {attempt + 1}/3] Timeout/Connection error, waiting {wait:.1f}s...")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 500, 502, 503):
                    last_exception = e
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[Retry {attempt + 1}/3] HTTP {e.response.status_code}, waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
        raise last_exception

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KeywordItem:
    term: str
    mesh_suggestion: Optional[str] = None
    category: Optional[str] = None  # diagnosis, intervention, population, symptom, lab, imaging, organism, exposure


@dataclass
class StructuredKeywords:
    core: List[KeywordItem] = field(default_factory=list)
    differential: List[KeywordItem] = field(default_factory=list)
    exclude: List[KeywordItem] = field(default_factory=list)


@dataclass
class RetrievalStrategy:
    name: str
    query: str
    filters: str
    sort: str
    goal: str  # P0_case_match, P1_treatment, P2_differential
    weight: int
    max_results: int


@dataclass
class RetrievedSource:
    source: str  # "PubMed", "EuropePMC", "SemanticScholar"
    title: str
    authors: str
    year: Optional[str]
    abstract: Optional[str]
    key_findings: Optional[str]
    relevance: Optional[str]
    url: Optional[str]
    strategy_name: Optional[str] = None
    strategy_goal: Optional[str] = None
    pub_type: Optional[str] = None


@dataclass
class RetrievalResult:
    case_id: str
    query: str  # backward compat: joined queries
    sources: List[RetrievedSource]
    formatted_text: str
    retrieval_time: float
    success: bool
    error_message: Optional[str] = None
    queries: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    source_breakdown: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    case_id TEXT PRIMARY KEY,
                    query TEXT,
                    result_json TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()
            conn.close()

    def get(self, case_id: str) -> Optional[RetrievalResult]:
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT result_json FROM retrieval_cache WHERE case_id = ?",
                    (case_id,)
                )
                row = cursor.fetchone()
                conn.close()
            except sqlite3.Error as e:
                print(f"[Cache] SQLite error on get: {e}")
                return None
        if row:
            try:
                data = json.loads(row[0])
                # Schema migration: old cache may miss new fields
                sources = [RetrievedSource(**s) for s in data.get("sources", [])]
                return RetrievalResult(
                    case_id=data["case_id"],
                    query=data.get("query", ""),
                    sources=sources,
                    formatted_text=data.get("formatted_text", ""),
                    retrieval_time=data.get("retrieval_time", 0.0),
                    success=data.get("success", False),
                    error_message=data.get("error_message"),
                    queries=data.get("queries", []),
                    strategies=data.get("strategies", []),
                    source_breakdown=data.get("source_breakdown", {}),
                )
            except Exception as e:
                print(f"[Cache] Failed to parse cached result for {case_id}: {e}")
                return None
        return None

    def set(self, case_id: str, result: RetrievalResult):
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cursor = conn.cursor()
                data = asdict(result)
                data["sources"] = [asdict(s) for s in result.sources]
                cursor.execute(
                    "INSERT OR REPLACE INTO retrieval_cache (case_id, query, result_json, timestamp) VALUES (?, ?, ?, ?)",
                    (case_id, result.query, json.dumps(data, ensure_ascii=False), time.time())
                )
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                print(f"[Cache] SQLite error on set: {e}")


# =============================================================================
# Structured Keyword Extraction
# =============================================================================

def _is_poor_keyword(keyword: str) -> bool:
    """Heuristic to detect obviously poor keywords."""
    if not keyword:
        return True
    lower = keyword.lower()
    poor_signals = [
        "nurse practitioner", "dr.", "physician", "doctor",
        "no evidence of", "no history of", "no known",
        "patient was", "hospital because",
    ]
    if any(s in lower for s in poor_signals):
        return True
    for term in _GEO_DEMO_TERMS:
        if lower == term or lower.startswith(term + " ") or lower.endswith(" " + term):
            return True
    if len(keyword) < 4:
        return True
    return False


def _parse_structured_keywords(text: str) -> Optional[StructuredKeywords]:
    """Parse LLM JSON output into StructuredKeywords with multiple fallbacks."""
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback 1: find JSON object in text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Fallback 2: try line-by-line extraction for simple lists
        if data is None:
            return None

    if not isinstance(data, dict):
        return None

    def _parse_layer(layer_data):
        items = []
        if not isinstance(layer_data, list):
            return items
        for item in layer_data:
            if isinstance(item, dict):
                term = str(item.get("term", "")).strip()
                if term and len(term) >= 3 and not _is_poor_keyword(term):
                    items.append(KeywordItem(
                        term=term,
                        mesh_suggestion=item.get("mesh_suggestion") or item.get("mesh"),
                        category=item.get("category") or item.get("type")
                    ))
            elif isinstance(item, str):
                term = item.strip()
                if term and len(term) >= 3 and not _is_poor_keyword(term):
                    items.append(KeywordItem(term=term))
        return items

    core = _parse_layer(data.get("core", []))
    if not core:
        return None  # Must have at least some core terms

    differential = _parse_layer(data.get("differential", []))
    exclude = _parse_layer(data.get("exclude", []))

    # Post-process: sanitize core layer to catch LLM misplacements
    sanitized_core = []
    moved_to_diff = []
    removed = []

    # Signals that indicate a term belongs in differential, not core
    _DIFFERENTIAL_SIGNALS = {
        "possible", "possibly", "suspected", "suspect", "likely", "unlikely",
        "considered", "consider", "differential", "versus", "vs", "rule out",
        "ruled out", "cannot exclude", "cannot rule out", "consistent with",
        "favors", "favoring", "suggestive of", "presumed", "probable",
    }
    _NEGATION_SIGNALS = {
        "no ", "not ", "without", "absent", "negative", "ruled out",
        "no evidence of", "no history of", "no known", "denies",
    }

    for item in core:
        term_lower = item.term.lower()
        # Check if term should be in differential instead
        if any(sig in term_lower for sig in _DIFFERENTIAL_SIGNALS):
            moved_to_diff.append(item)
            continue
        # Check if term is actually a negation / excluded finding
        if any(sig in term_lower for sig in _NEGATION_SIGNALS):
            removed.append(item)
            continue
        sanitized_core.append(item)

    # Move misclassified items to differential
    if moved_to_diff:
        differential = moved_to_diff + differential
        print(f"[Sanitize] Moved {len(moved_to_diff)} item(s) from core to differential: {[i.term for i in moved_to_diff]}")
    if removed:
        exclude = removed + exclude
        print(f"[Sanitize] Moved {len(removed)} negated item(s) from core to exclude: {[i.term for i in removed]}")

    # NEW: Truncate overly long terms to improve PubMed retrieval
    truncated_core = []
    for item in sanitized_core:
        if len(item.term) > 50:
            # Extract key medical terms from long descriptions
            words = item.term.split()
            # Keep first 2-4 meaningful words (skip articles/prepositions)
            skip_words = {'the', 'a', 'an', 'in', 'with', 'of', 'and', 'for', 'was', 'were'}
            meaningful = [w for w in words[:6] if w.lower() not in skip_words]
            if meaningful:
                new_term = ' '.join(meaningful[:4])
                if len(new_term) >= 3:
                    print(f"[Truncate] '{item.term[:50]}...' -> '{new_term}'")
                    item = KeywordItem(term=new_term, mesh_suggestion=item.mesh_suggestion, category=item.category)
        truncated_core.append(item)
    
    # Ensure we still have core terms after sanitization
    if not truncated_core:
        # If everything was moved out, keep the first original core term as fallback
        truncated_core = [core[0]]
        print(f"[Sanitize] Warning: all core items were sanitized away, keeping fallback: {core[0].term}")

    return StructuredKeywords(
        core=truncated_core,
        differential=differential,
        exclude=exclude,
    )


def extract_keywords_with_llm(case_text: str, additional_context: str = "") -> Optional[StructuredKeywords]:
    """Use LLM to extract structured PubMed keywords from the case.
    Args:
        case_text: Full CPC case presentation text.
        additional_context: Optional supplementary text (e.g., generated clinical questions)
                           to enrich keyword extraction. Appended to the case excerpt.
    """
    context = case_text[:4000].replace("\n", " ")
    if additional_context:
        context = context + "\n\nAdditional diagnostic considerations from peer review:\n" + additional_context[:1500]
    prompt = f"""You are a medical librarian building PubMed search queries for a difficult CPC (clinicopathological conference) case.

Your task is to extract structured search elements from the case presentation, organized into THREE layers.

## CRITICAL DISTINCTION
The case presentation is INCOMPLETE (CPC format) — the final diagnosis is NOT yet revealed. You must extract based ONLY on what findings are actually described in the text.

## CRITICAL PRINCIPLE: CPC Cases Are Rare Disease Teaching Cases
CPC cases involve UNUSUAL presentations and RARE diseases. The diagnosis is often determined by a specific exposure, medication, toxin, or dietary habit.

## Layer 1: Clinical Core (必检索) — EXPOSURES AND SPECIFIC FINDINGS ONLY
Extract ONLY the following from the case:

**HIGHEST PRIORITY — EXPOSURES AND RISK FACTORS (must include if present):**
- Medications (prescription, OTC, herbal, supplements) — include dose and duration if mentioned
- Toxins or occupational exposures
- Dietary habits or specific foods (e.g., "licorice consumption", "high-calcium diet")
- Travel history or geographic exposures
- Recreational drug use or alcohol
- Animal contacts or environmental exposures
- Family history suggesting hereditary conditions
- **Behavioral or social risk factors that are EXPLICITLY described in the case** (e.g., specific sexual practices, needle sharing, occupational risks, institutional exposure) — ONLY if explicitly mentioned in the case text, NOT inferred from demographics
- **Recent procedures, surgeries, or medical interventions** — include type and timing if mentioned

**SECONDARY PRIORITY — SPECIFIC OBJECTIVE FINDINGS:**
- Key abnormal LAB values with specific numbers (e.g., "hypokalemia 2.1 mmol/L")
- Critical IMAGING findings with specific descriptions (e.g., "cavitary lung lesions with cavitation")
- Specific ORGANISMS mentioned (e.g., "Acinetobacter baumannii")
- Key PHYSICAL EXAM signs (e.g., "Kussmaul respirations")
- **UNUSUAL SENSORY OR BEHAVIORAL FINDINGS (must include if present):** Any abnormal reactions to sensory stimuli (sound, touch, light), peculiar behaviors, abnormal reflexes, or unexpected responses to examination — these are often the most diagnostically specific clues in CPC cases

**LOWEST PRIORITY — AVOID THESE:**
- Do NOT include vague symptoms like "fever", "pain", "cough" without specific context
- Do NOT include common disease names unless they are explicitly confirmed (not suspected)
- Do NOT include differential diagnoses that were only considered by physicians
- Do NOT include normal findings or negative results
- **EXCEPTION: Do NOT filter out seemingly minor but specific findings** — in CPC cases, a single specific finding (e.g., an unusual sensory reaction, a subtle imaging pattern, a peculiar exam sign) can be the key to the correct diagnosis

## Layer 2: Differential Challenges (选检索)
ONLY extract diagnostic dilemmas that involve RARE or UNUSUAL conditions:
- Rare organisms or exposures that are suspected
- Diagnostic dilemmas involving rare diseases (e.g., "infection vs rejection in transplant")
- Unexpected complications mentioned as possibilities
- Do NOT include common differentials like "pneumonia vs heart failure"

## Layer 3: Noise to Exclude (排除)
Identify common diseases or populations that are EXPLICITLY ruled out in the case.
Use these for soft exclusion (ranking penalty, NOT hard query exclusion).

## Output Format
Return STRICT JSON only. No markdown, no explanation.

{{
  "core": [
    {{"term": "...", "mesh_suggestion": "...", "category": "diagnosis"}},
    ...
  ],
  "differential": [
    {{"term": "...", "mesh_suggestion": "...", "category": "..."}},
    ...
  ],
  "exclude": [
    {{"term": "...", "mesh_suggestion": "...", "category": "..."}},
    ...
  ]
}}

Rules:
- Focus on OBJECTIVE ABNORMAL findings described in the text, NOT inferred or suspected diagnoses
- CORE must contain ONLY findings that are explicitly described as present in the patient
- If the text says "physicians considered X" or "differential included Y", those go to DIFFERENTIAL, NOT core
- AVOID: geographic locations, age/sex/race, hospital names, vague single-word symptoms
- **DO NOT dismiss seemingly minor findings** — in CPC cases, unusual sensory reactions, subtle exam findings, or unexpected patterns can be the most diagnostically specific clues
- If an exposure is mentioned, pair it with a clinical feature
- CPC cases often involve rare or unexpected connections. Look for the UNUSUAL combination
- Output 3-5 core items, 1-3 differential items, 0-2 exclude items.

## CRITICAL: KEYWORD LENGTH AND FORMAT
- Each term MUST be 2-6 words maximum. NO long sentences.
- GOOD examples: "hypokalemia 2.0 mmol/L", "ventricular fibrillation", "licorice consumption", "cavitary lung lesions"
- BAD examples: "severe hypokalemia-induced ventricular fibrillation cardiac arrest due to poor dietary intake" (too long)
- BAD examples: "rectal pain and mucopurulent rectal discharge in HIV-positive patient with anal-receptive intercourse" (too long)
- Use standard medical terminology, not narrative descriptions
- If a finding is complex, extract the KEY concept only (e.g., "rectal LGV" not full clinical picture)
- LAB values should include the specific number when mentioned (e.g., "potassium 2.0", "calcium 16.9")
- Medications should include generic name and dose if mentioned (e.g., "acetaminophen 650mg", "hydrochlorothiazide")
- Max 35 characters per term. Terms longer than 35 chars will be rejected.

Case:
{context}

JSON:"""
    try:
        result = _call_llm(prompt, temperature=0.0, max_tokens=2000)
        sk = _parse_structured_keywords(result)
        if sk:
            return sk
    except Exception as e:
        print(f"[LLM Structured Keyword Extraction Error] {e}")
    return None


# =============================================================================
# Heuristic Keyword Extraction (Fallback)
# =============================================================================

_BROAD_SYMPTOMS = {"fever", "pain", "cough", "fatigue", "nausea", "vomiting", "diarrhea",
                    "headache", "dyspnea", "weakness", "malaise", "rash", "swelling",
                    "bleeding", "syncope", "seizure", "confusion", "anxiety", "depression"}

_GEO_DEMO_TERMS = {
    "america", "american", "africa", "african", "asia", "asian", "europe", "european",
    "central america", "south america", "north america", "latin america",
    "china", "chinese", "india", "indian", "japan", "japanese", "korea", "korean",
    "boston", "massachusetts", "new york", "california", "texas", "florida",
    "rural", "urban", "suburban", "village", "town", "city",
    "year-old", "year old", "man", "woman", "male", "female", "boy", "girl",
    "infant", "child", "adolescent", "adult", "elderly", "pediatric",
    "black", "white", "hispanic", "caucasian", "african-american", "asian-american",
    "married", "single", "divorced", "pregnant", "primigravid", "multigravid",
    "construction", "worker", "teacher", "student", "farmer", "military",
    "homeless", "immigrant", "traveler", "tourist",
    "hospital", "clinic", "emergency department", "icu", "intensive care",
    # Geographic movement terms that are not risk factors
    "emigrated", "emigrated from", "immigrated", "immigrated from", "returned from",
    "born in", "raised in", "lived in", "recently moved", "recently returned",
    "visited", "traveled to", "came from", "arrived from",
}


def _is_broad_keyword(keyword: str) -> bool:
    """Check if keyword is too broad to be useful alone."""
    lower = keyword.lower().strip()
    if lower in _BROAD_SYMPTOMS:
        return True
    if len(lower) < 8 and lower.split()[0] in _BROAD_SYMPTOMS:
        return True
    return False


def _extract_keywords_heuristic(case_text: str, max_keywords: int = 5) -> List[str]:
    """Heuristic extraction as fallback when LLM fails."""
    keywords = []
    cleaned_text = re.sub(r'^Dr\.\s+[A-Z][^:]*:\s*', '', case_text, count=1)

    # Pattern 1: Presenting complaint from first sentence
    first_sent_match = re.search(r'^([A-Z][^\.\n]{10,400}\.)', cleaned_text.strip())
    if first_sent_match:
        first_sent = first_sent_match.group(1)
        complaint_match = re.search(
            r'(?:because of|due to|after)\s+([^\.\n]{3,200})',
            first_sent, re.IGNORECASE
        )
        if not complaint_match:
            complaint_match = re.search(
                r'(?:evaluated for|presented with|admitted with|seen for)\s+([^\.\n]{3,200})',
                first_sent, re.IGNORECASE
            )
        if not complaint_match:
            complaint_match = re.search(
                r'(?:with|for)\s+([^\.\n]{3,60})',
                first_sent, re.IGNORECASE
            )
        if complaint_match:
            keywords.append(complaint_match.group(1).strip())
        else:
            keywords.append(first_sent.strip())

    # Pattern 2: Abnormal lab values
    lab_patterns = [
        r'\b(?:potassium|sodium|calcium|glucose|creatinine|hemoglobin|platelet|white.cell|wbc|bicarbonate|lactate)\s+(?:level|count|concentration)?\s+(?:was\s+)?(\d+\.?\d*)\s*(?:mmol|mg|g|U|ng|pg|μ|mcg)',
        r'\b(severe|profound|marked|mild|moderate)\s+(hypokalemia|hyperkalemia|hyponatremia|hypernatremia|hypocalcemia|hypercalcemia|anemia|leukocytosis|thrombocytopenia|acidosis|alkalosis|hypoglycemia|hyperglycemia)',
    ]
    for pattern in lab_patterns:
        m = re.search(pattern, case_text, re.IGNORECASE)
        if m:
            kw = m.group(0).strip()
            if 5 < len(kw) < 60:
                keywords.append(kw)
                break

    # Pattern 3: Key imaging findings
    imaging_match = re.search(
        r'\b(?:CT|MRI|X-ray|ultrasound|echocardiogram|angiography|angiogram|PET)\s+(?:showed|revealed|demonstrated|identified)\s+([^\.\n]{10,100})',
        case_text, re.IGNORECASE
    )
    if imaging_match:
        keywords.append(imaging_match.group(1).strip())

    # Pattern 4: Key physical exam findings
    exam_match = re.search(
        r'\b(?:examination|exam)\s+(?:showed|revealed|was notable for|revealed)\s+([^\.\n]{10,80})',
        case_text, re.IGNORECASE
    )
    if exam_match:
        keywords.append(exam_match.group(1).strip())

    # Pattern 5: Risk factors / exposures (behavioral, social, medical history)
    risk_patterns = [
        # Sexual/behavioral exposures
        r'(?:sexual|anal|vaginal|oral)\s+(?:intercourse|contact|exposure|encounter)',
        r'(?:men who have sex with men|MSM|homosexual|bisexual)',
        r'(?:multiple partners|new partner|unprotected sex)',
        # Substance use
        r'(?:intravenous drug use|IV drug|needle sharing|substance abuse)',
        r'(?:heavy alcohol|tobacco|smoking|pack-year)',
        # Medical/occupational
        r'(?:recent surgery|prior surgery|transplant|immunosuppression)',
        r'(?:healthcare worker|hospital exposure|institutional exposure)',
        r'(?:travel to|returned from|immigrated from)',
        # Animal/environmental
        r'(?:cat scratch|dog bite|tick bite|mosquito|animal contact)',
        r'(?:well water|contaminated water|foodborne)',
    ]
    for pattern in risk_patterns:
        m = re.search(pattern, case_text, re.IGNORECASE)
        if m:
            kw = m.group(0).strip()
            if 5 < len(kw) < 80 and kw not in keywords:
                keywords.append(kw)
                break  # Only add the first matched risk factor to avoid overloading

    cleaned = []
    for kw in keywords:
        kw = re.sub(r'\s+', ' ', kw).strip()
        if kw and kw not in cleaned and len(kw) < 200:
            cleaned.append(kw)

    if not cleaned:
        fallback = cleaned_text[:200].replace('\n', ' ').strip()
        if fallback:
            cleaned = [fallback]

    return cleaned[:max_keywords]


# Keyword cache DB path (pre-extracted LLM structured keywords)
KEYWORD_CACHE_DB_PATH = os.environ.get(
    "KEYWORD_CACHE_DB_PATH",
    os.path.join(_PROJECT_ROOT, "cache", "structured_keywords_cache.db")
)


def _load_cached_keywords(case_id: str) -> Optional[StructuredKeywords]:
    """Load pre-extracted structured keywords from SQLite cache."""
    if not os.path.exists(KEYWORD_CACHE_DB_PATH):
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(KEYWORD_CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT core_json, differential_json, exclude_json FROM structured_keywords_cache WHERE case_id = ?",
            (case_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None

        def _parse_items(j):
            if not j:
                return []
            data = json.loads(j)
            return [KeywordItem(term=d.get("term", ""), mesh_suggestion=d.get("mesh_suggestion"), category=d.get("category")) for d in data]

        return StructuredKeywords(
            core=_parse_items(row[0]),
            differential=_parse_items(row[1]),
            exclude=_parse_items(row[2]),
        )
    except Exception as e:
        print(f"[Keyword Cache] Failed to load for {case_id}: {e}")
        return None


def extract_structured_keywords_from_case(case_text: str, case_id: str = "",
                                           additional_context: str = "") -> StructuredKeywords:
    """
    Extract structured keywords: cache first, then LLM, then heuristic fallback.
    Args:
        additional_context: Supplementary text (e.g., clinical questions) to enrich
                           LLM-based keyword extraction. Ignored for cache hits.
    """
    # P0: Try pre-extracted cache first (fast, no API call)
    if case_id:
        cached = _load_cached_keywords(case_id)
        if cached and cached.core:
            print(f"[Keyword] Cache hit: {len(cached.core)} core, {len(cached.differential)} diff, {len(cached.exclude)} exclude")
            for item in cached.core[:3]:
                tag = f"[{item.category or '?'}]"
                mesh = f" -> {item.mesh_suggestion}" if item.mesh_suggestion else ""
                print(f"  {tag} {item.term}{mesh}")
            return cached

    # P1: Live LLM extraction
    if ENABLE_LLM_KEYWORDS and LLM_API_KEY:
        try:
            sk = extract_keywords_with_llm(case_text, additional_context=additional_context)
            if sk and sk.core:
                print(f"[Keyword] LLM structured: {len(sk.core)} core, {len(sk.differential)} diff, {len(sk.exclude)} exclude")
                for item in sk.core[:3]:
                    tag = f"[{item.category or '?'}]"
                    mesh = f" -> {item.mesh_suggestion}" if item.mesh_suggestion else ""
                    print(f"  {tag} {item.term}{mesh}")
                return sk
        except Exception as e:
            print(f"[Keyword] LLM structured extraction failed: {e}, falling back to heuristic")

    # P2: Heuristic fallback
    heuristic_kws = _extract_keywords_heuristic(case_text, max_keywords=5)
    core = []
    for kw in heuristic_kws:
        if _is_poor_keyword(kw):
            continue
        core.append(KeywordItem(term=kw))

    if not core:
        fallback = case_text[:200].replace('\n', ' ').strip()
        if fallback:
            core = [KeywordItem(term=fallback)]

    return StructuredKeywords(core=core[:5])


# =============================================================================
# Backward-compatible keyword extraction
# =============================================================================

def extract_keywords_from_case(case_text: str, max_keywords: int = 5) -> List[str]:
    """
    Backward-compatible entry point.
    Internally uses structured extraction then flattens to a string list.
    """
    sk = extract_structured_keywords_from_case(case_text)
    all_terms = []
    for item in sk.core + sk.differential:
        if item.term and item.term not in all_terms:
            all_terms.append(item.term)
    return all_terms[:max_keywords]


# =============================================================================
# Query Matrix Builder
# =============================================================================

def build_query_matrix(keywords: StructuredKeywords) -> List[RetrievalStrategy]:
    """Build multiple PubMed search strategies from structured keywords."""
    if not keywords or not keywords.core:
        return []

    def _fmt_term(item: KeywordItem, use_mesh: bool = False) -> str:
        term = item.mesh_suggestion if (use_mesh and item.mesh_suggestion) else item.term
        term = term.replace('"', '').strip()
        # Strip age/sex preamble
        preamble_patterns = [
            r'^[Aa]\s+\d+[-\s]year[-\s]old\s+(?:man|woman|male|female|boy|girl|infant|child|adolescent)\s+(?:with\s+[^\s]+\s+)?was\s+(?:admitted to|evaluated at|seen in|referred to)\s+(?:this\s+hospital|the\s+emergency\s+department|the\s+clinic)\s+(?:because of|due to|for|after|with)\s+',
            r'^[Aa]\s+\d+[-\s]year[-\s]old\s+(?:man|woman|male|female|boy|girl|infant|child|adolescent)\s+(?:with\s+[^\s]+\s+)?was\s+(?:admitted|evaluated|seen)\s+(?:to|at|in)\s+[^\.]{0,50}?\s+(?:because of|due to|for|after|with)\s+',
            r'^[Aa]\s+\d+[-\s]year[-\s]old\s+(?:man|woman|male|female|boy|girl|infant|child|adolescent)\s+(?:with\s+)?',
        ]
        for pat in preamble_patterns:
            simplified = re.sub(pat, '', term, flags=re.IGNORECASE)
            if simplified != term and 10 < len(simplified) < 120:
                term = simplified
                break
        # Strip geo/demo terms
        for demo_term in _GEO_DEMO_TERMS:
            term = re.sub(r'\b' + re.escape(demo_term) + r'\b', '', term, flags=re.IGNORECASE)
        term = re.sub(r'\s+', ' ', term).strip()
        if len(term) > 100:
            trunc = term[:100]
            last_space = trunc.rfind(' ')
            if last_space > 30:
                term = trunc[:last_space]
            else:
                term = trunc
        return term

    def _clean_terms(items: List[KeywordItem], max_n: int = 4, use_mesh: bool = False) -> List[str]:
        cleaned = []
        for item in items:
            t = _fmt_term(item, use_mesh)
            if t and len(t) >= 3 and not _is_poor_keyword(t):
                cleaned.append(t)
        broad_indices = [i for i, kw in enumerate(cleaned) if _is_broad_keyword(kw)]
        non_broad = [kw for i, kw in enumerate(cleaned) if i not in broad_indices]
        broad_only = [kw for i, kw in enumerate(cleaned) if i in broad_indices]
        cleaned = non_broad + broad_only
        return cleaned[:max_n]

    core_terms = _clean_terms(keywords.core, max_n=4)
    diff_terms = _clean_terms(keywords.differential, max_n=3)

    if not core_terms:
        return []

    def _sanitize_for_pubmed(term: str) -> str:
        """Remove characters and phrases that break PubMed queries."""
        # Remove numeric comparison operators and thresholds
        # e.g., "hypoglycemia <10 mg/dL" -> "hypoglycemia"
        # Pattern: optional space, optional < or >, optional =, number, optional unit
        term = re.sub(r'\s*[<>]=?\s*\d+\.?\d*\s*(?:mg/dL|mg|g|mmol|U|μ|mcg|ml|mL|L|/dL|/L|%)', '', term, flags=re.IGNORECASE)
        # Remove standalone numbers with units (e.g., "650mg", "10 mg/dL")
        term = re.sub(r'\b\d+\.?\d*\s*(?:mg/dL|mg|g|mmol|U|μ|mcg|ml|mL|L|/dL|/L|%)\b', '', term, flags=re.IGNORECASE)
        # Remove trailing /dL or /L that might remain
        term = re.sub(r'\s*/dL\b', '', term, flags=re.IGNORECASE)
        term = re.sub(r'\s*/L\b', '', term, flags=re.IGNORECASE)
        # Remove causal connectors that don't exist in literature
        term = re.sub(r'\s+causing\s+', ' ', term, flags=re.IGNORECASE)
        term = re.sub(r'\s+secondary to\s+', ' ', term, flags=re.IGNORECASE)
        # Clean up extra spaces
        term = re.sub(r'\s+', ' ', term).strip()
        return term

    def _term_to_pubmed_fragment(term: str) -> str:
        """Convert a term to a PubMed query fragment.
        For short precise terms, use quoted phrase search.
        For longer text, split into meaningful sub-phrases connected with AND.
        Limits output to at most 2 AND-connected fragments per term to avoid overly
        restrictive queries that return zero results.
        """
        # Clean: remove quotes and trailing/leading punctuation
        term = term.replace('"', '').strip()
        term = term.strip('.,;:!?')  # remove outer trailing punctuation
        
        # NEW: Sanitize for PubMed
        term = _sanitize_for_pubmed(term)
        
        words = term.split()

        # Short precise terms (<=3 words, <=35 chars): use as quoted phrase
        if len(words) <= 3 and len(term) <= 35:
            return f'"{term}"[Title/Abstract]'

        # Try to split by common connectors first
        parts = re.split(
            r'\s+(?:associated with|due to|because of|related to|secondary to|complicated by|after|before)\s+',
            term, flags=re.IGNORECASE
        )
        parts = [p.strip() for p in parts if len(p.strip()) > 3]

        if len(parts) >= 2:
            fragments = []
            for p in parts[:2]:  # limit to 2 parts
                p = re.sub(r'^(?:the|a|an)\s+', '', p, flags=re.IGNORECASE).strip()
                p = p.strip('.,;:!?')
                if len(p) > 3:
                    pw = p.split()
                    if len(pw) > 5:
                        # Still too long, keep only meaningful words
                        keep = [w.strip('.,;:!?') for w in pw if len(w.strip('.,;:!?')) > 3 and w.lower().strip('.,;:!?') not in {
                            'with', 'and', 'for', 'was', 'were', 'had', 'have', 'has', 'been',
                            'this', 'that', 'patient', 'hospital', 'presented', 'admitted'
                        }]
                        if keep:
                            p = ' '.join(keep[:3])  # limit to 3 words
                    fragments.append(f'"{p}"[Title/Abstract]')
            if fragments:
                return ' AND '.join(fragments)

        # Fallback: preserve noun-phrase structure instead of isolated words
        # CRITICAL FIX: avoid producing queries like "partially" AND "exophytic"
        # which match broadly without their governing noun.
        _STOPWORDS = {
            'with', 'and', 'for', 'the', 'was', 'were', 'had', 'have', 'has', 'been',
            'this', 'that', 'from', 'into', 'over', 'under', 'after', 'before', 'patient',
            'hospital', 'presented', 'admitted', 'evaluated', 'because', 'since', 'without',
            'during', 'while', 'within', 'among', 'between', 'against', 'about',
        }

        # Medical entity suffixes that indicate a noun phrase head
        _MEDICAL_ENTITY_SUFFIXES = {
            'syndrome', 'disease', 'disorder', 'carcinoma', 'cancer', 'tumor', 'neoplasm',
            'pneumonia', 'infection', 'inflammation', 'lesion', 'mass', 'nodule', 'cyst',
            'abscess', 'granuloma', 'fibrosis', 'sclerosis', 'dystrophy', 'pathy',
            'emia', 'uria', 'penia', 'plasia', 'lysis', 'oma', 'itis', 'osis',
            'pathy', 'megaly', 'ectasia', 'stenosis', 'occlusion', 'effusion',
            'hypertrophy', 'atrophy', 'hypoplasia', 'hyperplasia', 'metaplasia',
        }

        def _has_medical_entity(word_list: list[str]) -> bool:
            """Check if any word ends with a medical entity suffix."""
            for w in word_list:
                w_lower = w.lower().strip('.,;:!?')
                if any(w_lower.endswith(suffix) for suffix in _MEDICAL_ENTITY_SUFFIXES):
                    return True
            return False

        def _is_meaningful_word(w: str) -> bool:
            w_clean = w.strip('.,;:!?')
            return len(w_clean) > 3 and w_clean.lower() not in _STOPWORDS

        # Strategy 1: Try to keep the first noun phrase (up to 4 words)
        # Stop at the first stopword or after 4 words
        phrase_words = []
        for w in words:
            if not _is_meaningful_word(w):
                break
            phrase_words.append(w.strip('.,;:!?'))
            if len(phrase_words) >= 4:
                break

        if phrase_words and _has_medical_entity(phrase_words):
            # We have a phrase containing a medical entity — use it as a quoted phrase
            return f'"{" ".join(phrase_words)}"[Title/Abstract]'

        # Strategy 2: If no medical entity in first phrase, try to find one anywhere
        entity_window = []
        for i, w in enumerate(words):
            w_clean = w.strip('.,;:!?')
            if not _is_meaningful_word(w):
                entity_window = []
                continue
            entity_window.append(w_clean)
            if len(entity_window) > 4:
                entity_window.pop(0)
            if _has_medical_entity(entity_window):
                return f'"{" ".join(entity_window)}"[Title/Abstract]'

        # Strategy 3: If still no entity, use the first 3-4 meaningful words as a phrase
        # (better than isolated AND-connected words)
        meaningful = [w.strip('.,;:!?') for w in words if _is_meaningful_word(w)]
        if len(meaningful) >= 3:
            return f'"{" ".join(meaningful[:3])}"[Title/Abstract]'
        elif len(meaningful) == 2:
            return ' AND '.join(f'"{w}"[Title/Abstract]' for w in meaningful)
        elif meaningful:
            return f'"{meaningful[0]}"[Title/Abstract]'

        # Last resort: use the cleaned full term as a quoted phrase
        cleaned_term = ' '.join(w.strip('.,;:!?') for w in words)
        return f'"{cleaned_term}"[Title/Abstract]'

    def _count_ands(query: str) -> int:
        return query.count(" AND ")

    def _safe_combine(fragments: List[str], max_total_ands: int = 2) -> str:
        """Combine fragments with AND, but drop terms if total ANDs would exceed limit."""
        if not fragments:
            return ""
        if len(fragments) == 1:
            return fragments[0]
        # If first fragment already uses multiple ANDs, only use it alone
        if _count_ands(fragments[0]) >= 2:
            return fragments[0]
        # Try combining first two
        combined = " AND ".join(fragments[:2])
        if _count_ands(combined) <= max_total_ands:
            return combined
        # Otherwise fall back to first fragment only
        return fragments[0]

    strategies = []

    # Strategy A: 精准匹配 (P0) - ALWAYS included
    core_fragments = [_term_to_pubmed_fragment(t) for t in core_terms[:2]]
    core_query = _safe_combine(core_fragments)
    # Also add MeSH suggestions if available (max 2), but only if core_query is simple
    mesh_terms = []
    for item in keywords.core[:2]:
        if item.mesh_suggestion:
            # FIX: Split semicolon-separated MeSH terms
            for mesh_term in item.mesh_suggestion.split(';'):
                mesh_term = mesh_term.strip()
                if mesh_term and len(mesh_term) >= 3:
                    mesh_terms.append(mesh_term)

    if mesh_terms and core_query and _count_ands(core_query) <= 1:
        # FIX: Build proper MeSH query with individual terms
        mesh_query = " AND ".join(f'"{m}"[Mesh]' for m in mesh_terms[:2])
        core_query = f"({core_query}) OR ({mesh_query})"

    # LOCAL MODEL OPT: Clinical Queries Filter selection
    # Based on deep analysis: Narrow filter (sensitivity ~50%) causes too many retrieval failures
    # For local models needing knowledge supplementation, Broad filter or no filter is preferred
    use_clinical_queries = os.environ.get("USE_CLINICAL_QUERIES", "broad").lower()
    if use_clinical_queries == "narrow":
        p0_filters = f"humans[Mesh] AND case reports[pt] AND {CLINICAL_QUERIES_FILTER['narrow']}"
        p1_filters = f"humans[Mesh] AND (clinical trial[pt] OR journal article[pt] OR comparative study[pt]) AND {CLINICAL_QUERIES_FILTER['narrow']}"
        p2_filters = f"humans[Mesh] AND {CLINICAL_QUERIES_FILTER['narrow']}"
    elif use_clinical_queries == "broad":
        # Use Broad filter for better recall (sensitivity ~80%) while maintaining reasonable precision
        p0_filters = f"humans[Mesh] AND case reports[pt] AND {CLINICAL_QUERIES_FILTER['broad']}"
        p1_filters = f"humans[Mesh] AND (clinical trial[pt] OR journal article[pt] OR comparative study[pt]) AND {CLINICAL_QUERIES_FILTER['broad']}"
        p2_filters = f"humans[Mesh] AND {CLINICAL_QUERIES_FILTER['broad']}"
    else:
        # No Clinical Queries filter - maximum recall for local models
        p0_filters = "humans[Mesh] AND case reports[pt]"
        p1_filters = "humans[Mesh] AND (clinical trial[pt] OR journal article[pt] OR comparative study[pt])"
        p2_filters = "humans[Mesh]"

    if core_query:
        strategies.append(RetrievalStrategy(
            name="精准匹配",
            query=core_query,
            filters=p0_filters,
            sort="relevance",
            goal="P0_case_match",
            weight=STRATEGY_WEIGHTS["精准匹配"],
            max_results=STRATEGY_MAX_RESULTS_LOCAL["精准匹配"],
        ))

    # OPTIMIZATION: Limit to max 3 strategies total to reduce API calls
    # Priority: 精准匹配 > 风险因素导向 > 治疗证据

    # Strategy D: 风险因素导向 (Risk Factor-Guided) - HIGH PRIORITY for cases with risk factors
    # This is often the most discriminative strategy
    risk_factor_items = [item for item in keywords.core if item.category in (
        "exposure", "risk_factor", "behavioral", "social", "occupational",
        "travel", "medication", "substance", "family_history"
    )]
    # Also heuristic: if a core term contains risk-related keywords but wasn't categorized
    risk_keywords = {
        "sexual", "intercourse", "msm", "homosexual", "partner", "unprotected",
        "intravenous", "iv drug", "needle", "substance", "alcohol", "tobacco", "smoking",
        "travel", "immigrated", "returned from", "visited",
        "occupation", "worker", "exposure", "contact",
        "animal", "cat", "dog", "tick", "mosquito", "bite",
        "family history", "hereditary", "genetic",
        "medication", "drug", "prescription", "supplement", "herbal",
        "surgery", "transplant", "immunosuppression",
    }
    for item in keywords.core:
        if item not in risk_factor_items:
            term_lower = item.term.lower()
            if any(rk in term_lower for rk in risk_keywords):
                risk_factor_items.append(item)

    # Remove duplicates while preserving order
    seen = set()
    risk_factor_items_unique = []
    for item in risk_factor_items:
        if item.term not in seen:
            seen.add(item.term)
            risk_factor_items_unique.append(item)
    risk_factor_items = risk_factor_items_unique

    if risk_factor_items and len(strategies) < 3:
        # Build risk-factor-guided queries
        # Query 1: Risk factor + most specific symptom (NOT disease name)
        # This finds literature linking the risk factor to diseases presenting with these symptoms
        symptom_items = [item for item in keywords.core if item.category in ("symptom", "physical_exam", "lab", "imaging")]
        if not symptom_items and core_terms:
            # Fallback: use first non-risk-factor core term
            symptom_items = [item for item in keywords.core if item not in risk_factor_items][:1]

        if symptom_items:
            risk_term = _fmt_term(risk_factor_items[0])
            symptom_term = _fmt_term(symptom_items[0])
            if risk_term and symptom_term and risk_term != symptom_term:
                risk_frag = _term_to_pubmed_fragment(risk_term)
                symptom_frag = _term_to_pubmed_fragment(symptom_term)
                if risk_frag and symptom_frag:
                    risk_query = f"{risk_frag} AND {symptom_frag}"
                    if _count_ands(risk_query) <= 2:
                        strategies.append(RetrievalStrategy(
                            name="风险因素导向",
                            query=risk_query,
                            filters=p2_filters,
                            sort="relevance",
                            goal="P2_risk_factor",
                            weight=STRATEGY_WEIGHTS.get("风险因素导向", 0.8),
                            max_results=STRATEGY_MAX_RESULTS_LOCAL.get("风险因素导向", 5),
                        ))
                        print(f"[QueryMatrix] Added risk-factor strategy: {risk_query[:100]}...")

    # Strategy B: 治疗证据 (P1) - Only if we have room and interventions exist
    if len(strategies) < 3:
        intervention_items = [item for item in keywords.core if item.category in ("intervention", "procedure", "treatment", "drug")]
        has_intervention = len(intervention_items) > 0
        has_differential = len(diff_terms) > 0

        if has_intervention or has_differential:
            treatment_terms = core_terms[:1]
            if intervention_items:
                treatment_terms = [_fmt_term(intervention_items[0])]
            if diff_terms:
                treatment_terms = treatment_terms + diff_terms[:1]
            treatment_terms = list(dict.fromkeys(treatment_terms))[:2]
            treatment_fragments = [_term_to_pubmed_fragment(t) for t in treatment_terms]
            treatment_query = _safe_combine(treatment_fragments)
            if treatment_query:
                strategies.append(RetrievalStrategy(
                    name="治疗证据",
                    query=treatment_query,
                    filters=p1_filters,
                    sort="relevance",
                    goal="P1_treatment",
                    weight=STRATEGY_WEIGHTS["治疗证据"],
                    max_results=STRATEGY_MAX_RESULTS_LOCAL["治疗证据"],
                ))

    # Strategy C: 鉴别预后 (P2) - Only if we have room
    if len(strategies) < 3 and (diff_terms or len(core_terms) >= 2):
        diff_query_terms = diff_terms[:2] if diff_terms else core_terms[1:2]
        if len(diff_query_terms) >= 1:
            diff_fragments = [_term_to_pubmed_fragment(t) for t in diff_query_terms[:2]]
            diff_query = _safe_combine(diff_fragments)
            symptom_items = [item for item in keywords.core if item.category == "symptom"]
            if symptom_items and _count_ands(diff_query) <= 1:
                symptom_frag = _term_to_pubmed_fragment(_fmt_term(symptom_items[0]))
                if _count_ands(diff_query + " AND " + symptom_frag) <= 2:
                    diff_query += f' AND {symptom_frag}'
            if diff_query:
                strategies.append(RetrievalStrategy(
                    name="鉴别预后",
                    query=diff_query,
                    filters=p2_filters,
                    sort="relevance",
                    goal="P2_differential",
                    weight=STRATEGY_WEIGHTS["鉴别预后"],
                    max_results=STRATEGY_MAX_RESULTS_LOCAL["鉴别预后"],
                ))

    print(f"[QueryMatrix] Built {len(strategies)} strategies (max 3)")
    return strategies


def build_pubmed_query(keywords) -> str:
    """
    Build a precise PubMed boolean query.
    Backward compat: accepts either List[str] or StructuredKeywords.
    """
    if isinstance(keywords, StructuredKeywords):
        strategies = build_query_matrix(keywords)
        if strategies:
            return strategies[0].query
        return ""

    # Legacy: list of strings
    if not keywords:
        return ""

    cleaned = []
    for kw in keywords:
        kw = kw.replace('"', '').strip()
        preamble_patterns = [
            r'^[Aa]\s+\d+[-\s]year[-\s]old\s+(?:man|woman|male|female|boy|girl|infant|child|adolescent)\s+(?:with\s+[^\s]+\s+)?was\s+(?:admitted to|evaluated at|seen in|referred to)\s+(?:this\s+hospital|the\s+emergency\s+department|the\s+clinic)\s+(?:because of|due to|for|after|with)\s+',
            r'^[Aa]\s+\d+[-\s]year[-\s]old\s+(?:man|woman|male|female|boy|girl|infant|child|adolescent)\s+(?:with\s+[^\s]+\s+)?was\s+(?:admitted|evaluated|seen)\s+(?:to|at|in)\s+[^\.]{0,50}?\s+(?:because of|due to|for|after|with)\s+',
        ]
        for pat in preamble_patterns:
            simplified = re.sub(pat, '', kw, flags=re.IGNORECASE)
            if simplified != kw and 10 < len(simplified) < 120:
                kw = simplified
                break
        for term in _GEO_DEMO_TERMS:
            kw = re.sub(r'\b' + re.escape(term) + r'\b', '', kw, flags=re.IGNORECASE)
        kw = re.sub(r'\s+', ' ', kw).strip()
        if len(kw) > 100:
            trunc = kw[:100]
            last_comma = trunc.rfind(',')
            last_space = trunc.rfind(' ')
            if last_comma > 30:
                kw = trunc[:last_comma]
            elif last_space > 30:
                kw = trunc[:last_space]
            else:
                kw = trunc
        if kw and len(kw) >= 3 and not _is_poor_keyword(kw):
            cleaned.append(kw)

    if not cleaned:
        return ""

    broad_indices = [i for i, kw in enumerate(cleaned) if _is_broad_keyword(kw)]
    non_broad = [kw for i, kw in enumerate(cleaned) if i not in broad_indices]
    broad_only = [kw for i, kw in enumerate(cleaned) if i in broad_indices]
    cleaned = non_broad + broad_only

    selected = cleaned[:4]
    core_query = " AND ".join(selected)
    query = f"({core_query}) NOT (review[pt] OR guideline[pt] OR meta-analysis[pt] OR consensus[pt])"
    return query


# =============================================================================
# Case Domain Classification + Targeted Query Generation (NEW — replaces
# question_generation + keyword_extraction + query_matrix pipeline)
# =============================================================================
# Inspired by AnySearch's pre-search discovery pattern:
#   get_sub_domains(domain) → discover what search paths exist
#   THEN construct targeted queries for each path
# Here we do the same: classify the case FIRST, THEN generate a focused query.

DOMAIN_CLASSIFICATION_PROMPT = """You are a medical librarian preparing a SINGLE focused PubMed search for a CPC case.

## Step 1 — Case Domain Classification
Identify the PRIMARY diagnostic domain (pick ONE):
infectious_disease | autoimmune_inflammatory | neoplastic | toxic_metabolic | vascular | genetic_congenital | drug_induced | nutritional_deficiency | idiopathic

## Step 2 — Key Diagnostic Hypotheses
List 2-3 specific diseases or disease categories that best explain the case.
For each: what makes it likely? What finding speaks against it?

## Step 3 — Build ONE Focused PubMed Query
Construct ONE PubMed query that would find literature to CONFIRM or REFUTE your leading hypothesis.
The query should:
- Include the top differential diagnosis as a phrase
- Include 1-2 key clinical findings (lab values, imaging findings, exam signs)
- Use [Title/Abstract] field tags for precision
- NOT include: age, sex, geography, hospital names, "case report"
- Be under 150 characters

## Output Format (JSON only, no markdown)
{{
  "domain": "toxic_metabolic",
  "hypotheses": [
    {{"disease": "...", "supporting": "...", "opposing": "..."}},
    {{"disease": "...", "supporting": "...", "opposing": "..."}}
  ],
  "pubmed_query": "(\\"disease X\\"[Title/Abstract]) AND (\\"finding Y\\"[Title/Abstract])",
  "search_rationale": "One sentence explaining why this query is likely to return relevant evidence"
}}

## Case
{case_text}

JSON:"""


def classify_and_build_query(case_text: str, max_chars: int = 4000) -> dict:
    """Single LLM call: classify case domain + generate one focused PubMed query.

    Replaces the old pipeline:
      question_gen → keyword_extraction → build_query_matrix (3 strategies)

    Returns dict with keys: domain, hypotheses, pubmed_query, search_rationale.
    On failure, returns empty domain and query so callers can degrade gracefully.
    """
    prompt = DOMAIN_CLASSIFICATION_PROMPT.format(case_text=case_text[:max_chars])
    try:
        raw = _call_llm(prompt, temperature=0.0, max_tokens=1200)
    except Exception as e:
        print(f"[DomainClassify] LLM call failed: {e}")
        return {"domain": "", "hypotheses": [], "pubmed_query": "", "search_rationale": ""}

    # Parse JSON from LLM output (robust against markdown fences)
    data = _parse_json_from_llm(raw)
    if data is None:
        print(f"[DomainClassify] JSON parse failed, raw: {raw[:200]}")
        return {"domain": "", "hypotheses": [], "pubmed_query": "", "search_rationale": ""}

    pq = data.get("pubmed_query", "")
    # Validate: query must be non-empty and look like a PubMed query
    if not pq or len(pq) < 10:
        print(f"[DomainClassify] Empty or too-short query: '{pq}'")
        return {"domain": data.get("domain", ""), "hypotheses": data.get("hypotheses", []),
                "pubmed_query": "", "search_rationale": data.get("search_rationale", "")}

    # Sanitize: remove characters that break PubMed
    pq = _sanitize_query(pq)
    print(f"[DomainClassify] domain={data.get('domain','')} query='{pq[:100]}...'")
    for h in data.get("hypotheses", [])[:2]:
        print(f"  hypothesis: {h.get('disease', '')[:60]}")

    return {
        "domain": data.get("domain", ""),
        "hypotheses": data.get("hypotheses", []),
        "pubmed_query": pq,
        "search_rationale": data.get("search_rationale", ""),
    }


# =============================================================================
# Shared utilities
# =============================================================================

def _strip_json_fences(text: str) -> str:
    """Strip markdown JSON fences and return clean JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _parse_json_from_llm(text: str) -> Optional[dict]:
    """Parse JSON from LLM output, handling markdown fences and fallback regex."""
    text = _strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _sanitize_query(q: str) -> str:
    """Sanitize a PubMed query: normalize quotes and strip non-printable chars."""
    if not q:
        return q
    q = q.replace('“', '"').replace(').replace(“, ', '"')
    q = q.replace('\n', ' ').replace('\r', ' ')
    q = re.sub(r'[^\x20-\x7E]', '', q)
    return q.strip()


# =============================================================================
# Post-hoc RAG Query Builder (v3 — queries FROM the initial diagnosis)
# =============================================================================

POSTHOC_QUERY_PROMPT = """You are a medical librarian. A physician has made an INITIAL diagnosis on a CPC case with a DIFFERENTIAL list. Your job: build PubMed queries to CHALLENGE the leading diagnosis by exploring the OTHER differential diagnoses, and to generate COMPARATIVE queries that directly pit the leading diagnosis against alternatives.

## Initial Diagnosis
{initial_diagnosis}

## Differential Diagnoses
{differential_text}

## Case Text
{case_text}

## Instructions
Build 4 PubMed queries designed to:
1. **COMPARE leading vs top alternative** — Direct comparison query: "(leading diagnosis) vs (top alternative) differential diagnosis" or "(leading diagnosis) and (top alternative) clinical features" — This is the MOST IMPORTANT query. It finds literature that directly compares the two most likely diagnoses.
2. **EXPLORE the strongest alternative** — Search for the TOP differential diagnosis (NOT the leading one) + 1 key case finding that SUPPORTS it. Use ONLY 1 AND. This finds evidence FOR the alternative, not against it.
3. **SEARCH by clinical phenotype** — Combine 2 KEY CLINICAL FINDINGS from the case with ZERO disease names. Use ONLY 1 AND. This query is the safety net.
4. **REFUTE a specific differential** — Pick one differential diagnosis and search for evidence that it does NOT fit this case. Combine the differential diagnosis + a finding that argues against it. Use ONLY 1 AND.

CRITICAL: If the case contains ANY of the following risk factors or exposures, you MUST include them in at least ONE query (especially Query 2 or Query 3):
- Sexual exposures (e.g., MSM, anal intercourse, multiple partners, unprotected sex)
- Substance use (e.g., IV drug use, alcohol, tobacco)
- Travel history or geographic exposures
- Occupational or environmental exposures
- Animal contacts or bites
- Recent medical procedures or surgeries
- Family history of hereditary conditions

These risk factors are often the KEY to the correct diagnosis in CPC cases. Do NOT ignore them.

CRITICAL RULES:
- **MAXIMUM 1 AND per query** — this is the most important rule. More ANDs cause zero results.
- **Query 1 (COMPARE) is the most important** — If the leading diagnosis is wrong, this query will find literature supporting the alternative. Use phrases like "differential diagnosis", "clinical features", "diagnosis", or "versus".
- **Query 2 MUST be about a DIFFERENT disease than the leading diagnosis** — Search for evidence SUPPORTING the alternative, not refuting it.
- **Query 3: NO disease names, NO syndrome names, NO diagnostic labels** — only raw clinical observations
- **For organism-specific diagnoses**: Use the organism name + a moderately specific finding.
- Each query under 150 characters
- Use [Title/Abstract] field tags
- Do NOT include: age, sex, geography, "case report"

## Output Format (JSON only, no markdown)
{{
  "pubmed_queries": [
    "(\\"leading diagnosis X\\"[Title/Abstract]) AND (\\"top alternative Y\\"[Title/Abstract])",
    "(\\"alternative diagnosis Z\\"[Title/Abstract]) AND (\\"supporting finding W\\"[Title/Abstract])",
    "(\\"finding A\\"[Title/Abstract]) AND (\\"finding B\\"[Title/Abstract])",
    "(\\"differential to refute\\"[Title/Abstract]) AND (\\"finding against it\\"[Title/Abstract])"
  ],
  "search_rationale": "One sentence explaining the comparison-oriented search strategy"
}}

JSON:"""


def build_posthoc_queries(
    case_text: str,
    initial_diagnosis: str,
    differential: list,
    max_chars: int = 4000,
) -> dict:
    """Build PubMed queries AFTER initial diagnosis, designed to CHALLENGE it.

    Unlike extract_findings_and_build_queries() which uses disease-agnostic findings,
    this uses the initial diagnosis as a target to verify or refute.

    Returns dict with keys: pubmed_queries, search_rationale.
    """
    diff_lines = []
    for d in differential[:5]:
        disease = d.get("disease", "") if isinstance(d, dict) else str(d)
        rationale = d.get("rationale", "") if isinstance(d, dict) else ""
        diff_lines.append(f"- {disease}: {rationale}")
    differential_text = "\n".join(diff_lines) if diff_lines else "None"

    prompt = POSTHOC_QUERY_PROMPT.format(
        initial_diagnosis=initial_diagnosis[:300],
        differential_text=differential_text[:1000],
        case_text=case_text[:max_chars],
    )
    try:
        raw = _call_llm(prompt, temperature=0.0, max_tokens=1200)
    except Exception as e:
        print(f"[PosthocQuery] LLM call failed: {e}")
        return {"pubmed_queries": [], "search_rationale": ""}

    # Parse JSON
    data = _parse_json_from_llm(raw)
    if data is None:
        print(f"[PosthocQuery] JSON parse failed")
        return {"pubmed_queries": [], "search_rationale": ""}

    queries = data.get("pubmed_queries", [])
    if isinstance(queries, str):
        queries = [queries]

    valid_queries = []
    for q in queries:
        if not q or len(q) < 10:
            continue
        q = _sanitize_query(q)
        if len(q) >= 10:
            valid_queries.append(q)

    # DIAGNOSTIC DIVERSITY GUARD: If all queries mention the same disease,
    # add a broad query to ensure we don't miss alternative diagnoses.
    # This is a PURELY STRUCTURAL check — no disease-specific knowledge.
    if len(valid_queries) >= 2:
        # Extract disease names from queries (terms in quotes before [Title/Abstract])
        diseases_mentioned = set()
        for q in valid_queries:
            matches = re.findall(r'"([^"]+)"\[Title/Abstract\]', q)
            for m in matches:
                # Skip generic terms that are likely findings, not diseases
                generic_terms = {"fever", "pain", "pruritus", "cough", "nausea", "vomiting",
                                 "fatigue", "headache", "diarrhea", "constipation", "dyspnea",
                                 "chest pain", "abdominal pain", "joint pain", "back pain",
                                 "pregnancy", "neonatal", "elderly", "pediatric", "child",
                                 "infectious", "autoimmune", "toxic", "sexual transmission",
                                 "occupational exposure", "travel", "medication"}
                if m.lower() not in generic_terms and len(m) > 3:
                    diseases_mentioned.add(m.lower())

        # If all queries mention the same disease, add a diversity query
        if len(diseases_mentioned) == 1:
            dominant_disease = list(diseases_mentioned)[0]
            print(f"[DiversityGuard] All queries mention '{dominant_disease}' — adding broad alternative query")
            # Add a generic query using the most distinctive finding + broad category
            # Extract findings from the case text (first 1000 chars for speed)
            case_snippet = case_text[:1000].lower()
            # Look for distinctive findings (lab values, imaging, specific symptoms)
            distinctive_patterns = [
                r'(\d+\.?\d*\s*(?:mg|g|U|mmol|μ|mcg|ng|pg)/(?:dl|l|ml))',
                r'(bilateral|unilateral|diffuse|focal|segmental)\s+([a-z\s]+)',
                r'(elevated|decreased|high|low|abnormal)\s+([a-z\s]+)',
            ]
            distinctive_findings = []
            for pattern in distinctive_patterns:
                matches = re.findall(pattern, case_snippet)
                for m in matches:
                    if isinstance(m, tuple):
                        finding = ' '.join(m).strip()
                    else:
                        finding = m.strip()
                    if len(finding) > 5 and len(finding) < 50:
                        distinctive_findings.append(finding)

            if distinctive_findings:
                # Pick the most distinctive finding and combine with broad category
                broad_query = f'("{distinctive_findings[0]}"[Title/Abstract]) AND ("differential diagnosis"[Title/Abstract])'
                valid_queries.append(broad_query)
                print(f"[DiversityGuard] Added broad query: {broad_query[:100]}...")

    print(f"[PosthocQuery] built {len(valid_queries)} queries to challenge dx")
    for i, q in enumerate(valid_queries[:4]):
        print(f"  Q{i+1}: {q[:120]}...")

    return {
        "pubmed_queries": valid_queries[:4],
        "search_rationale": data.get("search_rationale", ""),
    }


# =============================================================================
# Reverse Query Builder (NEW — queries to REFUTE differential diagnoses)
# =============================================================================

REVERSE_QUERY_PROMPT = """You are a medical librarian. A physician has made an INITIAL diagnosis on a CPC case with a list of DIFFERENTIAL DIAGNOSES. Your job: build PubMed queries to actively REFUTE each differential diagnosis by finding evidence that it does NOT fit this case.

## Initial Diagnosis
{initial_diagnosis}

## Differential Diagnoses
{differential_text}

## Case Text
{case_text}

## Instructions
Build 2-3 PubMed queries designed to FIND EVIDENCE THAT RULES OUT or ARGUES AGAINST specific differential diagnoses.

- **Query 1 (Refute Leading Alternative):** Search for the TOP differential diagnosis + a case finding that SPEAKS AGAINST it. Use ONLY 1 AND. The finding should be relevant but not overly specific.
- **Query 2 (Refute Second Alternative):** Search for the SECOND differential diagnosis + a case finding that is ATYPICAL for it. Use ONLY 1 AND.
- **Query 3 (Find Better Fit):** Search for the case's most distinctive finding + a broad category (NOT a specific disease). Use ONLY 1 AND. The finding should be moderately specific (e.g., "anal fissure", "rectal ulcer") rather than ultra-specific (e.g., "sentinel skin tag with superficial fissure") or generic (e.g., "pain", "bleeding"). This finds diseases that match the finding but were NOT considered.

CRITICAL: If the case contains ANY risk factors or exposures (sexual history, substance use, travel, occupational, animal contact, etc.), Query 3 MUST include the risk factor as the "distinctive finding". For example: ("sexual encounter"[Title/Abstract]) AND ("proctitis"[Title/Abstract]) or ("MSM"[Title/Abstract]) AND ("rectal pain"[Title/Abstract]). Risk factors are often the key to finding the correct alternative diagnosis.

CRITICAL RULES:
- **MAXIMUM 1 AND per query** — this is the most important rule
- **BALANCE specificity**: Use ONE moderately specific finding. Avoid terms so rare they never appear in literature, and avoid terms so broad they return irrelevant results.
- Each query under 150 characters
- Use [Title/Abstract] field tags
- Do NOT include: age, sex, geography, "case report"
- Query 1 and 2 should combine: ("differential diagnosis X"[Title/Abstract]) AND ("finding that argues against it"[Title/Abstract])
- Query 3 should combine: ("distinctive finding"[Title/Abstract]) AND ("broad category"[Title/Abstract]) where broad category is: "sexually transmitted", "autoimmune", "toxic", "infectious", etc.
- The goal is to find literature that CHALLENGES the differential, not confirms it

## Output Format (JSON only, no markdown)
{{
  "pubmed_queries": [
    "(\\"differential diagnosis X\\"[Title/Abstract]) AND (\\"finding against it\\"[Title/Abstract])",
    "(\\"differential diagnosis Y\\"[Title/Abstract]) AND (\\"atypical finding\\"[Title/Abstract])",
    "(\\"distinctive finding\\"[Title/Abstract]) AND (\\"broad category\\"[Title/Abstract])"
  ],
  "search_rationale": "One sentence explaining the refutation-oriented search strategy"
}}

JSON:"""


def build_reverse_queries(
    case_text: str,
    initial_diagnosis: str,
    differential: list,
    max_chars: int = 4000,
) -> dict:
    """Build PubMed queries designed to REFUTE differential diagnoses.

    Unlike build_posthoc_queries() which verifies the leading diagnosis,
    this actively searches for evidence AGAINST the differential diagnoses.
    This helps catch cases where the initial diagnosis is wrong.

    Returns dict with keys: pubmed_queries, search_rationale.
    """
    diff_lines = []
    for d in differential[:5]:
        disease = d.get("disease", "") if isinstance(d, dict) else str(d)
        rationale = d.get("rationale", "") if isinstance(d, dict) else ""
        diff_lines.append(f"- {disease}: {rationale}")
    differential_text = "\n".join(diff_lines) if diff_lines else "None"

    prompt = REVERSE_QUERY_PROMPT.format(
        initial_diagnosis=initial_diagnosis[:300],
        differential_text=differential_text[:1000],
        case_text=case_text[:max_chars],
    )
    try:
        raw = _call_llm(prompt, temperature=0.0, max_tokens=1200)
    except Exception as e:
        print(f"[ReverseQuery] LLM call failed: {e}")
        return {"pubmed_queries": [], "search_rationale": ""}

    # Parse JSON
    data = _parse_json_from_llm(raw)
    if data is None:
        print(f"[ReverseQuery] JSON parse failed")
        return {"pubmed_queries": [], "search_rationale": ""}

    queries = data.get("pubmed_queries", [])
    if isinstance(queries, str):
        queries = [queries]

    valid_queries = []
    for q in queries:
        if not q or len(q) < 10:
            continue
        q = _sanitize_query(q)
        if len(q) >= 10:
            valid_queries.append(q)

    print(f"[ReverseQuery] built {len(valid_queries)} queries to refute differentials")
    for i, q in enumerate(valid_queries[:3]):
        print(f"  RQ{i+1}: {q[:120]}...")

    return {
        "pubmed_queries": valid_queries[:3],
        "search_rationale": data.get("search_rationale", ""),
    }


# =============================================================================
# Broad Net Query Builder (NEW — completely independent of Pass 1 diagnosis)
# =============================================================================

BROAD_NET_QUERY_PROMPT = """You are a medical librarian preparing PubMed searches for a CPC case.

**CRITICAL: You do NOT know the diagnosis. You only see the raw case text below.**
Your task: build 2-3 BROAD PubMed queries that will find diseases matching the case's clinical phenotype, WITHOUT using any diagnostic labels.

## Step 1 — Identify the Most Distinctive Raw Findings
Read the case text carefully. Pick 2-3 findings that are:
- SPECIFIC (not generic symptoms like "pain" or "fever")
- DISTINCTIVE (would narrow the differential to a small set of diseases)
- OBJECTIVE (observed, not interpreted)

Examples of good findings:
- "bright red blood per rectum" (not "bleeding")
- "sentinel skin tag at anal verge" (not "anal problem")
- "superficial anal fissure with rectal ulceration" (not "rectal pain")
- "sexual encounter with a man 6 weeks prior" (exposure)
- "elevated alkaline phosphatase 245 U/L" (lab)

## Step 2 — Build 2-3 Broad Queries
Each query should combine:
- ONE distinctive finding (specific)
- ONE broad category or body system (to cast a wide net)

Use ONLY 1 AND per query.

Query 1: ("most distinctive finding"[Title/Abstract]) AND ("broad category"[Title/Abstract])
Query 2: ("second distinctive finding"[Title/Abstract]) AND ("body system"[Title/Abstract])
Query 3: ("exposure or risk factor"[Title/Abstract]) AND ("clinical finding"[Title/Abstract])

Broad categories to use: "sexually transmitted", "autoimmune", "infectious", "inflammatory", "neoplastic", "toxic", "metabolic", "vascular", "genetic"
Body systems: "rectum", "anus", "colon", "liver", "lung", "skin", "kidney", "brain", "bone marrow", "lymph node"

CRITICAL RULES:
- **NO disease names of any kind** — not in the findings, not in the queries
- **NO diagnostic labels** like "proctitis", "pneumonia", "nephritis" — use the underlying findings instead
- **NO interpretations** like "consistent with", "suggestive of", "likely represents"
- **NO physician differential diagnoses** mentioned in the text
- **MAXIMUM 1 AND per query**
- Use [Title/Abstract] field tags
- Each query under 150 characters
- Do NOT include: age, sex, geography, hospital names, "case report"

## Output Format (JSON only, no markdown)
{{
  "pubmed_queries": [
    "(\"distinctive finding 1\"[Title/Abstract]) AND (\"broad category\"[Title/Abstract])",
    "(\"distinctive finding 2\"[Title/Abstract]) AND (\"body system\"[Title/Abstract])",
    "(\"exposure\"[Title/Abstract]) AND (\"clinical finding\"[Title/Abstract])"
  ],
  "search_rationale": "One sentence explaining the broad-net search strategy"
}}

## Case
{case_text}

JSON:"""


def build_broad_net_queries(
    case_text: str,
    max_chars: int = 4000,
) -> dict:
    """Build completely disease-agnostic broad-net PubMed queries.

    Unlike extract_findings_and_build_queries() which may be influenced by
    the LLM's implicit diagnostic bias, this prompt explicitly blinds the LLM
    to any diagnosis and forces it to use only raw clinical findings + broad categories.

    Returns dict with keys: pubmed_queries, search_rationale.
    """
    prompt = BROAD_NET_QUERY_PROMPT.format(case_text=case_text[:max_chars])
    try:
        raw = _call_llm(prompt, temperature=0.0, max_tokens=1200)
    except Exception as e:
        print(f"[BroadNetQuery] LLM call failed: {e}")
        return {"pubmed_queries": [], "search_rationale": ""}

    # Parse JSON with fallback
    data = _parse_json_from_llm(raw)
    if data is None:
        # Fallback: try to extract queries directly from text
        print(f"[BroadNetQuery] JSON parse failed, attempting fallback extraction")
        queries = _extract_queries_fallback(raw)
        if queries:
            print(f"[BroadNetQuery] Fallback extracted {len(queries)} queries")
            return {
                "pubmed_queries": queries,
                "search_rationale": "Fallback extraction from non-JSON output",
            }
        return {"pubmed_queries": [], "search_rationale": ""}

    queries = data.get("pubmed_queries", [])
    if isinstance(queries, str):
        queries = [queries]

    valid_queries = []
    for q in queries:
        if not q or len(q) < 10:
            continue
        q = _sanitize_query(q)
        if len(q) >= 10:
            valid_queries.append(q)

    print(f"[BroadNetQuery] built {len(valid_queries)} broad-net queries")
    for i, q in enumerate(valid_queries[:3]):
        print(f"  BNQ{i+1}: {q[:120]}...")

    return {
        "pubmed_queries": valid_queries[:3],
        "search_rationale": data.get("search_rationale", ""),
    }


def _extract_queries_fallback(text: str) -> list:
    """Extract PubMed queries from non-JSON LLM output using regex.

    Looks for patterns like: ("term"[Title/Abstract]) AND ("term"[Title/Abstract])
    """
    import re
    # Match patterns like: ("something"[Title/Abstract]) AND ("something"[Title/Abstract])
    pattern = r'\("([^"]+)"\[Title/Abstract\]\)\s+AND\s+\("([^"]+)"\[Title/Abstract\]\)'
    matches = re.findall(pattern, text)
    queries = []
    for m in matches:
        if len(m) == 2:
            q = f'("{m[0]}"[Title/Abstract]) AND ("{m[1]}"[Title/Abstract])'
            queries.append(q)
    return queries


# =============================================================================
# Clinical-Findings-Anchored Retrieval (v2 — replaces domain classification)
# =============================================================================

CLINICAL_FINDINGS_QUERY_PROMPT = """You are a medical librarian preparing PubMed searches for a CPC case.

Your task: extract KEY CLINICAL FINDINGS (disease-agnostic) and build MULTIPLE PubMed queries anchored to those findings, NOT to diagnostic guesses.

## Step 1 — Extract Key Clinical Findings (BLINDED — No Diagnosis Allowed)

**CRITICAL: You are NOT a physician. You are a medical librarian who has NOT seen any diagnosis.**
You only see the raw case text below. You do NOT know what disease this is.
Your job is to extract ONLY the objective facts written in the text — like a stenographer.

Identify 4-6 specific, measurable, diagnostically-informative findings from the case.
Focus on WHAT WAS OBSERVED, not what you think it means:

- **Abnormal lab values** (with numbers and units): e.g., "CK 5358 U/L", "creatinine 3.04 mg/dL", "eosinophils 83%"
- **Imaging findings** (CT/MRI/X-ray/ultrasound descriptions): e.g., "bilateral hilar lymphadenopathy", "splenic hypoattenuating lesions"
- **Physical exam signs**: e.g., "left facial sensory loss", "hoarseness with periorbital edema"
- **Specific symptom clusters or syndromes**: e.g., "subacute ataxia + dysarthria + weight loss"
- **Organ system involvement pattern**: e.g., "skin + lung + kidney involvement"
- **Key exposures or risk factors**: e.g., "sexual encounter with a man 6 weeks prior", "naproxen use", "construction worker"

**FORBIDDEN in findings** (these are interpretations, not observations):
- Any differential diagnosis mentioned by physicians in the text
- Any disease name: "proctitis", "pneumonia", "nephritis", "hepatitis", etc.
- Any syndrome name
- Any phrase like "consistent with", "suggestive of", "likely represents"
- Any interpretation of what the finding means

**RULE: If the text says "physicians considered X" or "differential included Y", do NOT include X or Y as a finding. Only include what the patient actually had.**

For each finding, note its diagnostic significance in ONE sentence — what categories it narrows TO or AWAY FROM. Do NOT name a specific disease unless the finding is pathognomonic.

## Step 2 — Build 5 Distinct PubMed Queries (DISEASE NAMES FORBIDDEN)

**CRITICAL: You do NOT know the diagnosis. You are searching for diseases that match these findings.**

QUERIES MUST CONTAIN ONLY: body parts, lab test names, imaging modality names, symptom words, physical exam terms, pathology descriptors, exposure terms. NO DISEASE NAMES OF ANY KIND.

**FORBIDDEN in queries** (this is checked by automated validation):
- Any specific disease name: "sarcoidosis", "lupus", "CTLA-4", "autoimmune hepatitis", "APS-1", "APECED", "CVID", "lymphoma", "cirrhosis", "hepatitis", etc.
- Any syndrome name that implies a specific diagnosis
- Any genetic/molecular name: "TERT", "AIRE", "KRAS", etc.
- Any diagnostic label: "proctitis", "pneumonia", "nephritis" — these are disease names, NOT findings
- Generic symptoms alone: "pain", "fever", "bleeding", "cough" — these are too broad and cause irrelevant results
- Any differential diagnosis mentioned by physicians in the case text

**ALLOWED in queries** (clinical descriptors only):
- Symptoms: "jaundice", "pruritus", "dyspnea", "ataxia", "seizure", "weight loss", "bright red blood per rectum"
- Exam signs: "leukoplakia", "hepatosplenomegaly", "lymphadenopathy", "clubbing", "edema", "sentinel skin tag", "bilateral anal fissures", "superficial anal fissure"
- Lab abnormalities: "pancytopenia", "thrombocytopenia", "eosinophilia", "hypercalcemia", "elevated creatine kinase"
- Imaging descriptions: "bilateral hilar lymphadenopathy", "splenic hypoattenuating lesions", "white matter lesions"
- Pathology terms: "granuloma", "vasculitis", "necrosis", "fibrosis", "angiocentric infiltrate"
- Anatomical terms: "liver", "lung", "kidney", "brain", "skin", "bone marrow", "rectum", "anus"
- Exposures: "sexual transmission", "medication use", "occupational exposure", "travel"

**VALIDATION: Before outputting each query, verify EVERY word is a clinical descriptor, not a disease name. If you find a disease name → replace it with the underlying clinical descriptor. Also verify you are NOT using generic symptoms alone — combine specific findings.**

**Query construction:**
- Query 1: Anchor on the most SPECIFIC lab abnormality or imaging finding. Use ONLY 1 AND. Avoid generic terms.
- Query 2: Anchor on a SPECIFIC PHYSICAL EXAM finding or unusual symptom cluster. Use ONLY 1 AND. Pick something DISTINCTIVE.
- Query 3: Anchor on an EXPOSURE or RISK FACTOR + a SPECIFIC clinical finding. Use ONLY 1 AND. Both parts should be specific.
- Query 4: Anchor on the second most distinctive lab or imaging finding (different from Query 1). Use ONLY 1 AND.
- Query 5: Anchor on a broad but distinctive clinical pattern (e.g., organ system involvement, multi-system findings). Use ONLY 1 AND. This is the safety net — it finds diseases that might have been missed.

**CRITICAL RULES:**
- **ZERO disease names. Clinical descriptors ONLY.**
- **MAXIMUM 1 AND per query** — more ANDs cause zero results in PubMed
- **BALANCE specificity**: Use ONE specific finding + ONE moderately broad finding. Avoid combinations so rare they never appear together in literature.
- Use [Title/Abstract] field tags for precision
- Each query under 150 characters
- Do NOT include: age, sex, geography, hospital names, "case report"
- If the case has a striking finding (e.g., "eosinophil count 120,690/mm3"), build at least one query around it
- If a finding has multiple common names, pick the most standard one
- **Query 3 MUST include an exposure or risk factor if present in the case**
- **Query 5 MUST be the most GENERAL query** — use the broadest distinctive finding + a broad category (e.g., "rectal ulcer" + "sexually transmitted" or "skin rash" + "autoimmune"). This is the BROAD NET query that catches diseases you haven't thought of.

## Output Format (JSON only, no markdown)
{{
  "clinical_findings": [
    {{"finding": "CK 5358 U/L with creatinine 3.04 mg/dL (AKI)", "significance": "Indicates severe rhabdomyolysis with renal involvement; narrows to causes of massive muscle breakdown"}},
    {{"finding": "Hoarseness + periorbital edema + weight gain 4.5 kg + fatigue", "significance": "Suggests mucopolysaccharide deposition or fluid retention syndrome; narrows away from primary renal or cardiac causes"}}
  ],
  "pubmed_queries": [
    "(CK 5358 U/L[Title/Abstract]) AND (creatinine 3.04 mg/dL[Title/Abstract])",
    "(hoarseness[Title/Abstract]) AND (periorbital edema[Title/Abstract])",
    "(elevated creatine kinase[Title/Abstract]) AND (myopathy[Title/Abstract])"
  ],
  "search_rationale": "One sentence explaining the multi-angle search strategy"
}}

## Case
{case_text}

JSON:"""


def extract_findings_and_build_queries(case_text: str, max_chars: int = 6000) -> dict:
    """Single LLM call: extract clinical findings + build 3 PubMed queries.

    REPLACES classify_and_build_query(). Instead of guessing a diagnostic domain,
    extracts disease-agnostic clinical findings and builds multi-angle queries from them.

    Returns dict with keys: clinical_findings, pubmed_queries, search_rationale.
    On failure, returns empty findings/queries so callers can degrade gracefully.
    """
    prompt = CLINICAL_FINDINGS_QUERY_PROMPT.format(case_text=case_text[:max_chars])
    try:
        raw = _call_llm(prompt, temperature=0.0, max_tokens=1500)
    except Exception as e:
        print(f"[FindingsExtract] LLM call failed: {e}")
        return {"clinical_findings": [], "pubmed_queries": [], "search_rationale": ""}

    # Parse JSON from LLM output (robust against markdown fences)
    data = _parse_json_from_llm(raw)
    if data is None:
        print(f"[FindingsExtract] JSON parse failed, raw: {raw[:200]}")
        return {"clinical_findings": [], "pubmed_queries": [], "search_rationale": ""}

    queries = data.get("pubmed_queries", [])
    if isinstance(queries, str):
        queries = [queries]

    # Validate and sanitize each query
    valid_queries = []
    for q in queries:
        if not q or len(q) < 10:
            continue
        q = _sanitize_query(q)
        if len(q) >= 10:
            valid_queries.append(q)

    findings_list = data.get("clinical_findings", [])
    print(f"[FindingsExtract] extracted {len(findings_list)} findings, {len(valid_queries)} valid queries")
    for i, q in enumerate(valid_queries[:5]):
        print(f"  query[{i}]: {q[:120]}...")

    return {
        "clinical_findings": findings_list,
        "pubmed_queries": valid_queries[:5],
        "search_rationale": data.get("search_rationale", ""),
    }


def retrieve_multi_query_with_fallback(
    pubmed_client,
    queries: List[str],
    top_k: int = 5,
) -> List:
    """Multi-query PubMed retrieval with progressively relaxed filters AND query relaxation.

    Two-phase relaxation:
    1. Filter relaxation: strict -> medium -> loose filters
    2. Query relaxation: if total results < 3, relax queries by removing AND conditions

    Returns deduplicated sources. Never returns empty if ANY query matches.
    Designed to eliminate the "0-1 results" degradation rate.
    """
    # Phase 0: Expand queries with spelling variants before trying retrieval
    expanded_queries = []
    for query in queries:
        expanded = _expand_query_with_variants(query)
        if expanded != query:
            print(f"[QueryExpand] '{query[:60]}...' -> '{expanded[:60]}...'")
            expanded_queries.append(expanded)
        else:
            expanded_queries.append(query)
    queries = expanded_queries

    # Phase 1: Try original queries with progressively relaxed filters
    filter_levels = [
        ("strict", "humans[Mesh] AND case reports[pt]"),
        ("medium", "humans[Mesh]"),
        ("loose", ""),
    ]

    all_sources = []
    for filter_name, filters in filter_levels:
        for query in queries:
            if not query or len(query) < 10:
                continue
            try:
                strategy = RetrievalStrategy(
                    name=f"cf_{filter_name}",
                    query=query,
                    filters=filters,
                    sort="relevance",
                    goal="P0_case_match",
                    weight=10,
                    max_results=5,
                )
                raw = pubmed_client.retrieve(strategy, top_k=top_k)
                for src in raw:
                    src.strategy_name = f"cf_{filter_name}"
                all_sources.extend(raw)
                if raw:
                    print(f"[MultiQuery] '{query[:60]}...' with {filter_name} filter → {len(raw)} results")
            except Exception as e:
                print(f"[MultiQuery] Query failed: {query[:60]}... → {e}")
                continue

        if len(all_sources) >= 3:
            break

    # Phase 2: Query relaxation — if total results < 3, relax queries by removing ANDs
    if len(all_sources) < 3:
        print(f"[MultiQuery] Only {len(all_sources)} results so far — attempting query relaxation")
        relaxed_queries = _relax_queries(queries)
        for relaxed_q in relaxed_queries:
            if not relaxed_q or len(relaxed_q) < 10:
                continue
            try:
                strategy = RetrievalStrategy(
                    name="cf_relaxed",
                    query=relaxed_q,
                    filters="",  # No filters for relaxed queries
                    sort="relevance",
                    goal="P0_case_match",
                    weight=10,
                    max_results=5,
                )
                raw = pubmed_client.retrieve(strategy, top_k=top_k)
                for src in raw:
                    src.strategy_name = "cf_relaxed"
                all_sources.extend(raw)
                if raw:
                    print(f"[MultiQuery] '{relaxed_q[:60]}...' relaxed → {len(raw)} results")
            except Exception as e:
                print(f"[MultiQuery] Relaxed query failed: {relaxed_q[:60]}... → {e}")
                continue

            if len(all_sources) >= 3:
                break

    if not all_sources:
        print("[MultiQuery] All queries failed — trying last-resort broad search")
        try:
            strategy = RetrievalStrategy(
                name="cf_last_resort",
                query=queries[0] if queries else "rare disease case report",
                filters="",
                sort="relevance",
                goal="P0_case_match",
                weight=10,
                max_results=5,
            )
            raw = pubmed_client.retrieve(strategy, top_k=top_k)
            all_sources.extend(raw)
        except Exception:
            pass

    # Deduplicate and filter (RetrievalOrchestrator is in same module, defined below)
    unique = RetrievalOrchestrator._deduplicate_sources(all_sources)
    filtered = RetrievalOrchestrator._filter_nejm_cpc(unique)
    return list(filtered)[:top_k * 3]


def _relax_queries(queries: List[str]) -> List[str]:
    """Generate relaxed versions of queries by removing AND conditions.

    This is a PURELY STRUCTURAL operation — it modifies the query syntax,
    not the medical content. It reduces the number of AND conditions to
    increase recall in PubMed.

    Examples:
      '("A"[Title/Abstract]) AND ("B"[Title/Abstract]) AND ("C"[Title/Abstract])'
      -> '("A"[Title/Abstract]) AND ("B"[Title/Abstract])'
      -> '("A"[Title/Abstract])'

    Returns a list of progressively relaxed queries, ordered from
    least relaxed to most relaxed.
    """
    relaxed = []
    for query in queries:
        if not query or len(query) < 10:
            continue

        # Count ANDs
        and_count = query.count(" AND ")
        if and_count <= 1:
            # Already minimal, try removing field tags for broader match
            broad = re.sub(r'\[Title/Abstract\]', '', query)
            broad = broad.replace('"', '').strip()
            if broad and len(broad) >= 3:
                relaxed.append(broad)
            continue

        # Strategy 1: Remove last AND condition
        parts = query.split(" AND ")
        if len(parts) >= 3:
            relaxed_query = " AND ".join(parts[:-1])
            if relaxed_query and len(relaxed_query) >= 10:
                relaxed.append(relaxed_query)

        # Strategy 2: Keep only first two parts
        if len(parts) >= 2:
            relaxed_query = " AND ".join(parts[:2])
            if relaxed_query and len(relaxed_query) >= 10 and relaxed_query not in relaxed:
                relaxed.append(relaxed_query)

        # Strategy 3: Keep only the first (most specific) part
        if len(parts) >= 1:
            first_part = parts[0].strip()
            if first_part and len(first_part) >= 10 and first_part not in relaxed:
                relaxed.append(first_part)

    # Remove duplicates while preserving order
    seen = set()
    unique_relaxed = []
    for q in relaxed:
        q_norm = q.lower().strip()
        if q_norm not in seen:
            seen.add(q_norm)
            unique_relaxed.append(q)

    print(f"[QueryRelax] Generated {len(unique_relaxed)} relaxed queries from {len(queries)} original")
    for i, q in enumerate(unique_relaxed[:5]):
        print(f"  relaxed[{i+1}]: {q[:100]}...")

    return unique_relaxed


# =============================================================================
# Rationale-Guided Retrieval (for local models)
# =============================================================================

def generate_diagnosis_rationale(case_text: str, llm_caller=None) -> str:
    """Generate diagnostic hypotheses to guide retrieval (Rationale-Guided Retrieval).

    Instead of using raw case text for keyword extraction, first generate 2-3
    diagnostic hypotheses. These hypotheses are closer to medical literature
    phrasing, improving retrieval relevance.

    Args:
        case_text: Full CPC case presentation text.
        llm_caller: Optional callable(prompt, temperature, max_tokens) -> str.
                    If None, uses the internal _call_llm.

    Returns:
        A string containing 2-3 diagnostic hypotheses.
    """
    if llm_caller is None:
        llm_caller = _call_llm

    prompt = f"""You are a senior physician analyzing a difficult CPC case.
Based on the case presentation, generate 2-3 concise diagnostic hypotheses.
Focus on: what disease mechanism might explain the key findings?

Case (first 2000 chars):
{case_text[:2000]}

Output EXACTLY:
HYPOTHESIS 1: [disease/mechanism, max 15 words]
HYPOTHESIS 2: [disease/mechanism, max 15 words]
HYPOTHESIS 3: [disease/mechanism, max 15 words, or "None"]
"""
    try:
        result = llm_caller(prompt, temperature=0.1, max_tokens=300)
        # Extract hypotheses
        hypotheses = []
        for i in range(1, 4):
            match = re.search(rf'HYPOTHESIS\s*{i}:\s*(.+?)(?:\n|$)', result, re.IGNORECASE)
            if match:
                h = match.group(1).strip()
                if h and h.lower() not in ("none", "n/a", ""):
                    hypotheses.append(h)
        return " ; ".join(hypotheses) if hypotheses else ""
    except Exception as e:
        print(f"[Rationale-Guided Retrieval] Failed to generate rationale: {e}")
        return ""


# =============================================================================
# Perplexity-Based Source Filtering (for local models)
# =============================================================================

def filter_sources_by_relevance_llm(
    sources: List[RetrievedSource],
    case_text: str,
    llm_caller=None,
    max_keep: int = 3
) -> List[RetrievedSource]:
    """Filter retrieved sources by asking the LLM to evaluate relevance.

    This is a lightweight perplexity-proxy: if the LLM says an article is
    NOT directly relevant, we drop it. For local models, this prevents
    noisy literature from entering the prompt.

    Args:
        sources: List of RetrievedSource objects.
        case_text: The case presentation text.
        llm_caller: Optional callable(prompt, temperature, max_tokens) -> str.
        max_keep: Maximum number of sources to keep.

    Returns:
        Filtered list of sources, max `max_keep` items.
    """
    if not sources:
        return []

    if llm_caller is None:
        llm_caller = _call_llm

    filtered = []
    for src in sources:
        prompt = f"""Case excerpt (first 1000 chars):
{case_text[:1000]}

Article:
Title: {src.title}
Abstract: {src.abstract[:300] if src.abstract else 'N/A'}

Is this article DIRECTLY relevant to diagnosing this specific case?
Answer ONLY "Yes" or "No"."""
        try:
            response = llm_caller(prompt, temperature=0.0, max_tokens=10)
            if "yes" in response.lower():
                filtered.append(src)
                if len(filtered) >= max_keep:
                    break
        except Exception as e:
            print(f"[Relevance Filter] Error evaluating '{src.title[:50]}': {e}")
            # On error, keep the source to be safe
            filtered.append(src)
            if len(filtered) >= max_keep:
                break

    print(f"[Relevance Filter] Kept {len(filtered)}/{len(sources)} sources")
    return filtered


# =============================================================================
# LLM Synthesis Helper
# =============================================================================

def synthesize_sources_with_llm(sources: List[RetrievedSource], keywords: List[str]) -> str:
    """Use LLM to synthesize retrieved articles into structured clinical points."""
    if not sources:
        return "No relevant medical literature was retrieved for this case."

    articles_text = ""
    for i, src in enumerate(sources[:6], 1):
        articles_text += "\nArticle " + str(i) + ": " + src.title
        if src.year:
            articles_text += " (" + str(src.year) + ")"
        if src.abstract:
            abstract = src.abstract.replace("\n", " ")
            articles_text += "\n  Abstract: " + abstract[:350]
        if src.key_findings:
            articles_text += "\n  Key findings: " + src.key_findings[:200]
        if src.strategy_name:
            articles_text += "\n  Source strategy: " + src.strategy_name
        articles_text += "\n"

    keyword_line = keywords[0] if keywords else "unknown clinical presentation"

    prompt = f"""You are a senior attending physician preparing for a clinicopathological conference. You have retrieved the following medical literature for a difficult case. Extract STRUCTURED clinical points.

Case presentation focus: {keyword_line}

Retrieved articles:{articles_text}

Instructions (STRICT):
1. For EACH article, extract:
   - DIAGNOSTIC POINTS: 1-2 bullet points on what clinical/lab/imaging findings support the diagnosis discussed.
   - DIFFERENTIAL EXCLUSION: 1 bullet point on what key findings would RULE OUT or argue against this diagnosis.
2. If the articles DISAGREE or describe DIFFERENT diseases, highlight the CONTRAST: "Article X favors [disease A] because...; Article Y favors [disease B] because..."
3. Do NOT simply list article titles. Write as if summarizing for a colleague.
4. Keep total output under 800 characters.
5. Output ONLY bullet points. No preamble, no conclusion.

Synthesis:"""

    try:
        result = _call_llm(prompt, temperature=0.1, max_tokens=4000)
        result = result.strip()
        if result:
            return result
    except Exception as e:
        print(f"[LLM Synthesis Error] {e}")
    return "No relevant medical literature was retrieved for this case."


# =============================================================================
# PubMed Client
# =============================================================================

class PubMedClient:
    # Class-level global rate limiter — shared across ALL instances and workers
    _global_lock = threading.Lock()
    _global_last_request = 0.0

    def __init__(self, cache: CacheManager):
        self.cache = cache

    @classmethod
    def _rate_limit(cls):
        """Global rate limiter: ~8 req/sec shared across all workers (NCBI API key limit is 10/s)."""
        with cls._global_lock:
            elapsed = time.time() - cls._global_last_request
            if elapsed < PUBMED_RATE_LIMIT:
                time.sleep(PUBMED_RATE_LIMIT - elapsed)
            cls._global_last_request = time.time()

    def search(self, query: str, max_results: int = 10, filters: str = "", sort: str = "relevance") -> List[str]:
        """Search PubMed and return list of PMIDs."""
        if not query:
            return []
        self._rate_limit()
        url = f"{PUBMED_BASE_URL}/esearch.fcgi"

        term = query
        if filters:
            term = f"({query}) AND ({filters})"

        params = {
            "db": "pubmed",
            "term": term,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort,
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        try:
            resp = requests.get(url, params=params, timeout=PUBMED_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            idlist = data.get("esearchresult", {}).get("idlist", [])
            return idlist
        except Exception as e:
            print(f"[PubMed Search Error] {e}")
            return []

    def fetch_summaries(self, pmids: List[str]) -> List[Dict]:
        """Fetch article summaries by PMID, including publication types."""
        if not pmids:
            return []
        self._rate_limit()
        url = f"{PUBMED_BASE_URL}/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        try:
            resp = requests.get(url, params=params, timeout=PUBMED_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for pmid in pmids:
                summary = data.get("result", {}).get(pmid, {})
                if summary:
                    pubtypes = []
                    for pt in summary.get("pubtype", []):
                        pubtypes.append(pt.lower())
                    results.append({
                        "pmid": pmid,
                        "title": summary.get("title", ""),
                        "authors": ", ".join([a.get("name", "") for a in summary.get("authors", [])[:3]]),
                        "year": str(summary.get("pubdate", ""))[:4] if summary.get("pubdate") else None,
                        "doi": summary.get("elocationid", "").replace("doi: ", "") if summary.get("elocationid") else None,
                        "pubtypes": pubtypes,
                    })
            return results
        except Exception as e:
            print(f"[PubMed Summary Error] {e}")
            return []

    def fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """Fetch abstracts by PMID."""
        if not pmids:
            return {}
        self._rate_limit()
        url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        try:
            resp = requests.get(url, params=params, timeout=PUBMED_TIMEOUT)
            resp.raise_for_status()
            abstracts = {}
            content = resp.text
            articles = re.findall(r'<PubmedArticle>(.+?)</PubmedArticle>', content, re.DOTALL)
            for article in articles:
                pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
                if pmid_match:
                    pmid = pmid_match.group(1)
                    abs_match = re.search(r'<AbstractText[^>]*>(.+?)</AbstractText>', article, re.DOTALL)
                    if abs_match:
                        abs_text = re.sub(r'<[^>]+>', '', abs_match.group(1))
                        abstracts[pmid] = abs_text
            return abstracts
        except Exception as e:
            print(f"[PubMed Abstract Error] {e}")
            return {}

    def retrieve(self, strategy: RetrievalStrategy, top_k: int = 5) -> List[RetrievedSource]:
        """Full retrieval pipeline for a specific strategy."""
        pmids = self.search(
            strategy.query,
            max_results=strategy.max_results,
            filters=strategy.filters,
            sort=strategy.sort,
        )
        if not pmids:
            return []

        summaries = self.fetch_summaries(pmids[:top_k])
        abstracts = self.fetch_abstracts([s["pmid"] for s in summaries])

        sources = []
        for s in summaries:
            pmid = s["pmid"]
            abstract = abstracts.get(pmid, "")
            abstract_short = abstract[:400] + "..." if len(abstract) > 400 else abstract
            pubtypes = s.get("pubtypes", [])
            pub_type_str = ", ".join(pubtypes) if pubtypes else None

            sources.append(RetrievedSource(
                source="PubMed",
                title=s.get("title", ""),
                authors=s.get("authors", ""),
                year=s.get("year"),
                abstract=abstract_short,
                key_findings=None,
                relevance=None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                strategy_name=strategy.name,
                strategy_goal=strategy.goal,
                pub_type=pub_type_str,
            ))
        return sources


# =============================================================================
# Europe PMC Client
# =============================================================================

class EuropePMCClient:
    def __init__(self, cache: CacheManager):
        self.cache = cache
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < EUROPE_PMC_RATE_LIMIT:
            time.sleep(EUROPE_PMC_RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Europe PMC and return articles with open access full text."""
        if not query:
            return []
        self._rate_limit()
        url = f"{EUROPE_PMC_BASE_URL}/search"
        params = {
            "query": query,
            "pageSize": max_results,
            "format": "json",
            "resultType": "core"
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("resultList", {}).get("result", [])
            oa_results = [r for r in results if r.get("isOpenAccess") == "Y"]
            return oa_results[:max_results]
        except Exception as e:
            print(f"[Europe PMC Error] {e}")
            return []

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedSource]:
        """Retrieve open access articles from Europe PMC."""
        results = self.search(query, max_results=top_k * 2)
        sources = []
        for r in results[:top_k]:
            abstract_text = r.get("abstractText", "")
            abstract_short = abstract_text[:300] + "..." if len(abstract_text) > 300 else abstract_text
            sources.append(RetrievedSource(
                source="EuropePMC",
                title=r.get("title", ""),
                authors=r.get("authorString", ""),
                year=str(r.get("pubYear", "")) if r.get("pubYear") else None,
                abstract=abstract_short,
                key_findings=None,
                relevance=None,
                url=(r.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "") if r.get("fullTextUrlList") else None)
            ))
        return sources


# =============================================================================
# Semantic Scholar Client
# =============================================================================

class SemanticScholarClient:
    def __init__(self, cache: CacheManager):
        self.cache = cache
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < SEMANTIC_SCHOLAR_RATE_LIMIT:
            time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Semantic Scholar for papers."""
        if not query:
            return []
        self._rate_limit()
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,tldr,citationCount,openAccessPdf"
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            print(f"[Semantic Scholar Error] {e}")
            return []

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedSource]:
        """Retrieve papers with TL;DR summaries."""
        results = self.search(query, max_results=top_k * 2)
        results = sorted(results, key=lambda x: x.get("citationCount", 0), reverse=True)
        sources = []
        for r in results[:top_k]:
            tldr = r.get("tldr", {})
            tldr_text = tldr.get("text", "") if tldr else ""
            abstract = r.get("abstract", "") or ""
            key_findings = tldr_text if tldr_text else abstract[:200]
            sources.append(RetrievedSource(
                source="SemanticScholar",
                title=r.get("title", ""),
                authors=", ".join([a.get("name", "") for a in r.get("authors", [])[:3]]),
                year=str(r.get("year", "")) if r.get("year") else None,
                abstract=abstract[:250] + "..." if len(abstract) > 250 else abstract,
                key_findings=key_findings,
                relevance=f"Cited {r.get('citationCount', 0)} times",
                url=r.get("openAccessPdf", {}).get("url", "") if r.get("openAccessPdf") else None
            ))
        return sources


# =============================================================================
# Retrieval Orchestrator
# =============================================================================

class RetrievalOrchestrator:
    def __init__(self, cache_path: str = CACHE_DB_PATH):
        self.cache = CacheManager(cache_path)
        self.pubmed = PubMedClient(self.cache)
        self.europe_pmc = EuropePMCClient(self.cache)
        self.semantic_scholar = SemanticScholarClient(self.cache)
        self._uptodate_client = None  # 懒加载

    def _get_uptodate_client(self):
        """懒加载UpToDate客户端"""
        if self._uptodate_client is None:
            try:
                from uptodate_client import UpToDateClient
                self._uptodate_client = UpToDateClient(
                    max_queries_per_case=1,
                    min_interval=5.0,
                    cache_ttl_hours=24
                )
            except ImportError:
                print("[UpToDate] Client not available. Install playwright and beautifulsoup4 if needed.")
                self._uptodate_client = False  # 标记为不可用
        return self._uptodate_client if self._uptodate_client is not False else None

    def query_uptodate_for_differential(self, diagnoses: List[str], specific_finding: str = "", clinical_context: str = "", case_context: str = "") -> Optional[str]:
        """当需要鉴别相似疾病时查询UpToDate的鉴别诊断功能

        利用UpToDate最强大的鉴别诊断模块，获取权威鉴别诊断知识。
        严格控制频率：每个病例最多2次查询。

        Args:
            diagnoses: 需要鉴别的疾病列表（最多3个）
            specific_finding: 特定的临床表现（可选）
            clinical_context: 临床场景（如"infant", "MSM"等）
            case_context: 完整病例文本，用于智能相关性评分

        Returns:
            UpToDate搜索结果文本，如果失败或达到限制则返回None
        """
        client = self._get_uptodate_client()
        if client is None:
            return None

        if not diagnoses or len(diagnoses) < 2:
            return None

        print(f"[UpToDate] Querying differential for: {', '.join(diagnoses[:3])}")
        return client.search_differential(diagnoses, specific_finding, clinical_context, use_smart=True, case_context=case_context)

    def query_uptodate_clinical_approach(self, presentation: str, population: str = "") -> Optional[str]:
        """查询UpToDate的临床诊断方法

        当面对不典型表现或需要系统评估时，查询UpToDate的Clinical Approach。

        Args:
            presentation: 主要临床表现（如"rectal pain", "developmental regression"）
            population: 特定人群（如"MSM", "infant"）

        Returns:
            UpToDate搜索结果文本
        """
        client = self._get_uptodate_client()
        if client is None:
            return None

        if not presentation:
            return None

        print(f"[UpToDate] Querying clinical approach for: {presentation}")
        return client.search_clinical_approach(presentation, population)

    def query_uptodate_diagnostic_criteria(self, disease: str, key_feature: str = "") -> Optional[str]:
        """查询UpToDate的诊断标准

        当需要确认某个疾病的诊断标准或关键鉴别特征时查询。

        Args:
            disease: 疾病名称
            key_feature: 关键特征（可选）

        Returns:
            UpToDate搜索结果文本
        """
        client = self._get_uptodate_client()
        if client is None:
            return None

        if not disease:
            return None

        print(f"[UpToDate] Querying diagnostic criteria for: {disease}")
        return client.search_diagnostic_criteria(disease, key_feature)

    def query_uptodate_for_finding(self, finding: str, context: str = "") -> Optional[str]:
        """查询特定临床发现的鉴别诊断意义

        当检测到未解释的特异性发现时，查询UpToDate获取相关知识。

        Args:
            finding: 临床发现（如"startle response"）
            context: 上下文（如"infant"）

        Returns:
            UpToDate搜索结果文本
        """
        client = self._get_uptodate_client()
        if client is None:
            return None

        if not finding:
            return None

        print(f"[UpToDate] Querying finding: {finding}")
        return client.search_finding(finding, context)

    def should_query_uptodate(self, pass1_candidates: List[Dict], case_text: str) -> bool:
        """判断是否需要查询UpToDate — 完全通用，不针对任何特定疾病类别

        核心原则：当系统面临"多个候选诊断难以区分"的困境时，查询权威知识库。
        触发条件基于诊断的通用特征，而非特定疾病名称。

        Args:
            pass1_candidates: Pass 1的候选诊断列表
            case_text: 病例文本

        Returns:
            是否需要查询UpToDate
        """
        if not pass1_candidates or len(pass1_candidates) < 2:
            return False

        top2 = pass1_candidates[:2]
        diag1 = str(top2[0].get("disease", "")).lower().strip()
        diag2 = str(top2[1].get("disease", "")).lower().strip()

        if not diag1 or not diag2:
            return False

        # 通用条件1：前2个诊断共享相同的疾病机制或解剖部位关键词
        # 提取诊断名称中的医学实体（去掉常见修饰词）
        def _extract_core_terms(diagnosis: str) -> set:
            """提取诊断名称中的核心医学术语"""
            # 移除常见修饰词
            modifiers = [
                "acute", "chronic", "severe", "mild", "moderate", "progressive",
                "primary", "secondary", "idiopathic", "familial", "hereditary",
                "infantile", "juvenile", "adult-onset", "late-onset", "early-onset",
                "type i", "type ii", "type iii", "type 1", "type 2", "type 3",
                "variant", "form", "syndrome", "disease", "disorder", "condition"
            ]
            cleaned = diagnosis.lower()
            for mod in modifiers:
                cleaned = cleaned.replace(mod, "")
            # 提取有意义的词（长度>3，避免介词等）
            words = [w.strip() for w in cleaned.split() if len(w.strip()) > 3]
            return set(words)

        core1 = _extract_core_terms(diag1)
        core2 = _extract_core_terms(diag2)

        # 如果核心术语有重叠，说明是同类疾病难以区分
        if core1 and core2:
            overlap = core1.intersection(core2)
            if overlap:
                print(f"[UpToDate] Trigger: top 2 diagnoses share core terms: {overlap}")
                return True

        # 通用条件2：前2个诊断的置信度都很接近且不高（系统不确定）
        conf1 = str(top2[0].get("confidence", "")).upper()
        conf2 = str(top2[1].get("confidence", "")).upper()
        # 如果前2个都是MEDIUM或LOW，说明系统不确定
        if conf1 in ("MEDIUM", "LOW") and conf2 in ("MEDIUM", "LOW"):
            print("[UpToDate] Trigger: top 2 diagnoses have low/medium confidence")
            return True

        # 通用条件3：病例文本明确提到鉴别诊断困难或需要排除多种疾病
        case_lower = case_text.lower()
        uncertainty_indicators = [
            "differential diagnosis", "difficult to distinguish", "must be differentiated",
            "cannot be distinguished", "similar presentation", "overlap with",
            "mimics", "can present with", "must be excluded"
        ]
        if any(indicator in case_lower for indicator in uncertainty_indicators):
            print("[UpToDate] Trigger: case text indicates diagnostic uncertainty")
            return True

        return False

    def retrieve_for_case(
        self,
        case_id: str,
        case_text: str,
        preliminary_dx: List[str] = None,
        use_pubmed: bool = True,
        use_europe_pmc: bool = True,
        use_semantic_scholar: bool = False,
        top_k: int = 5,
        force_refresh: bool = False,
        additional_context: str = "",
        use_rationale_guided: bool = True,
        use_relevance_filter: bool = True,
    ) -> RetrievalResult:
        """
        Main entry point: retrieve medical literature for a given case using
        structured keyword extraction and multi-strategy retrieval.
        Args:
            additional_context: Supplementary text (e.g., clinical questions) to
                               enrich LLM-based keyword extraction.
            use_rationale_guided: If True, generate diagnostic hypotheses first
                                 to guide keyword extraction (local model opt).
            use_relevance_filter: If True, post-filter sources with LLM relevance
                                 check (local model opt).
        """
        # Check cache
        if not force_refresh:
            cached = self.cache.get(case_id)
            if cached:
                print(f"[Cache Hit] {case_id}")
                return cached

        start_time = time.time()

        # LOCAL MODEL OPT: Rationale-Guided Retrieval
        # Generate diagnostic hypotheses first, then use them to enrich keyword extraction
        rationale_context = additional_context
        if use_rationale_guided and ENABLE_LLM_KEYWORDS and LLM_API_KEY:
            try:
                rationale = generate_diagnosis_rationale(case_text, llm_caller=_call_llm)
                if rationale:
                    print(f"[{case_id}] Generated rationale: {rationale[:120]}...")
                    if rationale_context:
                        rationale_context = rationale_context + " ; " + rationale
                    else:
                        rationale_context = rationale
            except Exception as e:
                print(f"[{case_id}] Rationale generation failed: {e}")

        # Step 1: Structured keyword extraction
        sk = extract_structured_keywords_from_case(case_text, case_id=case_id,
                                                    additional_context=rationale_context)
        print(f"[{case_id}] Structured keywords: {len(sk.core)} core")
        for c in sk.core[:3]:
            tag = f"[{c.category or '?'}]"
            mesh = f" -> {c.mesh_suggestion}" if c.mesh_suggestion else ""
            print(f"  {tag} {c.term}{mesh}")

        # Step 2: Build query matrix
        strategies = build_query_matrix(sk)

        # NEW: When preliminary_dx is explicitly provided (e.g., for differential retrieval),
        # add direct disease-name search strategies. This is critical for diseases that
        # may not appear in keyword extraction (e.g., rare diseases, specific infections).
        if preliminary_dx and len(preliminary_dx) > 0:
            for dx in preliminary_dx:
                if not dx or len(dx.strip()) < 3:
                    continue
                # Clean the disease name
                clean_dx = dx.strip().rstrip('.').rstrip(',')
                # Remove parenthetical content for cleaner search
                clean_dx = re.sub(r'\s*\([^)]*\)', '', clean_dx).strip()
                if len(clean_dx) < 3:
                    continue

                # Add direct search strategy for this disease
                direct_query = f'"{clean_dx}"[Title/Abstract]'
                # Also try with broader terms (e.g., "Syphilitic proctitis" -> "Syphilis" AND "proctitis")
                words = clean_dx.split()
                if len(words) >= 2:
                    broad_query = f'"{words[0]}"[Title/Abstract] AND "{" ".join(words[1:])}"[Title/Abstract]'
                else:
                    broad_query = direct_query

                strategies.insert(0, RetrievalStrategy(
                    name=f"direct_dx_{clean_dx[:20]}",
                    query=direct_query,
                    filters="",
                    sort="relevance",
                    goal="P0_direct_disease",
                    weight=2.0,  # Higher weight for direct disease search
                    max_results=10,
                ))
                if broad_query != direct_query:
                    strategies.insert(1, RetrievalStrategy(
                        name=f"broad_dx_{clean_dx[:20]}",
                        query=broad_query,
                        filters="",
                        sort="relevance",
                        goal="P0_broad_disease",
                        weight=1.5,
                        max_results=10,
                    ))
                print(f"[{case_id}] Added direct disease search for: '{clean_dx}'")

        if not strategies:
            # Fallback to legacy single query
            legacy_kws = extract_keywords_from_case(case_text, max_keywords=5)
            legacy_query = build_pubmed_query(legacy_kws)
            strategies = [RetrievalStrategy(
                name="legacy",
                query=legacy_query,
                filters="",
                sort="relevance",
                goal="P0_case_match",
                weight=STRATEGY_WEIGHTS["legacy"],
                max_results=10,
            )]

        for s in strategies:
            print(f"[{case_id}] Strategy '{s.name}': {s.query}")

        all_sources = []
        errors = []
        queries_executed = []
        strategy_names = []

        # CONSERVATIVE: Serial execution of strategies to avoid PubMed rate-limit bursts
        # Step 3: Execute each strategy with cost control (single-threaded)
        strategy_a_success = False

        for i, strategy in enumerate(strategies):
            if not use_pubmed or not strategy.query:
                continue

            # Cost control: if Strategy A got good results, reduce B/C depth
            if i > 0 and strategy_a_success:
                adjusted_top_k = min(top_k, 3)
                adjusted_max_results = min(strategy.max_results, 5)
            else:
                adjusted_top_k = top_k
                adjusted_max_results = strategy.max_results

            try:
                sources = self.pubmed.retrieve(strategy, top_k=adjusted_top_k)
                for src in sources:
                    src.strategy_name = strategy.name
                    src.strategy_goal = strategy.goal
                print(f"[{case_id}] Strategy '{strategy.name}': {len(sources)} sources")
                if sources:
                    all_sources.extend(sources)
                    queries_executed.append(strategy.query)
                    strategy_names.append(strategy.name)
                    if i == 0 and len(sources) >= 3:
                        strategy_a_success = True
            except Exception as e:
                error_msg = f"PubMed/{strategy.name}: {e}"
                errors.append(error_msg)
                queries_executed.append(strategy.query)
                print(f"[{case_id}] Strategy '{strategy.name}' failed: {e}")

        # Europe PMC fallback (use first strategy query) - run in parallel with PubMed
        europe_pmc_sources = []
        if use_europe_pmc and strategies:
            try:
                legacy_query = strategies[0].query
                sources = self.europe_pmc.retrieve(legacy_query, top_k=top_k)
                for src in sources:
                    src.strategy_name = strategies[0].name
                    src.strategy_goal = strategies[0].goal
                europe_pmc_sources = sources
                print(f"[{case_id}] EuropePMC: {len(sources)} sources")
            except Exception as e:
                errors.append(f"EuropePMC: {e}")
                print(f"[{case_id}] EuropePMC failed: {e}")

        all_sources.extend(europe_pmc_sources)

        # Semantic Scholar
        if use_semantic_scholar and sk.core:
            try:
                ss_query = " ".join([k.term for k in sk.core[:3]])
                sources = self.semantic_scholar.retrieve(ss_query, top_k=top_k - 2)
                for src in sources:
                    src.strategy_name = "semantic"
                    src.strategy_goal = "P2_differential"
                all_sources.extend(sources)
                print(f"[{case_id}] SemanticScholar: {len(sources)} sources")
            except Exception as e:
                errors.append(f"SemanticScholar: {e}")
                print(f"[{case_id}] SemanticScholar failed: {e}")

        # ENHANCED: Multi-level fallback for local models
        # Level 1: Simplified keywords with no filters
        if not all_sources and sk.core:
            print(f"[{case_id}] Primary retrieval failed, trying Level 1 fallback...")
            # Extract simple terms (max 3 words) from core keywords
            simple_terms = []
            for item in sk.core:
                words = item.term.split()
                # Skip articles/prepositions, keep first 2-3 meaningful words
                skip = {'the', 'a', 'an', 'in', 'with', 'of', 'and', 'for'}
                meaningful = [w for w in words if w.lower() not in skip]
                if meaningful:
                    simple = ' '.join(meaningful[:3])
                    if len(simple) >= 3 and simple not in simple_terms:
                        simple_terms.append(simple)

            if simple_terms:
                fallback_query = ' OR '.join([f'"{t}"[Title/Abstract]' for t in simple_terms[:3]])
                print(f"[{case_id}] Level 1 fallback query: {fallback_query}")
                try:
                    fallback_strategy = RetrievalStrategy(
                        name="fallback",
                        query=fallback_query,
                        filters="",  # No filters for maximum recall
                        sort="relevance",
                        goal="P0_case_match",
                        weight=5,
                        max_results=10,
                    )
                    sources = self.pubmed.retrieve(fallback_strategy, top_k=top_k)
                    for src in sources:
                        src.strategy_name = "fallback"
                        src.strategy_goal = "P0_case_match"
                    all_sources.extend(sources)
                    queries_executed.append(fallback_query)
                    strategy_names.append("fallback")
                    print(f"[{case_id}] Level 1 fallback: {len(sources)} sources")
                except Exception as e:
                    print(f"[{case_id}] Level 1 fallback failed: {e}")

        # Level 2: Use only the most generic MeSH terms if available
        if not all_sources and sk.core:
            print(f"[{case_id}] Level 1 failed, trying Level 2 fallback (MeSH only)...")
            mesh_terms = []
            for item in sk.core:
                if item.mesh_suggestion:
                    # Take only the first/main MeSH term
                    main_mesh = item.mesh_suggestion.split(';')[0].strip()
                    if main_mesh and main_mesh not in mesh_terms:
                        mesh_terms.append(main_mesh)

            if mesh_terms:
                mesh_query = ' OR '.join([f'"{m}"[Mesh]' for m in mesh_terms[:2]])
                print(f"[{case_id}] Level 2 fallback query: {mesh_query}")
                try:
                    mesh_strategy = RetrievalStrategy(
                        name="fallback_mesh",
                        query=mesh_query,
                        filters="humans[Mesh]",
                        sort="relevance",
                        goal="P0_case_match",
                        weight=3,
                        max_results=10,
                    )
                    sources = self.pubmed.retrieve(mesh_strategy, top_k=top_k)
                    for src in sources:
                        src.strategy_name = "fallback_mesh"
                        src.strategy_goal = "P0_case_match"
                    all_sources.extend(sources)
                    queries_executed.append(mesh_query)
                    strategy_names.append("fallback_mesh")
                    print(f"[{case_id}] Level 2 fallback: {len(sources)} sources")
                except Exception as e:
                    print(f"[{case_id}] Level 2 fallback failed: {e}")
        
        # Step 4: Deduplicate, filter NEJM CPC (prevent data leakage), score, rank, and filter
        unique_sources = self._deduplicate_sources(all_sources)
        unique_sources = self._filter_nejm_cpc(unique_sources)
        strategy_weight_map = {s.name: s.weight for s in strategies if s.name in strategy_names}
        ranked_sources = self._rank_sources_by_relevance(unique_sources, sk, strategy_weight_map)
        # P1: Post-retrieval filtering — drop sources that don't match any core keyword
        # RELAXED: If we only have fallback sources, be less strict with filtering
        if any(src.strategy_name == "fallback" for src in ranked_sources):
            # Keep more sources when using fallback
            filtered_sources = ranked_sources[:top_k * 2]
            print(f"[{case_id}] Using relaxed filtering for fallback sources")
        else:
            filtered_sources = self._filter_sources_by_relevance(ranked_sources, sk, case_text)

        # LOCAL MODEL OPTIMIZATION: Hard cap on total sources to prevent context overload
        final_sources = filtered_sources[:min(top_k, MAX_TOTAL_SOURCES_LOCAL)]
        if len(filtered_sources) > MAX_TOTAL_SOURCES_LOCAL:
            print(f"[{case_id}] Truncated sources from {len(filtered_sources)} to {MAX_TOTAL_SOURCES_LOCAL} (local model limit)")

        # LOCAL MODEL OPTIMIZATION: Perplexity-proxy relevance filtering
        # Ask the LLM to evaluate each source's relevance and drop irrelevant ones
        if use_relevance_filter and ENABLE_LLM_KEYWORDS and LLM_API_KEY and len(final_sources) > 1:
            try:
                final_sources = filter_sources_by_relevance_llm(
                    final_sources, case_text, llm_caller=_call_llm, max_keep=MAX_TOTAL_SOURCES_LOCAL
                )
            except Exception as e:
                print(f"[{case_id}] Relevance filtering failed: {e}, keeping original sources")

        # Source breakdown for logging
        breakdown = {}
        for src in all_sources:
            key = f"{src.source}/{src.strategy_name or 'unknown'}"
            breakdown[key] = breakdown.get(key, 0) + 1

        # HERMES-INSPIRED OPTIMIZATION: Apply retrieval optimization pipeline
        # 1. Relevance filtering 2. Content compression 3. Context length control
        # =========================================================================
        try:
            from retrieval_optimizer import optimize_retrieval
            
            # Convert sources to dict format for optimizer
            raw_sources = []
            for src in final_sources:
                raw_sources.append({
                    "title": src.title,
                    "url": getattr(src, 'url', ''),
                    "content": getattr(src, 'abstract', '') or getattr(src, 'content', ''),
                    "source_type": getattr(src, 'source_type', 'pubmed'),
                    "strategy_name": getattr(src, 'strategy_name', ''),
                })
            
            # Apply optimization
            optimized_context, optimized_sources = optimize_retrieval(
                raw_sources=raw_sources,
                case_text=case_text,
                case_id=case_id
            )
            
            # Update final_sources with optimized versions
            if optimized_sources:
                optimized_final = []
                for opt_src in optimized_sources:
                    # Find matching original source
                    matching = None
                    for orig in final_sources:
                        if orig.title == opt_src.title:
                            matching = orig
                            break
                    
                    if matching:
                        # Update with optimized content
                        matching.abstract = opt_src.summary if opt_src.summary else opt_src.content
                        matching.relevance_score = opt_src.relevance_score
                        optimized_final.append(matching)
                    else:
                        optimized_final.append(opt_src)
                
                final_sources = optimized_final
                print(f"[RetrievalModule] {case_id} Hermes optimization: {len(optimized_sources)} sources retained")
                
                # Use optimized context directly if available
                if optimized_context:
                    formatted_text = optimized_context
                else:
                    flat_keywords = [k.term for k in sk.core]
                    formatted_text = self._format_sources(final_sources, flat_keywords)
            else:
                flat_keywords = [k.term for k in sk.core]
                formatted_text = self._format_sources(final_sources, flat_keywords)
                
        except ImportError:
            print(f"[RetrievalModule] {case_id} retrieval_optimizer not available, using original formatting")
            flat_keywords = [k.term for k in sk.core]
            formatted_text = self._format_sources(final_sources, flat_keywords)
        except Exception as e:
            print(f"[RetrievalModule] {case_id} Optimization failed: {e}, using original formatting")
            flat_keywords = [k.term for k in sk.core]
            formatted_text = self._format_sources(final_sources, flat_keywords)

        retrieval_time = time.time() - start_time

        result = RetrievalResult(
            case_id=case_id,
            query=" | ".join(queries_executed) if queries_executed else "",
            sources=final_sources,
            formatted_text=formatted_text,
            retrieval_time=retrieval_time,
            success=len(final_sources) > 0,
            error_message="; ".join(errors) if errors else None,
            queries=queries_executed,
            strategies=strategy_names,
            source_breakdown=breakdown,
        )

        self.cache.set(case_id, result)
        return result

    def retrieve_negative_signals(
        self,
        prelim_dx: List[str],
        top_k: int = 2
    ) -> List[RetrievedSource]:
        """Retrieve articles about mimics and atypical presentations for preliminary diagnoses.
        Backward-compatible method from original MDT.
        """
        if not prelim_dx:
            return []

        negative_sources = []
        for dx in prelim_dx[:2]:
            if not dx or len(dx) < 3:
                continue
            clean_dx = dx.replace('"', '').strip()
            if not clean_dx:
                continue

            query1 = f'"{clean_dx}" mimics differential diagnosis rare case report'
            query2 = f'"{clean_dx}" atypical presentation unusual mimic case report'

            try:
                # Build a simple strategy for the query
                strategy1 = RetrievalStrategy(
                    name="negative_mimic",
                    query=query1,
                    filters="",
                    sort="relevance",
                    goal="P2_differential",
                    weight=3,
                    max_results=top_k * 2,
                )
                sources1 = self.pubmed.retrieve(strategy1, top_k=top_k)
                negative_sources.extend(sources1)
                print(f"[NegativeRetrieval] '{clean_dx}' mimic query: {len(sources1)} sources")
            except Exception as e:
                print(f"[NegativeRetrieval] PubMed query1 failed for '{clean_dx}': {e}")

            try:
                strategy2 = RetrievalStrategy(
                    name="negative_atypical",
                    query=query2,
                    filters="",
                    sort="relevance",
                    goal="P2_differential",
                    weight=3,
                    max_results=top_k * 2,
                )
                sources2 = self.pubmed.retrieve(strategy2, top_k=top_k)
                negative_sources.extend(sources2)
                print(f"[NegativeRetrieval] '{clean_dx}' atypical query: {len(sources2)} sources")
            except Exception as e:
                print(f"[NegativeRetrieval] PubMed query2 failed for '{clean_dx}': {e}")

        unique = self._deduplicate_sources(negative_sources)
        unique = self._filter_nejm_cpc(unique)
        return unique[:4]

    @staticmethod
    def _deduplicate_sources(sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Remove duplicate articles by normalized title."""
        seen = set()
        unique = []
        for src in sources:
            key = src.title.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(src)
        return unique

    _NEJM_CPC_PATTERNS = [
        r'case\s+\d{1,3}\s*[-–]\s*\d{4}',
        r'case records of the massachusetts general hospital',
        r'clinicopathological conference',
    ]

    @classmethod
    def _filter_nejm_cpc(cls, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        filtered = []
        removed_count = 0
        for src in sources:
            title_lower = (src.title or "").lower()
            is_cpc = any(re.search(pat, title_lower) for pat in cls._NEJM_CPC_PATTERNS)
            if is_cpc:
                removed_count += 1
                print(f"[Filter-CPC] Removed test-set article: {src.title[:80]}...")
            else:
                filtered.append(src)
        if removed_count:
            print(f"[Filter-CPC] Removed {removed_count} CPC articles to prevent data leakage")
        return filtered

    @staticmethod
    def _rank_sources_by_relevance(
        sources: List[RetrievedSource],
        keywords: StructuredKeywords,
        strategy_weights: Optional[Dict[str, int]] = None,
    ) -> List[RetrievedSource]:
        """Score sources by strategy source + evidence level + keyword relevance + signals."""
        if not keywords or not keywords.core:
            return sources

        strategy_weights = strategy_weights or STRATEGY_WEIGHTS

        def _score(src: RetrievedSource) -> int:
            text = f"{src.title or ''} {src.abstract or ''}"
            text_lower = text.lower()
            title_lower = (src.title or "").lower()
            score = 0

            # 1. Strategy source weight
            if src.strategy_name and src.strategy_name in strategy_weights:
                score += strategy_weights[src.strategy_name]

            # 2. Evidence level scoring from pub_type
            pub_type_lower = (src.pub_type or "").lower()
            if pub_type_lower:
                for level, signals in EVIDENCE_LEVEL_SCORES.items():
                    for signal, bonus in signals.items():
                        if signal in pub_type_lower:
                            score += bonus
                            break

            # 3. Keyword relevance (core weighted higher than differential)
            all_items = keywords.core + keywords.differential
            for item in all_items:
                kw_lower = item.term.lower()
                title_hits = title_lower.count(kw_lower)
                abstract_hits = text_lower.count(kw_lower) - title_hits
                weight = 3 if item in keywords.core else 2
                score += title_hits * weight + abstract_hits * 1

            # 4. Exclude term soft penalty
            for ex in keywords.exclude:
                ex_lower = ex.term.lower()
                if ex_lower in title_lower:
                    score -= 5
                elif ex_lower in text_lower:
                    score -= 2

            # 5. Rare/differential diagnosis signals
            for signal, bonus in RARE_SIGNALS.items():
                if signal in title_lower:
                    score += bonus * 2
                elif signal in text_lower:
                    score += bonus

            # 6. Penalty for common review/epidemiology
            for signal, penalty in PENALTY_SIGNALS.items():
                if signal in title_lower:
                    score += penalty * 2
                elif signal in text_lower:
                    score += penalty

            # 7. Boost case reports and original articles (title-based)
            if "case report" in title_lower or "case series" in title_lower:
                score += 4
            if "original article" in title_lower or "clinical report" in title_lower:
                score += 2

            return score

        return sorted(sources, key=_score, reverse=True)

    def _filter_sources_by_relevance(
        self,
        sources: List[RetrievedSource],
        keywords: StructuredKeywords,
        case_text: str = "",
    ) -> List[RetrievedSource]:
        """Post-retrieval filtering: remove sources whose titles do not contain
        any core keyword.  This is a fast heuristic to catch obviously irrelevant
        articles (e.g. brainstem glioma retrieved for a renal mass case).
        """
        if not keywords or not keywords.core:
            return sources

        core_terms = [k.term.lower() for k in keywords.core if k.term]
        # Also include mesh suggestions as valid terms
        mesh_terms = [k.mesh_suggestion.lower() for k in keywords.core if k.mesh_suggestion]
        all_valid_terms = core_terms + mesh_terms

        filtered = []
        for src in sources:
            title_lower = (src.title or "").lower()
            abstract_lower = (src.abstract or "").lower()

            # Check if any core keyword appears in title or abstract
            matched = False
            for term in all_valid_terms:
                # Allow partial match for multi-word terms (match any word)
                term_words = term.split()
                if len(term_words) > 1:
                    # For multi-word terms, check if ANY word appears
                    # HERMES FIX: Use word boundary matching to avoid partial word matches
                    # e.g., "edema" should not match "academic" or "pediatric"
                    if any(
                        re.search(r'\b' + re.escape(w) + r'\b', title_lower) or
                        re.search(r'\b' + re.escape(w) + r'\b', abstract_lower)
                        for w in term_words if len(w) > 3
                    ):
                        matched = True
                        break
                else:
                    # HERMES FIX: Use word boundary matching for single-word terms too
                    if re.search(r'\b' + re.escape(term) + r'\b', title_lower) or \
                       re.search(r'\b' + re.escape(term) + r'\b', abstract_lower):
                        matched = True
                        break

            if matched:
                filtered.append(src)
            else:
                safe_title = src.title[:80].encode('ascii', 'replace').decode('ascii')
                print(f"[Filter] Dropped irrelevant source: {safe_title}...")

        # If filtering removes everything, keep the top source as fallback
        if not filtered and sources:
            print("[Filter] All sources filtered out, keeping top-ranked as fallback")
            filtered = sources[:1]

        return filtered

    def _format_sources(self, sources: List[RetrievedSource], keywords: Optional[List[str]] = None) -> str:
        """Format retrieved sources for prompt injection.
        If ENABLE_LLM_SYNTHESIS is on and an LLM key is available, synthesize
        the articles into clinically relevant bullet points instead of raw listing.
        """
        if not sources:
            return "No relevant medical literature was retrieved for this case."

        # Try LLM synthesis for higher quality
        if ENABLE_LLM_SYNTHESIS and LLM_API_KEY:
            try:
                synthesis = synthesize_sources_with_llm(sources, keywords or [])
                # Append top 2 source citations for traceability
                citations = []
                for src in sources[:2]:
                    cite = f"- {src.title}"
                    if src.year:
                        cite += f" ({src.year})"
                    if src.strategy_name:
                        cite += f" [{src.strategy_name}]"
                    citations.append(cite)
                if citations:
                    synthesis += "\n\nKey references:\n" + "\n".join(citations)
                return synthesis
            except Exception as e:
                print(f"[Format] LLM synthesis failed ({e}), falling back to raw format")

        # Fallback: raw template formatting (limit to 1200 chars)
        lines = []
        total_chars = 0
        max_chars = 1200

        for i, src in enumerate(sources, 1):
            block = f"### Source {i}: {src.source} - {src.title}\n"
            if src.year:
                block += f"Year: {src.year}\n"
            if src.authors:
                block += f"Authors: {src.authors}\n"
            if src.abstract:
                block += f"Abstract: {src.abstract}\n"
            if src.key_findings:
                block += f"Key Findings: {src.key_findings}\n"
            if src.relevance:
                block += f"Relevance: {src.relevance}\n"
            if src.strategy_name:
                block += f"Search strategy: {src.strategy_name}\n"
            block += "\n"

            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 50:
                    block = block[:remaining] + "...\n\n"
                    lines.append(block)
                break

            lines.append(block)
            total_chars += len(block)

        return "".join(lines)


# =============================================================================
# Hybrid Retrieval + MedCPT Reranker (optional, for Scheme B)
# =============================================================================

class MedCPTReranker:
    """
    Optional dense reranker using NCBI's MedCPT models.

    If transformers/torch are unavailable, initialization fails gracefully and
    `rerank` returns the original sources unchanged.
    """

    DEFAULT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
    DEFAULT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"

    def __init__(
        self,
        query_encoder_name: Optional[str] = None,
        article_encoder_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.query_encoder_name = query_encoder_name or self.DEFAULT_QUERY_ENCODER
        self.article_encoder_name = article_encoder_name or self.DEFAULT_ARTICLE_ENCODER
        self.device = device
        self._tokenizer = None
        self._query_encoder = None
        self._article_encoder = None
        self._mode = "uninitialized"  # 'medcpt' or 'disabled'

    def _load(self) -> bool:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            print(f"[MedCPT] transformers/torch unavailable: {e}. Dense reranking disabled.")
            return False

        try:
            print(f"[MedCPT] Loading query encoder: {self.query_encoder_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.query_encoder_name)
            self._query_encoder = AutoModel.from_pretrained(self.query_encoder_name)
            self._article_encoder = AutoModel.from_pretrained(self.article_encoder_name)

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._query_encoder.to(device)
            self._article_encoder.to(device)
            self._query_encoder.eval()
            self._article_encoder.eval()
            self._device = device
            self._mode = "medcpt"
            print(f"[MedCPT] Encoders loaded on {device}")
            return True
        except Exception as e:
            print(f"[MedCPT] Failed to load encoders: {e}. Dense reranking disabled.")
            return False

    def _ensure_initialized(self):
        if self._mode == "uninitialized":
            if not self._load():
                self._mode = "disabled"

    def _encode(self, texts: List[str], model: str = "article", max_length: int = 512):
        import torch

        if model == "query":
            self._model = self._query_encoder
        else:
            self._model = self._article_encoder

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def rerank(
        self,
        query: str,
        sources: List[RetrievedSource],
        top_k: Optional[int] = None,
    ) -> List[RetrievedSource]:
        """
        Rerank sources by cosine similarity between query embedding and article embeddings.
        Falls back to original order if MedCPT is unavailable.
        """
        self._ensure_initialized()
        if self._mode != "medcpt" or not sources:
            return sources

        try:
            import numpy as np

            article_texts = []
            for src in sources:
                text = f"{src.title or ''} {src.abstract or ''}".strip()
                if not text:
                    text = src.title or "untitled"
                article_texts.append(text[:512])

            # Encode query with query encoder, articles with article encoder
            q_emb = self._encode([query], model="query", max_length=64)
            self._model = self._article_encoder
            a_emb = self._encode(article_texts, model="article", max_length=512)
            self._model = None

            scores = (q_emb @ a_emb.T)[0]
            ranked = sorted(
                zip(sources, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            result = [src for src, _ in ranked]
            if top_k:
                result = result[:top_k]
            return result
        except Exception as e:
            print(f"[MedCPT] Reranking failed: {e}")
            return sources

    def _encode(self, texts: List[str], model: str = "article", max_length: int = 512):
        import torch

        if model == "query":
            self._model = self._query_encoder
        else:
            self._model = self._article_encoder

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class HybridRetriever:
    """
    Hybrid retriever for Scheme B: sparse (BM25/PubMed) recall + optional dense (MedCPT) rerank.

    This is intentionally lightweight compared to RetrievalOrchestrator.retrieve_for_case.
    """

    def __init__(
        self,
        pubmed_client: Optional[PubMedClient] = None,
        cache: Optional[CacheManager] = None,
        enable_dense_rerank: bool = True,
    ):
        self.cache = cache or CacheManager(CACHE_DB_PATH)
        self.pubmed = pubmed_client or PubMedClient(self.cache)
        self.reranker = MedCPTReranker() if enable_dense_rerank else None

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
    ) -> List[RetrievedSource]:
        """
        Retrieve sources using multiple queries, deduplicate, and optionally rerank.

        Args:
            queries: list of PubMed queries
            top_k: number of sources to return after sparse retrieval + dedup
            rerank_top_k: if dense reranker is available, rerank and keep top N

        Returns:
            List of RetrievedSource objects.
        """
        raw_sources = retrieve_multi_query_with_fallback(
            self.pubmed,
            queries,
            top_k=top_k,
        )

        if self.reranker and raw_sources:
            # Use the first query as the representative query for reranking
            query_text = queries[0] if queries else ""
            # Strip PubMed field tags for plain-text reranking
            clean_query = re.sub(r'\[Title/Abstract\]|\[Mesh\]|\[pt\]|\[Mesh\]', '', query_text)
            clean_query = re.sub(r'["()]', '', clean_query).strip()
            reranked = self.reranker.rerank(clean_query, raw_sources, top_k=rerank_top_k or top_k)
            return reranked

        return raw_sources


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demo the retrieval module with a real case from the dataset."""
    import json
    dataset_path = "D:/VscodeProjects/BaiduSyncdisk/MDT/data/mgh_qa_dataset.json"
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        case = dataset[0]
        case_id = case["case_id"]
        case_text = case["Q"]
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return

    orchestrator = RetrievalOrchestrator()
    result = orchestrator.retrieve_for_case(case_id, case_text, force_refresh=True)

    print(f"\n{'='*60}")
    print(f"Retrieval Result for {case_id}")
    print(f"Success: {result.success}")
    print(f"Time: {result.retrieval_time:.2f}s")
    print(f"Sources: {len(result.sources)}")
    print(f"Strategies: {result.strategies}")
    print(f"Source breakdown: {result.source_breakdown}")
    print(f"Queries: {result.queries}")
    print(f"\nFormatted Text (first 800 chars):\n{result.formatted_text[:800]}")


def demo_structured():
    """Demo the structured extraction without full retrieval (no API calls)."""
    import json
    dataset_path = "D:/VscodeProjects/BaiduSyncdisk/MDT/data/mgh_qa_dataset.json"
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return

    print("=" * 70)
    print("STRUCTURED KEYWORD EXTRACTION DEMO (no PubMed calls)")
    print("=" * 70)

    for case in dataset[:3]:
        case_id = case["case_id"]
        case_text = case["Q"]
        gold = case["A"]["gold_standard_diagnosis"]

        print(f"\n--- {case_id} ---")
        print(f"Gold diagnosis: {gold}")

        sk = extract_structured_keywords_from_case(case_text)
        print(f"\nCore keywords ({len(sk.core)}):")
        for item in sk.core:
            mesh = f" [MeSH: {item.mesh_suggestion}]" if item.mesh_suggestion else ""
            print(f"  - [{item.category or '?'}] {item.term}{mesh}")

        if sk.differential:
            print(f"\nDifferential keywords ({len(sk.differential)}):")
            for item in sk.differential:
                print(f"  - [{item.category or '?'}] {item.term}")

        if sk.exclude:
            print(f"\nExclude keywords ({len(sk.exclude)}):")
            for item in sk.exclude:
                print(f"  - [{item.category or '?'}] {item.term}")

        strategies = build_query_matrix(sk)
        print(f"\nRetrieval strategies ({len(strategies)}):")
        for s in strategies:
            print(f"  [{s.name} | {s.goal}] {s.query}")
            print(f"    filters: {s.filters}")

        print("-" * 70)


if __name__ == "__main__":
    demo()
