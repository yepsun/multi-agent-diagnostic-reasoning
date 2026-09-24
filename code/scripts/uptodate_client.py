#!/usr/bin/env python3
"""UpToDate查询客户端 — 支持两种访问模式：
1. 直接内网访问（医院网络）
2. aTrust VPN 访问（通过代理）

自动检测当前网络环境，选择最佳访问方式。
严格控制访问频率以避免被识别为爬虫。
"""

import os
import time
import random
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# 加载 .env 文件（如果存在）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # 项目根目录
for _env_path in [os.path.join(project_dir, '.env'), os.path.join(script_dir, '.env')]:
    if os.path.exists(_env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
            break
        except ImportError:
            pass

# UpToDate domain configuration
UPTODATE_BASE_URL = os.environ.get("UPTODATE_BASE_URL", "https://www.uptodate.cn")



class UpToDateClient:
    """UpToDate查询客户端，带频率控制和缓存

    支持两种访问模式：
    - DIRECT: 直接内网访问（医院网络）
    - ATRUST: 通过aTrust VPN访问（需要代理）

    自动检测模式，优先使用直接访问。
    """

    def __init__(self, max_queries_per_case: int = 1, min_interval: float = 5.0,
                 cache_ttl_hours: int = 24, access_mode: str = "auto"):
        """
        Args:
            max_queries_per_case: 每个病例最大查询次数
            min_interval: 两次查询之间的最小间隔（秒）
            cache_ttl_hours: 缓存有效期（小时）
            access_mode: 访问模式 ("auto", "direct", "atrust")
        """
        self.max_queries_per_case = max_queries_per_case
        self.min_interval = min_interval
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.query_count = 0
        self.last_query_time = 0.0

        # Proxy configuration (only used for aTrust mode)
        # MUST be set before _detect_access_mode
        self.proxy = os.environ.get("UPTODATE_PROXY", None)
        self.proxy_user = os.environ.get("UPTODATE_PROXY_USER", None)
        self.proxy_pass = os.environ.get("UPTODATE_PROXY_PASS", None)

        # 确定访问模式
        self.access_mode = self._detect_access_mode(access_mode)
        print(f"[UpToDate] Access mode: {self.access_mode}")

        # 缓存数据库路径
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_db = os.path.join(_project_root, "cache", "uptodate_cache.db")
        self._init_cache()

    def _detect_access_mode(self, mode: str) -> str:
        """检测或设置访问模式"""
        if mode != "auto":
            return mode

        # 检查是否有代理配置
        if self.proxy or os.environ.get("UPTODATE_PROXY"):
            return "atrust"

        # 检查网络环境：尝试快速ping UpToDate CN
        try:
            import socket
            socket.setdefaulttimeout(3)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("www.uptodate.cn", 443))
            sock.close()
            if result == 0:
                return "direct"
        except Exception:
            pass

        # 默认使用 direct（如果无法检测）
        return "direct"

    def _init_cache(self):
        """初始化SQLite缓存数据库"""
        try:
            os.makedirs(os.path.dirname(self.cache_db), exist_ok=True)
            conn = sqlite3.connect(self.cache_db)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uptodate_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[UpToDate] Cache init warning: {e}")

    def _get_cache(self, query: str) -> Optional[str]:
        """从缓存获取结果

        Returns:
            缓存的结果文本，如果不存在或已过期则返回None
        """
        try:
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.execute(
                "SELECT result, created_at FROM uptodate_cache WHERE query_hash = ?",
                (query_hash,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                result, created_at = row
                created = datetime.fromisoformat(created_at)
                if datetime.now() - created < self.cache_ttl:
                    print(f"[UpToDate] Cache hit for: {query[:60]}...")
                    return result
        except Exception as e:
            print(f"[UpToDate] Cache read error: {e}")
        return None

    def _save_cache(self, query: str, result: str):
        """保存结果到缓存"""
        try:
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            conn = sqlite3.connect(self.cache_db)
            conn.execute(
                "INSERT OR REPLACE INTO uptodate_cache (query_hash, query, result) VALUES (?, ?, ?)",
                (query_hash, query, result)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[UpToDate] Cache save error: {e}")

    def _rate_limit(self):
        """频率控制：确保查询间隔并添加随机延迟"""
        # 随机延迟（模拟人类行为）
        delay = random.uniform(0.5, 2.0)
        time.sleep(delay)

        # 确保最小间隔
        elapsed = time.time() - self.last_query_time
        if elapsed < self.min_interval and self.last_query_time > 0:
            wait_time = self.min_interval - elapsed
            print(f"[UpToDate] Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        self.last_query_time = time.time()

    def _score_result_relevance(self, result: Dict, query: str, case_context: str = "") -> int:
        """Score how relevant a search result is to the query and case context.

        Uses multiple signals:
        1. Title-keyword matching (exact > partial)
        2. Snippet relevance (contains key terms from query)
        3. URL path relevance (e.g., /differential-diagnosis/ paths)
        4. Case context matching (if provided)
        5. Medical entity detection in title (disease names, anatomical terms)
        6. Content type diversity scoring (prefer clinical content over patient education)

        Returns a score from 0-100, higher = more relevant.
        """
        score = 0
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()

        # Extract key terms from query (remove common words)
        query_lower = query.lower()
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
                        "vs", "versus", "differential", "diagnosis", "approach", "evaluation"}
        query_terms = [t for t in query_lower.split() if len(t) > 3 and t not in common_words]

        # Signal 1: Title-keyword matching (higher weight for exact matches)
        for term in query_terms:
            if term in title:
                score += 15  # Exact term match in title
            # Check for word boundary match (more precise)
            if re.search(r'\b' + re.escape(term) + r'\b', title):
                score += 10

        # Signal 2: Title contains medical entity terms (indicates clinical specificity)
        medical_entity_suffixes = {
            'itis', 'osis', 'emia', 'uria', 'pathy', 'plasia', 'lysis', 'oma', 'megaly',
            'syndrome', 'disease', 'disorder', 'infection', 'inflammation', 'cancer',
            'carcinoma', 'tumor', 'neoplasm', 'lesion', 'abscess', 'granuloma',
            'proctitis', 'pneumonia', 'hepatitis', 'nephritis', 'colitis', 'vasculitis',
            'fissure', 'ulcer', 'erosion', 'stricture', 'fistula',
        }
        title_words = title.split()
        for word in title_words:
            word_clean = word.strip('.,;:!?()[]').lower()
            if any(word_clean.endswith(suffix) for suffix in medical_entity_suffixes):
                score += 8
            # Check for disease names in title
            if len(word_clean) > 5 and word_clean in query_lower:
                score += 12

        # Signal 3: Snippet relevance
        for term in query_terms:
            if term in snippet:
                score += 8  # Term appears in snippet
            if re.search(r'\b' + re.escape(term) + r'\b', snippet):
                score += 5

        # Signal 4: URL path relevance (strong signal for content type)
        if "/contents/" in url:
            score += 10  # Main content article
        if "differential" in url or "diagnosis" in url:
            score += 20  # Differential diagnosis specific page (high value)
        if "approach" in url or "evaluation" in url:
            score += 15  # Clinical approach page (high value)
        if "-overview" in url or "-basics" in url:
            score += 8   # Overview pages (good for general info)
        if "summary-and-recommendations" in url:
            score += 10   # Summary pages are useful
        if "patient-education" in url or "beyond-the-basics" in url:
            score -= 30  # Patient-oriented, less useful for clinician
        if "search" in url and url.endswith("search"):
            score -= 10  # Search page itself, not a content page

        # Signal 5: Content length indicator (longer snippets often mean more substantive content)
        snippet_len = len(snippet)
        if snippet_len > 100:
            score += 5
        if snippet_len > 200:
            score += 3

        # Signal 6: Case context matching (if provided) - high weight for direct relevance
        if case_context:
            case_lower = case_context.lower()
            # Extract meaningful terms from case context (focus on clinical findings)
            case_terms = []
            for t in case_lower.split():
                t_clean = t.strip('.,;:!?()[]')
                if len(t_clean) > 4 and t_clean not in common_words:
                    # Prioritize medical terms
                    if any(t_clean.endswith(suffix) for suffix in medical_entity_suffixes):
                        case_terms.append(t_clean)
                    elif len(t_clean) > 6:  # Longer terms are more likely to be specific
                        case_terms.append(t_clean)
            # Limit to top 15 most specific terms
            case_terms = case_terms[:15]

            for term in case_terms:
                if term in title:
                    score += 8
                if term in snippet:
                    score += 5
                # Word boundary match for higher precision
                if re.search(r'\b' + re.escape(term) + r'\b', title):
                    score += 5
                if re.search(r'\b' + re.escape(term) + r'\b', snippet):
                    score += 3

        return min(score, 100)  # Cap at 100

    def _select_best_results_with_llm(self, results: List[Dict], query: str,
                                         case_context: str = "", max_select: int = 5) -> List[Dict]:
        """使用LLM智能选择最相关的UpToDate搜索结果。

        LLM自行决定选择几个页面（0到max_select个），如果无合适页面可返回空列表。
        改进策略：
        1. 先对所有结果按相关性评分排序
        2. 让LLM看到更多候选（15个而非10个），提高发现相关文章的概率
        3. 确保选择的页面覆盖不同方面（诊断、治疗、鉴别诊断）
        4. 如果搜索结果涵盖多个疾病，确保每个疾病至少有一篇相关文章

        Args:
            results: 搜索结果列表，每个结果包含title, url, snippet
            query: 原始搜索查询
            case_context: 病例文本，用于智能相关性评分
            max_select: 最多可选择的页面数（LLM可决定选更少，包括0个）

        Returns:
            被选中的结果列表，可能为空列表（如果LLM认为无合适页面）
        """
        if not results:
            return []

        # Step 1: Score and sort all results by relevance
        scored_results = []
        for result in results:
            score = self._score_result_relevance(result, query, case_context)
            scored_results.append((score, result))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Step 2: Deduplicate by title similarity before sending to LLM
        deduplicated = []
        seen_titles = set()
        for score, result in scored_results:
            title = result.get("title", "").lower().strip()
            # Skip very similar titles
            is_duplicate = False
            for seen in seen_titles:
                # Check if titles are very similar (>80% overlap or one contains the other)
                if title in seen or seen in title:
                    is_duplicate = True
                    break
                # Check word overlap for longer titles
                title_words = set(title.split())
                seen_words = set(seen.split())
                if len(title_words) > 3 and len(seen_words) > 3:
                    overlap = len(title_words & seen_words) / min(len(title_words), len(seen_words))
                    if overlap > 0.8:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append((score, result))
                seen_titles.add(title)

        # Step 3: Build LLM prompt with top 15 deduplicated results
        # Show LLM the top 15 results (increased from 10) for better coverage
        top_results = deduplicated[:15]

        results_text = "\n\n".join([
            f"[{i+1}] {r.get('title', '')} (Relevance Score: {s}/100)\n"
            f"    URL: {r.get('url', '')}\n"
            f"    Summary: {r.get('snippet', '')[:200]}..."
            for i, (s, r) in enumerate(top_results)
        ])

        prompt = f"""You are a senior clinician selecting the most relevant UpToDate articles for a difficult case.

## Case Context
{case_context[:800] if case_context else 'No additional case context provided.'}

## Search Query
{query}

## Available UpToDate Search Results (sorted by relevance score)
{results_text}

## Your Task
Evaluate these search results and select ONLY the most relevant articles for this specific case.

Selection rules:
1. Select 0 to {max_select} articles - quality over quantity
2. If NONE of the articles are directly relevant to the case, return an empty selection (selected_indices: [])
3. Avoid pediatric articles unless the case involves children
4. Avoid patient-education articles
5. Prioritize articles that directly address the suspected diagnosis or differential
6. Consider clinical utility - will this article help with diagnosis or management?
7. **DIVERSITY REQUIREMENT**: If the search results cover different aspects (e.g., diagnosis vs treatment vs differential diagnosis), try to select at least one from each relevant aspect. The goal is comprehensive coverage, not just the single highest-scoring article.
8. **DISEASE COVERAGE**: If the query involves multiple diseases (e.g., "disease A vs disease B"), ensure articles covering EACH disease are represented if available.

## Output Format (JSON only)
{{
  "selected_indices": [1, 3, 5],
  "reasoning": "Brief explanation of selection rationale. If empty, explain why no articles were selected.",
  "relevance_assessment": "Brief assessment of overall result quality (good/fair/poor)",
  "coverage_analysis": "Which diseases/aspects are covered by selected articles"
}}

Return ONLY valid JSON."""

        try:
            # 调用LLM
            llm_response = self._call_llm_for_selection(prompt)

            # 解析JSON响应 - 直接尝试解析整个响应，不使用正则回退
            import json

            # 清理可能的 markdown 代码块标记
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # 直接解析JSON
            data = json.loads(cleaned_response)
            selected_indices = data.get("selected_indices", [])
            reasoning = data.get("reasoning", "")
            relevance = data.get("relevance_assessment", "unknown")
            coverage = data.get("coverage_analysis", "")

            # 验证 selected_indices 是列表
            if not isinstance(selected_indices, list):
                print(f"[UpToDate] LLM returned invalid indices format, returning empty")
                return []

            # 如果LLM选择为空列表，返回空（表示无合适页面）
            if not selected_indices:
                print(f"[UpToDate] LLM determined no relevant articles found. Reason: {reasoning}")
                return []

            selected = []
            for idx in selected_indices:
                if 1 <= idx <= len(top_results):
                    score, result = top_results[idx - 1]
                    selected.append({
                        **result,
                        "relevance_score": score,
                        "selection_reason": f"LLM selected: {reasoning[:100]}"
                    })

            if selected:
                print(f"[UpToDate] LLM selected {len(selected)} articles (assessment: {relevance}): {reasoning[:100]}...")
                if coverage:
                    print(f"[UpToDate] Coverage: {coverage[:150]}...")
                return selected[:max_select]
            else:
                print(f"[UpToDate] LLM returned valid indices but none matched available results")
                return []

        except Exception as e:
            print(f"[UpToDate] LLM selection failed ({e}), no articles selected")
            return []

    def _call_llm_for_selection(self, prompt: str) -> str:
        """调用LLM进行页面选择，使用与retrieval_module相同的配置。"""
        # 导入retrieval_module的LLM配置
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from retrieval_module import LLM_API_KEY, LLM_API_BASE, LLM_MODEL, LLM_PROVIDER
        except ImportError:
            LLM_API_KEY = os.environ.get("RETRIEVAL_LLM_API_KEY", "") or os.environ.get("DEEP_SEEK_API", "") or os.environ.get("DEEPSEEK_API_KEY", "")
            LLM_API_BASE = os.environ.get("RETRIEVAL_LLM_API_BASE", "") or os.environ.get("LLM_API_BASE", "") or "https://api.deepseek.com/v1"
            LLM_MODEL = os.environ.get("RETRIEVAL_LLM_MODEL", "") or os.environ.get("LLM_MODEL", "") or "deepseek-chat"
            LLM_PROVIDER = "deepseek-flash"

        if not LLM_API_KEY:
            raise RuntimeError("No LLM API key available")

        import requests

        if LLM_PROVIDER in ("deepseek-flash", "openai"):
            base = LLM_API_BASE.rstrip("/") if LLM_API_BASE else "https://api.deepseek.com/v1"
            url = f"{base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            if result.get("choices"):
                return result["choices"][0]["message"]["content"]
            else:
                return ""
        else:
            raise RuntimeError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    def _select_best_results(self, results: List[Dict], query: str,
                             case_context: str = "", max_select: int = 3) -> List[Dict]:
        """Select the most relevant results from search results.

        Args:
            results: List of search result dicts
            query: Original search query
            case_context: Optional case text for context-aware scoring
            max_select: Maximum number of results to select

        Returns:
            List of selected result dicts, sorted by relevance
        """
        if not results:
            return []

        # Score each result
        scored = []
        for result in results:
            score = self._score_result_relevance(result, query, case_context)
            scored.append((score, result))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select top results, ensuring diversity (don't pick multiple results with same title)
        selected = []
        seen_titles = set()
        for score, result in scored:
            title = result.get("title", "")
            # Simple dedup: skip if title is very similar to already selected
            is_duplicate = False
            for seen in seen_titles:
                if title.lower() in seen.lower() or seen.lower() in title.lower():
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append({
                    **result,
                    "relevance_score": score,
                    "selection_reason": f"Relevance score: {score}/100"
                })
                seen_titles.add(title)

            if len(selected) >= max_select:
                break

        return selected

    def fetch_article_content(self, url: str, max_chars: int = 3000) -> Optional[str]:
        """Fetch and extract the main content from an UpToDate article page.

        Uses Playwright to render the article page and extract the clinical content.
        Includes multiple fallback strategies for content extraction.

        Args:
            url: UpToDate article URL
            max_chars: Maximum characters to extract

        Returns:
            Extracted article text, or None if failed
        """
        try:
            from playwright.sync_api import sync_playwright
            from bs4 import BeautifulSoup

            print(f"[UpToDate] Fetching article content: {url[:80]}...")

            with sync_playwright() as p:
                # Try WebKit first (more reliable on macOS)
                try:
                    browser = p.webkit.launch(headless=True)
                except Exception:
                    browser = p.chromium.launch(headless=True)

                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )
                page = context.new_page()

                # Navigate to article with shorter timeout and less strict wait
                try:
                    response = page.goto(url, timeout=15000, wait_until="domcontentloaded")
                    if not response or response.status != 200:
                        print(f"[UpToDate] Article page failed to load: {response.status if response else 'no response'}")
                        browser.close()
                        return None
                except Exception as e:
                    print(f"[UpToDate] Page navigation timeout, trying to extract anyway...")

                # Wait for content to render
                page.wait_for_timeout(3000)

                # Try to find main content area
                content_selectors = [
                    "[class*='content-body']",
                    "[class*='article-content']",
                    "[class*='topic-content']",
                    "main",
                    "article",
                    "[role='main']",
                    ".content",
                    "#content",
                    "[class*='body']",
                ]

                main_text = ""
                for selector in content_selectors:
                    try:
                        if page.locator(selector).count() > 0:
                            # Extract text using page.evaluate for better performance
                            text = page.evaluate(f"""
                                () => {{
                                    const el = document.querySelector('{selector}');
                                    if (el) {{
                                        // Remove script and style elements
                                        const scripts = el.querySelectorAll('script, style, nav, header, footer');
                                        scripts.forEach(s => s.remove());
                                        return el.innerText;
                                    }}
                                    return '';
                                }}
                            """)
                            if text and len(text) > 200:
                                main_text = text
                                print(f"[UpToDate] Found content with selector: {selector}")
                                break
                    except Exception:
                        continue

                # If no content found with selectors, try getting all paragraph text
                if not main_text:
                    try:
                        main_text = page.evaluate("""
                            () => {
                                const paragraphs = document.querySelectorAll('p');
                                return Array.from(paragraphs)
                                    .map(p => p.innerText.trim())
                                    .filter(t => t.length > 50)
                                    .join('\\n\\n');
                            }
                        """)
                    except Exception:
                        pass

                browser.close()

                if main_text:
                    # Clean up the text
                    lines = [line.strip() for line in main_text.split('\n') if line.strip()]
                    cleaned = '\n'.join(lines)

                    # Truncate if too long
                    if len(cleaned) > max_chars:
                        cleaned = cleaned[:max_chars] + "..."

                    print(f"[UpToDate] Extracted {len(cleaned)} chars from article")
                    return cleaned
                else:
                    print("[UpToDate] No content found in article page")
                    return None

        except Exception as e:
            print(f"[UpToDate] Failed to fetch article content: {e}")
            return None

    def search_with_content(self, query: str, case_context: str = "",
                           fetch_content: bool = True, max_articles: int = 2) -> Optional[str]:
        """Smart UpToDate search that selects best results and optionally fetches full content.

        This is the enhanced version of search() that:
        1. Gets search results
        2. Scores and selects most relevant results
        3. Optionally fetches full article content for top results
        4. Returns formatted output with both summaries and full content

        Args:
            query: Search query
            case_context: Optional case text for relevance scoring
            fetch_content: Whether to fetch full article content
            max_articles: Maximum number of articles to fetch content for

        Returns:
            Formatted string with search results and optionally article content
        """
        # Step 1: Get search results using existing search method
        search_result = self.search(query)
        if not search_result:
            return None

        # Parse the search results to get structured data
        # (We need to re-run the search to get structured results)
        # For now, let's do a fresh search with structured output
        return self._search_with_smart_selection(query, case_context, fetch_content, max_articles)

    def _search_with_smart_selection(self, query: str, case_context: str = "",
                                      fetch_content: bool = True, max_articles: int = 2) -> Optional[str]:
        """Internal method that performs smart selection and content fetching."""
        # Check cache first
        cache_key = f"smart:{query}:{case_context[:100]}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        # Check query limit
        if self.query_count >= self.max_queries_per_case:
            print(f"[UpToDate] Query limit reached")
            return None

        self._rate_limit()

        try:
            from playwright.sync_api import sync_playwright
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus

            print(f"[UpToDate] Smart search: {query}")

            # Step 1: Get search results
            with sync_playwright() as p:
                try:
                    browser = p.webkit.launch(headless=True)
                except Exception:
                    browser = p.chromium.launch(headless=True)

                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )
                page = context.new_page()

                search_url = f"{UPTODATE_BASE_URL}/contents/search?search={quote_plus(query)}"
                response = page.goto(search_url, timeout=60000, wait_until="domcontentloaded")

                if response:
                    print(f"[UpToDate] Search page loaded: {response.status}")

                page.wait_for_timeout(3000)

                # Check for results
                selectors = [
                    ".search-result",
                    ".search-results-item",
                    "[data-search-result]",
                    "[class*='search-result']",
                ]

                content = page.content()
                browser.close()

            # Parse results
            soup = BeautifulSoup(content, "html.parser")
            results = []

            for selector in selectors:
                items = soup.select(selector)
                if items:
                    for item in items:
                        title_el = item.select_one("a, .title, .result-title, h3, h2, [class*='title']")
                        snippet_el = item.select_one(".snippet, .description, .abstract, p, [class*='snippet'], [class*='description']")
                        if title_el:
                            href = title_el.get("href", "")
                            if href and not href.startswith("http"):
                                href = f"{UPTODATE_BASE_URL}" + href
                            results.append({
                                "title": title_el.get_text(strip=True),
                                "url": href,
                                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                            })
                    break

            if not results:
                print("[UpToDate] No search results found")
                return None

            print(f"[UpToDate] Found {len(results)} results, selecting best...")

            # Step 2: Select best results using LLM (LLM decides how many, 0-5)
            selected = self._select_best_results_with_llm(results, query, case_context, max_select=5)

            if not selected:
                print("[UpToDate] LLM determined no relevant articles to download")
                # Return search results without downloading any content
                lines = [f"UpToDate Search: {query}", "=" * 60]
                lines.append("\n## Search Results (No articles selected for download)\n")
                lines.append("LLM assessment: No sufficiently relevant articles found for this case.")
                lines.append("Consider refining the search query or trying alternative search terms.")
                return "\n".join(lines)

            # Step 3: Build output
            lines = [f"UpToDate Search: {query}", "=" * 60]

            # Add selected results with scores
            lines.append(f"\n## Top Relevant Results (LLM selected {len(selected)} articles)\n")
            for i, r in enumerate(selected, 1):
                lines.append(f"\n[{i}] {r['title']} (Score: {r.get('relevance_score', 0)}/100)")
                lines.append(f"    URL: {r['url']}")
                if r.get("snippet"):
                    lines.append(f"    Summary: {r['snippet'][:200]}...")
                if r.get("selection_reason"):
                    lines.append(f"    Reason: {r['selection_reason']}")

            # Step 4: Fetch full content for top articles (max 1 - minimize per-query cost)
            max_download = 1  # Hard limit: download only the single most relevant article
            articles_to_download = selected[:max_download]

            if fetch_content and articles_to_download:
                lines.append(f"\n\n## Detailed Content (downloading {len(articles_to_download)} of {len(selected)} selected articles)\n")

                downloaded_count = 0
                for i, article in enumerate(articles_to_download, 1):
                    if article.get("url"):
                        print(f"[UpToDate] Downloading article {i}/{len(articles_to_download)}: {article['title'][:60]}...")
                        content = self.fetch_article_content(article["url"], max_chars=2500)
                        if content:
                            lines.append(f"\n### Article {i}: {article['title']}\n")
                            lines.append(content)
                            lines.append("\n" + "-" * 40)
                            downloaded_count += 1
                        else:
                            lines.append(f"\n### Article {i}: {article['title']}\n")
                            lines.append("[Failed to download content]")
                            lines.append("\n" + "-" * 40)

                if downloaded_count == 0:
                    lines.append("\n*Note: Failed to download content for selected articles.*")

            result_text = "\n".join(lines)

            # Save to cache
            self._save_cache(cache_key, result_text)
            self.query_count += 1

            print(f"[UpToDate] Smart search complete ({self.query_count}/{self.max_queries_per_case})")
            return result_text

        except Exception as e:
            print(f"[UpToDate] Smart search failed: {e}")
            return None
        """使用requests+cloudscraper作为playwright的替代方案

        当playwright在aTrust环境下无法工作时，使用此方法。
        cloudscraper可以绕过Cloudflare等WAF检测。
        """
        try:
            import cloudscraper
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus

            print(f"[UpToDate] Trying cloudscraper fallback for: {query}")

            scraper = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'darwin',
                    'desktop': True
                }
            )

            search_url = f"{UPTODATE_BASE_URL}/contents/search?search={quote_plus(query)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
            }

            response = scraper.get(search_url, headers=headers, timeout=30)
            print(f"[UpToDate] cloudscraper status: {response.status_code}")

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # 尝试提取搜索结果
                results = []
                selectors = [
                    ".search-result",
                    ".search-results-item",
                    "[data-search-result]",
                    ".result-item",
                    "article",
                    "[class*='search-result']",
                    "[class*='SearchResult']",
                ]

                for selector in selectors:
                    items = soup.select(selector)
                    if items:
                        for item in items[:5]:
                            title_el = item.select_one("a, .title, .result-title, h3, h2")
                            snippet_el = item.select_one(".snippet, .description, .abstract, p")
                            if title_el:
                                href = title_el.get("href", "")
                                if href and not href.startswith("http"):
                                    href = f"{UPTODATE_BASE_URL}" + href
                                results.append({
                                    "title": title_el.get_text(strip=True),
                                    "url": href,
                                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                                })
                        if results:
                            break

                if results:
                    lines = [f"UpToDate Search: {query}", "=" * 50]
                    for i, r in enumerate(results, 1):
                        lines.append(f"\n[{i}] {r['title']}")
                        if r["url"]:
                            lines.append(f"    URL: {r['url']}")
                        if r["snippet"]:
                            lines.append(f"    Snippet: {r['snippet'][:200]}...")

                    result_text = "\n".join(lines)
                    self._save_cache(query, result_text)
                    self.query_count += 1
                    print(f"[UpToDate] cloudscraper query successful ({self.query_count}/{self.max_queries_per_case})")
                    return result_text

            print("[UpToDate] cloudscraper returned no results")
            return None

        except ImportError:
            print("[UpToDate] cloudscraper not installed. Try: pip install cloudscraper")
            return None
        except Exception as e:
            print(f"[UpToDate] cloudscraper failed: {e}")
            return None

    def search(self, query: str) -> Optional[str]:
        """执行UpToDate搜索，根据访问模式选择最佳策略

        访问模式：
        - DIRECT: 直接内网访问，使用标准playwright配置
        - ATRUST: 通过aTrust VPN，需要禁用web security等

        Args:
            query: 搜索查询词（英文）

        Returns:
            格式化的搜索结果文本，如果失败或达到查询限制则返回None
        """
        # 检查查询限制
        if self.query_count >= self.max_queries_per_case:
            print(f"[UpToDate] Query limit reached ({self.max_queries_per_case} per case)")
            return None

        # 检查缓存
        cached = self._get_cache(query)
        if cached:
            return cached

        # 频率控制
        self._rate_limit()

        # 根据访问模式选择策略
        if self.access_mode == "atrust":
            return self._search_atrust(query)
        else:
            return self._search_direct(query)

    def _search_direct(self, query: str) -> Optional[str]:
        """直接内网访问模式 - 标准playwright配置"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                from playwright.sync_api import sync_playwright
                from bs4 import BeautifulSoup
                from urllib.parse import quote_plus

                print(f"[UpToDate] Searching (direct mode, attempt {attempt + 1}/{max_retries + 1}): {query}")

                with sync_playwright() as p:
                    # 直接模式：使用标准webkit配置
                    browser = None
                    try:
                        browser = p.webkit.launch(headless=True)
                        print("[UpToDate] Using WebKit browser (direct mode)")
                    except Exception as e:
                        print(f"[UpToDate] WebKit not available ({e}), trying chromium...")
                        browser = p.chromium.launch(headless=True)
                        print("[UpToDate] Using Chromium browser (direct mode)")

                    context = browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        locale="en-US",
                    )
                    page = context.new_page()

                    search_url = f"{UPTODATE_BASE_URL}/contents/search?search={quote_plus(query)}"
                    response = page.goto(search_url, timeout=60000, wait_until="domcontentloaded")

                    if response:
                        print(f"[UpToDate] Page loaded with status: {response.status}")

                    # 等待内容渲染
                    page.wait_for_timeout(3000)

                    # 检查搜索结果
                    selectors = [
                        ".search-result",
                        ".search-results-item",
                        "[data-search-result]",
                        "[class*='search-result']",
                    ]

                    content = page.content()
                    browser.close()

                    # 解析结果
                    soup = BeautifulSoup(content, "html.parser")
                    results = []

                    for selector in selectors:
                        items = soup.select(selector)
                        if items:
                            for item in items:
                                title_el = item.select_one("a, .title, .result-title, h3, h2, [class*='title']")
                                snippet_el = item.select_one(".snippet, .description, .abstract, p, [class*='snippet'], [class*='description']")
                                if title_el:
                                    href = title_el.get("href", "")
                                    if href and not href.startswith("http"):
                                        href = f"{UPTODATE_BASE_URL}" + href
                                    results.append({
                                        "title": title_el.get_text(strip=True),
                                        "url": href,
                                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                                    })
                            break

                    if not results:
                        print("[UpToDate] No search results found")
                        return None

                    lines = [f"UpToDate Search: {query}", "=" * 50]
                    for i, r in enumerate(results, 1):
                        lines.append(f"\n[{i}] {r['title']}")
                        if r["url"]:
                            lines.append(f"    URL: {r['url']}")
                        if r["snippet"]:
                            lines.append(f"    Snippet: {r['snippet'][:200]}...")

                    result_text = "\n".join(lines)
                    self._save_cache(query, result_text)
                    self.query_count += 1
                    print(f"[UpToDate] Query successful ({self.query_count}/{self.max_queries_per_case})")
                    return result_text

            except Exception as e:
                print(f"[UpToDate] Search failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    wait_time = 5 * (attempt + 1)
                    print(f"[UpToDate] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        return None

    def _search_atrust(self, query: str) -> Optional[str]:
        """aTrust VPN访问模式 - 需要禁用web security等"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                from playwright.sync_api import sync_playwright
                from bs4 import BeautifulSoup
                from urllib.parse import quote_plus

                print(f"[UpToDate] Searching (aTrust mode, attempt {attempt + 1}/{max_retries + 1}): {query}")

                with sync_playwright() as p:
                    # aTrust模式：使用chromium并禁用web security
                    browser = None
                    try:
                        browser = p.chromium.launch(
                            headless=True,
                            args=[
                                '--disable-web-security',
                                '--disable-features=IsolateOrigins,site-per-process',
                                '--disable-site-isolation-trials',
                                '--disable-blink-features=AutomationControlled',
                                '--no-sandbox',
                                '--disable-setuid-sandbox',
                            ]
                        )
                        print("[UpToDate] Using Chromium browser (aTrust mode)")
                    except Exception as e:
                        print(f"[UpToDate] Chromium not available ({e}), trying WebKit...")
                        browser = p.webkit.launch(headless=True)
                        print("[UpToDate] Using WebKit browser (aTrust mode)")

                    context = browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        locale="en-US",
                    )
                    page = context.new_page()

                    search_url = f"{UPTODATE_BASE_URL}/contents/search?search={quote_plus(query)}"
                    response = page.goto(search_url, timeout=60000, wait_until="domcontentloaded")

                    if response:
                        print(f"[UpToDate] Page loaded with status: {response.status}")

                    # 等待内容渲染
                    page.wait_for_timeout(3000)

                    # 检查搜索结果
                    selectors = [
                        ".search-result",
                        ".search-results-item",
                        "[data-search-result]",
                        "[class*='search-result']",
                    ]

                    content = page.content()
                    browser.close()

                    # 解析结果
                    soup = BeautifulSoup(content, "html.parser")
                    results = []

                    for selector in selectors:
                        items = soup.select(selector)
                        if items:
                            for item in items:
                                title_el = item.select_one("a, .title, .result-title, h3, h2, [class*='title']")
                                snippet_el = item.select_one(".snippet, .description, .abstract, p, [class*='snippet'], [class*='description']")
                                if title_el:
                                    href = title_el.get("href", "")
                                    if href and not href.startswith("http"):
                                        href = f"{UPTODATE_BASE_URL}" + href
                                    results.append({
                                        "title": title_el.get_text(strip=True),
                                        "url": href,
                                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                                    })
                            break

                    if not results:
                        print("[UpToDate] No search results found")
                        return None

                    lines = [f"UpToDate Search: {query}", "=" * 50]
                    for i, r in enumerate(results, 1):
                        lines.append(f"\n[{i}] {r['title']}")
                        if r["url"]:
                            lines.append(f"    URL: {r['url']}")
                        if r["snippet"]:
                            lines.append(f"    Snippet: {r['snippet'][:200]}...")

                    result_text = "\n".join(lines)
                    self._save_cache(query, result_text)
                    self.query_count += 1
                    print(f"[UpToDate] Query successful ({self.query_count}/{self.max_queries_per_case})")
                    return result_text

            except Exception as e:
                print(f"[UpToDate] Search failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    wait_time = 5 * (attempt + 1)
                    print(f"[UpToDate] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        return None


    def search_differential(self, diagnoses: List[str], specific_finding: str = "", clinical_context: str = "", use_smart: bool = True, case_context: str = "") -> Optional[str]:
        """查询鉴别诊断信息 — 利用UpToDate最强大的鉴别诊断功能

        构建针对鉴别诊断的查询，而非单一疾病查询。
        例如："Tay-Sachs disease vs Krabbe disease differential diagnosis"

        Args:
            diagnoses: 需要鉴别的疾病列表（最多3个）
            specific_finding: 特定的临床表现（可选）
            clinical_context: 临床场景（如"infant", "adult", "MSM"等）
            use_smart: 是否使用智能选择和内容获取
            case_context: 完整病例文本，用于智能相关性评分

        Returns:
            格式化的搜索结果
        """
        if not diagnoses:
            return None

        # 构建鉴别诊断查询 - OPTIMIZED: shorter queries to avoid timeouts
        # Use only the primary diagnosis + one differential, or symptoms only
        primary_dx = diagnoses[0] if diagnoses else ""
        secondary_dx = diagnoses[1] if len(diagnoses) > 1 else ""

        # Build short query strategies
        queries = []

        # 策略1：最简短的核心鉴别诊断查询（优先，避免超时）
        if secondary_dx:
            queries.append(f"{primary_dx} {secondary_dx}")
        else:
            queries.append(primary_dx)

        # 策略2：仅使用主要诊断
        queries.append(primary_dx)

        # 策略3：加入主要临床表现（如果简短）
        if specific_finding and len(specific_finding) < 40:
            queries.append(f"{primary_dx} {specific_finding}")

        # 策略4：针对主要症状/表现的查询（如果太长，用症状替代）
        if specific_finding and clinical_context:
            short_context = clinical_context.split()[0] if len(clinical_context.split()) > 1 else clinical_context
            queries.append(f"{specific_finding} {short_context}")

        # 选择最短且非空的查询，避免超时
        query = ""
        for q in queries:
            if q and len(q) <= 80:
                query = q
                break
        if not query:
            query = queries[0] if queries else primary_dx

        print(f"[UpToDate] Simplified differential query: {query}")

        if use_smart:
            return self.search_with_content(query, case_context=case_context, fetch_content=True, max_articles=2)
        else:
            return self.search(query)

    def search_clinical_approach(self, presentation: str, population: str = "") -> Optional[str]:
        """查询临床诊断方法 — 利用UpToDate的Clinical Approach/Evaluation模块

        当面对不典型表现或需要系统评估时，查询UpToDate的临床方法。
        例如："approach to rectal pain in homosexual men"

        Args:
            presentation: 主要临床表现（如"rectal pain", "developmental regression"）
            population: 特定人群（如"MSM", "infant"）

        Returns:
            格式化的搜索结果
        """
        if not presentation:
            return None

        if population:
            query = f"approach to {presentation} in {population}"
        else:
            query = f"approach to {presentation}"

        return self.search(query)

    def search_diagnostic_criteria(self, disease: str, key_feature: str = "") -> Optional[str]:
        """查询诊断标准/关键特征 — 利用UpToDate的Diagnosis模块

        当需要确认某个疾病的诊断标准或关键鉴别特征时查询。
        例如："diagnostic criteria for syphilitic proctitis"

        Args:
            disease: 疾病名称
            key_feature: 关键特征（可选）

        Returns:
            格式化的搜索结果
        """
        if not disease:
            return None

        if key_feature:
            query = f"{disease} diagnostic criteria {key_feature}"
        else:
            query = f"{disease} diagnostic criteria"

        return self.search(query)

    def search_finding(self, finding: str, context: str = "") -> Optional[str]:
        """查询特定临床发现的鉴别诊断意义

        例如："startle response infant differential diagnosis"
        """
        if not finding:
            return None

        if context:
            query = f"{finding} {context} differential diagnosis"
        else:
            query = f"{finding} differential diagnosis"

        return self.search(query)
