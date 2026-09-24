"""PubMed E-utilities client.

Privacy: only the diagnosis search term leaves the intranet — never the
case text. Citations are always rendered from real retrieval metadata; the
LLM never generates references.
"""
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

from . import config

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _base_params() -> Dict:
    params = {"tool": "mdt_webapp", "retmode": "json"}
    key = os.environ.get("NCBI_API_KEY", "")
    if key:
        params["api_key"] = key
    return params


def search_pubmed(query: str, retmax: int = None, timeout: int = 30) -> List[Dict]:
    """Search PubMed and return articles with real metadata + abstracts.

    Raises ValueError on a too-short query, RuntimeError on NCBI errors.
    """
    query = (query or "").strip()
    if len(query) < 2:
        raise ValueError("检索词太短（至少2个字符）")
    query = query[:200]
    retmax = retmax or config.PUBMED_RETMAX

    r = requests.get(f"{EUTILS}/esearch.fcgi",
                     params={**_base_params(), "db": "pubmed", "term": query,
                             "retmax": retmax, "sort": "relevance"},
                     timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"esearch HTTP {r.status_code}")
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    r = requests.get(f"{EUTILS}/esummary.fcgi",
                     params={**_base_params(), "db": "pubmed",
                             "id": ",".join(ids)},
                     timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"esummary HTTP {r.status_code}")
    summaries = r.json().get("result", {})

    articles = []
    for pmid in ids:
        info = summaries.get(pmid, {})
        authors = [a.get("name", "") for a in info.get("authors", [])[:3]]
        label = ", ".join(a for a in authors if a)
        if len(info.get("authors", [])) > 3:
            label += ", et al."
        articles.append({
            "pmid": pmid,
            "title": info.get("title", ""),
            "journal": info.get("source", ""),
            "pubdate": info.get("pubdate", ""),
            "authors": label,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract": "",
        })

    for a in articles:
        time.sleep(0.35)  # NCBI rate limit
        r = requests.get(f"{EUTILS}/efetch.fcgi",
                         params={**_base_params(), "db": "pubmed", "id": a["pmid"],
                                 "rettype": "abstract", "retmode": "xml"},
                         timeout=timeout)
        if r.ok:
            a["abstract"] = _abstract_from_efetch(r.text)
    return articles


def _abstract_from_efetch(text: str) -> str:
    """Pull AbstractText sections out of an efetch XML record."""
    try:
        root = ET.fromstring(text)
        parts = ["".join(node.itertext()) for node in root.findall(".//AbstractText")]
        return "\n".join(p.strip() for p in parts if p.strip())
    except ET.ParseError:
        return ""
    marker = "Abstract\n"
    idx = text.find(marker)
    if idx != -1:
        rest = text[idx + len(marker):]
        for stop in ("\n \n", "\n\nPMID", "\nPMID"):
            j = rest.find(stop)
            if j != -1:
                return rest[:j].strip()
        return rest.strip()
    return ""
