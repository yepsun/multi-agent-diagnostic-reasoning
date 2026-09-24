import pytest

from webapp import pubmed


class FakeResp:
    def __init__(self, payload=None, text="", status=200):
        self._payload = payload
        self.text = text
        self.status_code = status

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    @property
    def ok(self):
        return self.status_code < 400


ESEARCH = {"esearchresult": {"idlist": ["111", "222"]}}
ESUMMARY = {"result": {
    "111": {"title": "Multicentric Castleman disease", "source": "Blood",
            "pubdate": "2020 Jan", "authors": [{"name": "A"}, {"name": "B"},
                                               {"name": "C"}, {"name": "D"}]},
    "222": {"title": "POEMS syndrome", "source": "NEJM",
            "pubdate": "2019", "authors": [{"name": "X"}]},
}}
EFETCH_XML = ('<?xml version="1.0"?><PubmedArticleSet><PubmedArticle>'
              '<MedlineCitation><Article><Abstract>'
              '<AbstractText>Background part.</AbstractText>'
              '<AbstractText>Methods part.</AbstractText>'
              '</Abstract></Article></MedlineCitation></PubmedArticle>'
              '</PubmedArticleSet>')


class TestSearchPubmed:
    def test_builds_articles_with_real_metadata(self, monkeypatch):
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params))
            if "esearch" in url:
                return FakeResp(payload=ESEARCH)
            if "esummary" in url:
                return FakeResp(payload=ESUMMARY)
            return FakeResp(text=EFETCH_XML)

        monkeypatch.setattr(pubmed.requests, "get", fake_get)
        monkeypatch.setattr(pubmed.time, "sleep", lambda s: None)
        arts = pubmed.search_pubmed("multicentric castleman disease", retmax=2)

        assert len(arts) == 2
        a = arts[0]
        assert a["pmid"] == "111"
        assert a["title"] == "Multicentric Castleman disease"
        assert a["journal"] == "Blood"
        assert a["authors"].startswith("A, B, C") and a["authors"].endswith("et al.")
        assert a["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
        assert "Background part." in a["abstract"]
        assert "Methods part." in a["abstract"]

        esearch_url, esearch_params = calls[0]
        assert "esearch" in esearch_url
        assert esearch_params["term"] == "multicentric castleman disease"
        assert esearch_params["sort"] == "relevance"
        assert esearch_params["db"] == "pubmed"

    def test_api_key_injected_from_env(self, monkeypatch):
        monkeypatch.setenv("NCBI_API_KEY", "ktest")
        seen = {}

        def fake_get(url, params=None, timeout=None):
            seen.update(params or {})
            if "esearch" in url:
                return FakeResp(payload={"esearchresult": {"idlist": []}})
            return FakeResp(payload={})

        monkeypatch.setattr(pubmed.requests, "get", fake_get)
        pubmed.search_pubmed("anemia")
        assert seen["api_key"] == "ktest"

    def test_no_results_returns_empty(self, monkeypatch):
        monkeypatch.setattr(pubmed.requests, "get",
                            lambda url, params=None, timeout=None:
                            FakeResp(payload={"esearchresult": {"idlist": []}}))
        assert pubmed.search_pubmed("zzzqqq") == []

    def test_short_query_raises(self):
        with pytest.raises(ValueError):
            pubmed.search_pubmed("a")

    def test_http_error_raises_runtime(self, monkeypatch):
        monkeypatch.setattr(pubmed.requests, "get",
                            lambda url, params=None, timeout=None:
                            FakeResp(status=503))
        with pytest.raises(RuntimeError):
            pubmed.search_pubmed("anemia")
