"""
Unit tests for NazoratAI's XERJ search and early warning anomaly detection integration.
"""

import json
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from nazorat_ai.integrations.xerj import XerjEWSClient


class TestXerjEWSClient:
    """Test suite for XerjEWSClient functionality."""

    def test_client_init(self):
        client = XerjEWSClient(base_url="http://localhost:9200/", api_key="secret123")
        assert client.base_url == "http://localhost:9200"
        assert client.api_key == "secret123"

    @patch("urllib.request.urlopen")
    def test_ping_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"tagline": "You Know, for Search"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = XerjEWSClient()
        assert client.ping() is True

    @patch("urllib.request.urlopen")
    def test_ping_offline(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = XerjEWSClient(base_url="http://127.0.0.1:9999")
        assert client.ping() is False

    @patch("urllib.request.urlopen")
    def test_create_credit_anomaly_detector(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "job_id": "credit-turnover-drop",
            "job_type": "anomaly_detector",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = XerjEWSClient()
        res = client.create_credit_anomaly_detector(
            job_id="credit-turnover-drop",
            source_index="corporate_turnovers",
            field="turnover_uzs",
            bucket_span="1d",
            threshold=3.0,
        )
        assert res.get("job_id") == "credit-turnover-drop"

    @patch("urllib.request.urlopen")
    def test_create_and_manage_datafeed(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "datafeed_id": "credit-turnover-feed",
            "job_id": "credit-turnover-drop",
            "state": "stopped",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = XerjEWSClient()
        feed = client.create_datafeed("credit-turnover-feed", "credit-turnover-drop", frequency="5s")
        assert feed.get("datafeed_id") == "credit-turnover-feed"

        # Test start
        mock_resp.read.return_value = json.dumps({"started": True}).encode("utf-8")
        start_res = client.start_datafeed("credit-turnover-feed")
        assert start_res.get("started") is True

        # Test stop
        mock_resp.read.return_value = json.dumps({"stopped": True}).encode("utf-8")
        stop_res = client.stop_datafeed("credit-turnover-feed")
        assert stop_res.get("stopped") is True

    @patch("urllib.request.urlopen")
    def test_ingest_telemetry_batch(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "items": [{"index": {"status": 201}}, {"index": {"status": 201}}],
            "errors": False,
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = XerjEWSClient()
        records = [
            {"@timestamp": "2026-10-24T00:00:00Z", "inn": "300001000", "turnover_uzs": 50000000.0},
            {"@timestamp": "2026-10-24T01:00:00Z", "inn": "300001000", "turnover_uzs": 48000000.0},
        ]
        count = client.ingest_telemetry_batch(records, index_name="corporate_turnovers")
        assert count == 2

    @patch("urllib.request.urlopen")
    def test_borrower_memory_store_and_recall(self, mock_urlopen):
        mock_resp = MagicMock()
        # Test store
        mock_resp.read.return_value = json.dumps({
            "id": "mem-101",
            "namespace": "300001000",
            "created": True,
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = XerjEWSClient()
        res_store = client.store_borrower_memory(
            inn="300001000",
            note_text="Zavodda modernizatsiya ishlari ketmoqda, asosiy texnologik liniya vaqtincha to'xtatilgan.",
            metadata={"monitoring_type": "favqulodda", "officer": "Xajibayev A."},
        )
        assert res_store.get("id") == "mem-101"

        # Test recall
        mock_resp.read.return_value = json.dumps({
            "hits": [
                {
                    "id": "mem-101",
                    "text": "Zavodda modernizatsiya ishlari ketmoqda",
                    "score": 0.89,
                }
            ]
        }).encode("utf-8")
        hits = client.recall_borrower_memory(inn="300001000", query="ishlab chiqarish to'xtagan")
        assert len(hits) == 1
        assert "modernizatsiya" in hits[0]["text"]

    @patch("urllib.request.urlopen")
    def test_sync_portfolio_borrowers(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "items": [{"index": {"status": 201}}, {"index": {"status": 201}}],
            "errors": False,
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        df = pd.DataFrame([
            {"inn": "300001000", "company_name": "Agro Tex Klaster MCHJ", "loan_balance_uzs": 15000000000.0},
            {"inn": "300002000", "company_name": "Toshkent Shisha Zavodi AJ", "loan_balance_uzs": 8500000000.0},
        ])
        client = XerjEWSClient()
        count = client.sync_portfolio_borrowers(df)
        assert count == 2
