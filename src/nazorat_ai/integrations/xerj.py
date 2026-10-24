"""
XERJ Search & Anomaly Detection Integration for NazoratAI.

Provides high-speed Elasticsearch-compatible integration with XERJ:
1. Dynamic early warning anomaly detection (_ml/anomaly_detectors & _ml/datafeeds)
   for borrower daily account turnovers, 2-Kartoteka debt velocity, and utility streams.
2. Long-term qualitative borrower memory (/_memory/{inn}) for field inspection acts.
3. Sub-second hybrid search across corporate borrower profiles and inspection acts.

Developed jointly for NazoratAI and the XERJ ecosystem.
Author: Xajibayev Azizbek (@Zizx4uM) & XERJ Community.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error
import pandas as pd

logger = logging.getLogger(__name__)


class XerjEWSClient:
    """
    Client connecting NazoratAI to a local or remote XERJ search & anomaly engine.
    Communicates via standard HTTP with zero external binary dependencies.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9200",
        api_key: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Any] = None,
        is_ndjson: bool = False,
    ) -> Dict[str, Any]:
        """Execute HTTP request against the XERJ engine."""
        url = f"{self.base_url}{path}"
        headers: Dict[str, str] = {}

        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"

        data: Optional[bytes] = None
        if body is not None:
            if is_ndjson or isinstance(body, str):
                data = body.encode("utf-8") if isinstance(body, str) else str(body).encode("utf-8")
                headers["Content-Type"] = "application/x-ndjson"
            else:
                data = json.dumps(body).encode("utf-8")
                headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            logger.warning("XERJ HTTP %s error on %s: %s", exc.code, path, err_body)
            raise ConnectionError(f"XERJ returned HTTP {exc.code} on {path}: {err_body}") from exc
        except urllib.error.URLError as exc:
            logger.debug("Failed to connect to XERJ at %s: %s", url, exc.reason)
            raise ConnectionError(f"Could not connect to XERJ at {url}: {exc.reason}") from exc

    def ping(self) -> bool:
        """Verify whether the XERJ node is alive and answering."""
        try:
            res = self._request("GET", "/")
            return bool(res.get("tagline") or res.get("version") or "cluster_name" in res)
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Anomaly Detection & Continuous Datafeeds (_ml)
    # -------------------------------------------------------------------------

    def create_credit_anomaly_detector(
        self,
        job_id: str = "credit-turnover-drop",
        source_index: str = "corporate_turnovers",
        time_field: str = "@timestamp",
        field: str = "turnover_uzs",
        function: str = "mean",
        bucket_span: str = "1d",
        threshold: float = 3.0,
        description: str = "Corporate credit turnover moving z-score anomaly detector",
    ) -> Dict[str, Any]:
        """
        Create a moving-window z-score anomaly detector on borrower metrics.
        Supported functions: mean, count, min, max, sum.
        """
        payload = {
            "source_index": source_index,
            "time_field": time_field,
            "function": function,
            "field": field,
            "bucket_span": bucket_span,
            "anomaly_threshold": threshold,
            "description": description,
        }
        return self._request("PUT", f"/_ml/anomaly_detectors/{job_id}", body=payload)

    def create_datafeed(
        self,
        datafeed_id: str,
        job_id: str,
        frequency: str = "5s",
        indices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a continuous background datafeed watching an anomaly detector."""
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "frequency": frequency,
        }
        if indices:
            payload["indices"] = indices
        return self._request("PUT", f"/_ml/datafeeds/{datafeed_id}", body=payload)

    def start_datafeed(self, datafeed_id: str) -> Dict[str, Any]:
        """Start a continuous anomaly scoring datafeed background task."""
        return self._request("POST", f"/_ml/datafeeds/{datafeed_id}/_start")

    def stop_datafeed(self, datafeed_id: str) -> Dict[str, Any]:
        """Stop a continuous anomaly scoring datafeed."""
        return self._request("POST", f"/_ml/datafeeds/{datafeed_id}/_stop")

    def score_anomalies(self, job_id: str) -> List[Dict[str, Any]]:
        """Score the anomaly detector series synchronously on-demand."""
        res = self._request("POST", f"/_ml/anomaly_detectors/{job_id}/_score")
        return res.get("records", [])

    def get_datafeed_records(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve recent anomalies flagged by the background datafeed."""
        res = self._request("GET", f"/_ml/anomaly_detectors/{job_id}/results/records")
        return res.get("records", [])

    def ingest_telemetry_batch(
        self,
        records: List[Dict[str, Any]],
        index_name: str = "corporate_turnovers",
    ) -> int:
        """
        Bulk ingest timestamped borrower telemetry (turnovers, electricity, K-2 arrears).
        """
        if not records:
            return 0

        ndjson_lines: List[str] = []
        for r in records:
            action = {"index": {"_index": index_name}}
            ndjson_lines.append(json.dumps(action))
            ndjson_lines.append(json.dumps(r))

        body = "\n".join(ndjson_lines) + "\n"
        res = self._request("POST", "/_bulk?refresh=true", body=body, is_ndjson=True)
        items = res.get("items", [])
        return len(items)

    # -------------------------------------------------------------------------
    # Qualitative Borrower Memory (/_memory/{inn})
    # -------------------------------------------------------------------------

    def store_borrower_memory(
        self,
        inn: str,
        note_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save qualitative field monitoring observations or credit committee notes into XERJ memory.
        Isolated to namespace /_memory/{inn}.
        """
        payload: Dict[str, Any] = {"text": note_text}
        if metadata:
            payload["metadata"] = metadata
        if vector:
            payload["vector"] = vector
        if memory_id:
            payload["id"] = memory_id

        return self._request("POST", f"/_memory/{inn}", body=payload)

    def recall_borrower_memory(
        self,
        inn: str,
        query: str,
        k: int = 5,
        semantic: bool = False,
        hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant historical field notes for a borrower using BM25, semantic, or hybrid fusion.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "k": k,
        }
        if hybrid:
            payload["hybrid"] = True
        elif semantic:
            payload["semantic"] = True

        res = self._request("POST", f"/_memory/{inn}/_recall", body=payload)
        return res.get("hits", [])

    def list_borrower_memories(self, inn: str) -> List[Dict[str, Any]]:
        """List stored memories for a borrower namespace."""
        res = self._request("GET", f"/_memory/{inn}")
        return res.get("entries", [])

    # -------------------------------------------------------------------------
    # Corporate Portfolio Search & Indexing
    # -------------------------------------------------------------------------

    def sync_portfolio_borrowers(
        self,
        df: pd.DataFrame,
        index_name: str = "corporate_borrowers",
    ) -> int:
        """
        Index corporate portfolio records into XERJ for sub-millisecond Elasticsearch queries.
        """
        records = df.to_dict(orient="records")
        return self.ingest_telemetry_batch(records, index_name=index_name)

    def search_borrowers(
        self,
        query_str: str,
        index_name: str = "corporate_borrowers",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search corporate borrower accounts with full-text BM25 matching."""
        payload = {
            "query": {
                "multi_match": {
                    "query": query_str,
                    "fields": ["company_name", "inn", "sector", "region", "branch_name"],
                }
            },
            "size": limit,
        }
        res = self._request("POST", f"/{index_name}/_search", body=payload)
        hits = res.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]
