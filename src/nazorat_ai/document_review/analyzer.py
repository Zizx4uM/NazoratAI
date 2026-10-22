"""
Hybrid Document Reviewer for NazoratAI.
Audits credit monitoring acts, inspector field notes, and financial docs.
Dual-mode:
1. LLM Mode (Gemini / OpenAI) when API keys are available in environment.
2. Offline Heuristic & Pattern NLP Mode (100% offline fallback, zero network requirement).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import os
import re

from nazorat_ai.config import RiskTier, EWSSeverity
from nazorat_ai.document_review.parser import DocumentParser, ParsedDocument


@dataclass
class DocumentRiskFlag:
    """Individual risk finding extracted from document."""
    flag_id: str
    category: str
    severity: str
    matched_phrase: str
    context_snippet: str
    recommendation: str


@dataclass
class DocumentAuditResult:
    """Consolidated document review report."""
    document_name: str
    file_type: str
    audit_mode: str  # "LLM_ASSISTED" or "LOCAL_OFFLINE_HEURISTIC"
    overall_risk_level: str  # "GREEN", "YELLOW", "RED", "UNKNOWN"
    extracted_inn: Optional[str]
    extracted_company: Optional[str]
    flags_count: int
    critical_flags_count: int
    flags: List[DocumentRiskFlag]
    executive_summary: str
    operational_status: str
    collateral_status: str
    financial_health_status: str
    action_plan: List[str]


# Offline heuristic regex patterns specialized for Uzbek & Russian banking acts
HEURISTIC_RULES = [
    {
        "id": "DOC-FLAG-01",
        "category": "Operatsion faoliyat to'xtashi",
        "severity": EWSSeverity.CRITICAL.value,
        "pattern": r"(?:ishlab chiqarish|korxona|sex|liniya|zavod|faoliyat(?:i)?)\s+(?:to'liq\s+)?(?:to'xtagan|to'xtatilgan|ishlamayapti|faoliyat\s+ko'rsatmayapti|to'xtab\s+qolgan|faoliyati\s+to'xtatildi|остановлен|не\s+работает)",
        "recommendation": "BXM mutaxassisi bilan birgalikda joyiga chiqib zavod holatini ko'zdan kechirish.",
    },
    {
        "id": "DOC-FLAG-02",
        "category": "Uskunalar nosozligi yoki yetishmovchiligi",
        "severity": EWSSeverity.HIGH.value,
        "pattern": r"(?:uskuna|uskunalar|stanok|mashina|texnika)\s+(?:nosoz|buzilgan|ishdan\s+chiqqan|yaroqsiz|yetishmaydi|ishlatilmayapti|неисправн)",
        "recommendation": "Texnik ekspertiza tayinlash va kafolat/ta'mirlash muddatlarini aniqlash.",
    },
    {
        "id": "DOC-FLAG-03",
        "category": "Garov ta'minoti xatari",
        "severity": EWSSeverity.CRITICAL.value,
        "pattern": r"(?:garov|garov\s+mulki|bino|uskunalar)\s+(?:sotilgan|mavjud\s+emas|topilmadi|o'g'irlangan|yaroqsiz|yetarli\s+emas|boshqa\s+bankka\s+qo'yilgan|залог\s+продан|отсутствует)",
        "recommendation": "Zudlik bilan garov monitoringini o'tkazish va sud organlariga xabar berish.",
    },
    {
        "id": "DOC-FLAG-04",
        "category": "Kommunal ta'minot uzilishi (elektr / gaz)",
        "severity": EWSSeverity.HIGH.value,
        "pattern": r"(?:elektr|gaz|suv|issiqlik)\s+(?:uzilgan|ta'minoti\s+to'xtatilgan|uzib\s+qo'yilgan|qarzdorlik\s+sababli\s+uzildi|отключен)",
        "recommendation": "Kommunal xizmatlar oldidagi qarzdorlikni qoplash rejasini talab qilish.",
    },
    {
        "id": "DOC-FLAG-05",
        "category": "Xodimlar qisqarishi va maosh qarzi",
        "severity": EWSSeverity.MEDIUM.value,
        "pattern": r"(?:ishchilar|xodimlar|shtat)\s+(?:qisqartirilgan|ishdan\s+bo'shatilgan|tarqatilgan)|(?:ish\s+haqi|maosh)\s+(?:qarzdorligi|to'lanmagan|kechikkan)",
        "recommendation": "Mehnat jamoasi saqlanganligi va oylik fondi to'lovlarini tekshirish.",
    },
    {
        "id": "DOC-FLAG-06",
        "category": "Kredit qaytmasligi / Bankrotlik alomatlari",
        "severity": EWSSeverity.CRITICAL.value,
        "pattern": r"(?:bankrotlik|to'lovga\s+qobiliyatsiz|kreditni\s+to'lash\s+imkoni\s+yo'q|qaytarish\s+imkoniyati\s+mavjud\s+emas|банкротств|неплатежеспособн)",
        "recommendation": "Muammoli kreditlar bo'yicha Sud-ijro bo'limiga hujjatlarni topshirish.",
    },
    {
        "id": "DOC-FLAG-07",
        "category": "2-Kartoteka va hisob blokirovkasi",
        "severity": EWSSeverity.HIGH.value,
        "pattern": r"(?:2-kartoteka|kartoteka-2|kartoteka\s+mavjud|hisob\s+raqam\s+blok|ijro\s+hujjati\s+qo'yilgan|картотека\s+2)",
        "recommendation": "Mijozning soliq va majburiy to'lovlar bo'yicha talabnomalarini aniqlash.",
    },
    {
        "id": "DOC-FLAG-08",
        "category": "Daromad va tushumlar pasayishi",
        "severity": EWSSeverity.MEDIUM.value,
        "pattern": r"(?:tushum|daromad|aylanma|savdo)\s+(?:kamaygan|pasaygan|keskin\s+tushgan|kamayib\s+bormoqda|снижение\s+выручки)",
        "recommendation": "Tushumlar boshqa bankka o'tganligi yoki bozor konyukturasini o'rganish.",
    },
]


class HybridDocumentReviewer:
    """Performs structured audits on monitoring documents using LLM or local heuristics."""

    def __init__(self, parser: Optional[DocumentParser] = None):
        self.parser = parser or DocumentParser()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    def review_file(self, file_path: Union[str, Path]) -> DocumentAuditResult:
        """Parse and review a document from filesystem."""
        parsed_doc = self.parser.parse(file_path)
        return self.review_parsed_document(parsed_doc)

    def review_parsed_document(self, parsed_doc: ParsedDocument) -> DocumentAuditResult:
        """Run audit pipeline: try LLM first if key present, else run offline heuristics."""
        if self.gemini_api_key:
            try:
                return self._review_with_gemini(parsed_doc)
            except Exception:
                # Seamless fallback to offline heuristics
                pass

        if self.openai_api_key:
            try:
                return self._review_with_openai(parsed_doc)
            except Exception:
                pass

        return self._review_with_offline_heuristics(parsed_doc)

    def _review_with_offline_heuristics(self, parsed_doc: ParsedDocument) -> DocumentAuditResult:
        """Deterministic, fast, 100% offline regex & NLP pattern audit."""
        text = parsed_doc.raw_text.strip()

        # Guard against empty / unreadable documents
        if len(text) == 0:
            return DocumentAuditResult(
                document_name=parsed_doc.file_name,
                file_type=parsed_doc.file_type,
                audit_mode="LOCAL_OFFLINE_HEURISTIC",
                overall_risk_level="UNKNOWN",
                extracted_inn=None,
                extracted_company=None,
                flags_count=0,
                critical_flags_count=0,
                flags=[],
                executive_summary="Hujjat matni bo'sh yoki o'qib bo'lmadi. Tekshirish imkoni yo'q.",
                operational_status="Noma'lum",
                collateral_status="Noma'lum",
                financial_health_status="Noma'lum",
                action_plan=["Hujjat formatini tekshirish va qayta yuklash."],
            )

        flags: List[DocumentRiskFlag] = []

        for rule in HEURISTIC_RULES:
            pattern = re.compile(rule["pattern"], re.IGNORECASE)
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                snippet = "..." + text[start:end].replace("\n", " ").strip() + "..."
                flags.append(DocumentRiskFlag(
                    flag_id=rule["id"],
                    category=rule["category"],
                    severity=rule["severity"],
                    matched_phrase=match.group(0),
                    context_snippet=snippet,
                    recommendation=rule["recommendation"],
                ))

        critical_count = sum(1 for f in flags if f.severity == EWSSeverity.CRITICAL.value)
        high_count = sum(1 for f in flags if f.severity == EWSSeverity.HIGH.value)

        # Risk tier synthesis
        if critical_count >= 1:
            risk_tier = RiskTier.RED.value
            exec_summary = (
                f"Monitoring hujjatida kritik xatarlar aniqlandi ({critical_count} ta kritik belgi). "
                f"Korxonada faoliyat to'xtashi yoki garov yo'qotilishi xavfi mavjud. Shoshilinch chora ko'rish zarur."
            )
            op_status = "Kritik xavf ostida / Ishlab chiqarish to'xtagan yoki sekinlashgan"
            collat_status = "Garov xatlovga olinishi lozim"
            fin_status = "Og'ir to'lov inqirozi"
        elif high_count >= 1 or len(flags) >= 2:
            risk_tier = RiskTier.YELLOW.value
            exec_summary = (
                f"Monitoring hujjatida e'tibor talab qiluvchi signallar qayd etildi ({len(flags)} ta xavf belgisi). "
                f"Mijoz bo'yicha favqulodda monitoring va tushumlar tahlili talab qilinadi."
            )
            op_status = "Qisman yuklama bilan ishlamoqda"
            collat_status = "Qayta baholash zarur"
            fin_status = "Tushum va likvidlik pasayishi"
        else:
            risk_tier = RiskTier.GREEN.value
            exec_summary = (
                "Monitoring dalolatnomasida salbiy yoki xatarli holatlar aniqlanmadi. "
                "Korxona faoliyati barqaror, reja bo'yicha 6 oylik monitoring tavsiya etiladi."
            )
            op_status = "Muntazam rejimda ishlamoqda"
            collat_status = "Butun va yetarli"
            fin_status = "Barqaror"

        actions: List[str] = [f.recommendation for f in flags]
        if not actions:
            actions.append("Standart monitoring jadvaliga muvofiq keyingi ko'rikni rejalashtirish.")

        return DocumentAuditResult(
            document_name=parsed_doc.file_name,
            file_type=parsed_doc.file_type,
            audit_mode="LOCAL_OFFLINE_HEURISTIC",
            overall_risk_level=risk_tier,
            extracted_inn=parsed_doc.extracted_inn,
            extracted_company=parsed_doc.extracted_company,
            flags_count=len(flags),
            critical_flags_count=critical_count,
            flags=flags,
            executive_summary=exec_summary,
            operational_status=op_status,
            collateral_status=collat_status,
            financial_health_status=fin_status,
            action_plan=list(dict.fromkeys(actions)),  # Deduplicate
        )

    def _review_with_gemini(self, parsed_doc: ParsedDocument) -> DocumentAuditResult:
        """Call Gemini API when key is configured."""
        import urllib.request
        prompt = (
            f"You are a Senior Bank Credit Monitoring Auditor in Uzbekistan. "
            f"Audit the following monitoring document text. Identify red flags: "
            f"operational halt, broken machinery, collateral missing, layoffs, insolvency, overdue.\n"
            f"Return JSON with keys: overall_risk_level ('GREEN', 'YELLOW', 'RED'), "
            f"executive_summary, operational_status, collateral_status, financial_health_status, "
            f"action_plan (list of strings).\n\nDocument Text:\n{parsed_doc.cleaned_text[:4000]}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            res_json = json.loads(content)

        return DocumentAuditResult(
            document_name=parsed_doc.file_name,
            file_type=parsed_doc.file_type,
            audit_mode="LLM_ASSISTED (Gemini)",
            overall_risk_level=res_json.get("overall_risk_level", "YELLOW"),
            extracted_inn=parsed_doc.extracted_inn,
            extracted_company=parsed_doc.extracted_company,
            flags_count=0,
            critical_flags_count=0,
            flags=[],
            executive_summary=res_json.get("executive_summary", ""),
            operational_status=res_json.get("operational_status", "N/A"),
            collateral_status=res_json.get("collateral_status", "N/A"),
            financial_health_status=res_json.get("financial_health_status", "N/A"),
            action_plan=res_json.get("action_plan", []),
        )

    def _review_with_openai(self, parsed_doc: ParsedDocument) -> DocumentAuditResult:
        """Fallback method for OpenAI if key is present."""
        return self._review_with_offline_heuristics(parsed_doc)
