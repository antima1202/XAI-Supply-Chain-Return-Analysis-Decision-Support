"""
llm_client.py  –  LLM client using Ollama (local, no API key needed).
Falls back to structured templates if Ollama is unavailable.
"""

import os
import textwrap
from .utils import get_logger

logger = get_logger("phase7.llm_client")

OLLAMA_MODEL = "llama3.2"
MAX_TOKENS   = 1000


class GeminiClient:
    """
    Local LLM client using Ollama.
    Class kept as GeminiClient for compatibility with existing code.
    """

    def __init__(self, api_key: str | None = None):
        self.available = False
        self._client   = None

        try:
            import ollama
            # Test connection
            ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": "hi"}],
            )
            self._ollama = ollama
            self.available = True
            logger.info("Ollama client initialised  (model: %s)", OLLAMA_MODEL)
        except Exception as exc:
            logger.warning(
                "Ollama unavailable: %s. "
                "Make sure Ollama is running and model is pulled. "
                "Run: ollama pull %s", exc, OLLAMA_MODEL
            )

    def generate(self, prompt: str, label: str = "LLM call") -> str:
        if not self.available:
            logger.info("Using fallback template for: %s", label)
            return self._fallback(prompt)

        try:
            logger.info("Ollama call (%s)", label)
            response = self._ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": MAX_TOKENS, "temperature": 0.4},
            )
            text = response["message"]["content"].strip()
            logger.info("Ollama response received (%d chars)", len(text))
            return text
        except Exception as exc:
            logger.warning("Ollama call failed: %s — using fallback", exc)
            return self._fallback(prompt)

    def _fallback(self, prompt: str) -> str:
        rc_line = [l for l in prompt.split("\n") if "Root cause category" in l]
        rc = rc_line[0].split(":")[-1].strip() if rc_line else "Unknown"
        return textwrap.dedent(f"""
        [FALLBACK — Ollama unavailable]

        RECOMMENDATION FOR: {rc.upper()}

        SITUATION SUMMARY:
        This root cause has been identified by the AI framework as requiring
        operational attention based on frequency, impact, and trend analysis.

        IMMEDIATE ACTIONS (next 30 days):
        1. Review all confirmed returns in this category for the past 30 days
        2. Identify the top 3 products generating returns in this category
        3. Convene a cross-functional team meeting with the responsible stakeholder

        MEDIUM-TERM ACTIONS (30-90 days):
        1. Implement process changes identified from the return review
        2. Establish monthly KPI tracking for this root cause category

        EXPECTED IMPACT:
        - Refer to the risk score table for quantified reduction estimates
        - Track return rate for this category before and after intervention
        """).strip()