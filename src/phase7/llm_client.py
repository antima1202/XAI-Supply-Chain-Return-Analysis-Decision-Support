"""
llm_client.py  –  Gemini API client for the Phase 7 recommendation engine.

Uses the new google-genai SDK (pip install google-genai).
Reads GEMINI_API_KEY from environment — never hardcoded.
Falls back to structured templates if the API is unavailable.
"""

import os
import time
import textwrap

from .utils import get_logger

logger = get_logger("phase7.llm_client")

GEMINI_MODEL      = "gemini-2.0-flash"
MAX_RETRIES       = 3
RETRY_DELAY       = 5
MAX_OUTPUT_TOKENS = 1500


class GeminiClient:
    """
    Wrapper around the Google Gemini API (google-genai SDK).

    Set your API key before running:
        Windows: $env:GEMINI_API_KEY = "your-key-here"
        Mac/Linux: export GEMINI_API_KEY="your-key-here"
    """

    def __init__(self, api_key: str | None = None):
        self.api_key   = api_key or os.getenv("GEMINI_API_KEY", "")
        self.available = False
        self._client   = None

        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY not set — using structured fallback templates. "
                "Set the environment variable to enable live Gemini generation."
            )
            return

        try:
            from google import genai
            self._client   = genai.Client(api_key=self.api_key)
            self.available = True
            logger.info("Gemini client initialised  (model: %s)", GEMINI_MODEL)
        except ImportError:
            logger.warning("google-genai not installed. Run: pip install google-genai")
        except Exception as exc:
            logger.warning("Gemini initialisation failed: %s", exc)

    def generate(self, prompt: str, label: str = "LLM call") -> str:
        """
        Send a prompt to Gemini and return the generated text.
        Falls back to a structured template if the API is unavailable.
        """
        if not self.available or self._client is None:
            logger.info("Using fallback template for: %s", label)
            return self._fallback(prompt)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("Gemini call (%s) — attempt %d/%d", label, attempt, MAX_RETRIES)
                from google import genai
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.4,
                        top_p=0.85,
                    ),
                )
                text = response.text.strip()
                logger.info("Gemini response received (%d chars)", len(text))
                return text

            except Exception as exc:
                logger.warning("Gemini attempt %d failed: %s", attempt, exc)
                if attempt < MAX_RETRIES:
                    logger.info("Retrying in %d seconds …", RETRY_DELAY * attempt)
                    time.sleep(RETRY_DELAY * attempt)

        logger.warning("All Gemini retries exhausted — using fallback for: %s", label)
        return self._fallback(prompt)

    def _fallback(self, prompt: str) -> str:
        """Structured placeholder when the LLM is unavailable."""
        rc_line = [l for l in prompt.split("\n") if "Root cause category" in l]
        rc = rc_line[0].split(":")[-1].strip() if rc_line else "Unknown"

        return textwrap.dedent(f"""
        [FALLBACK — Gemini API unavailable]

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

        NOTE: Set GEMINI_API_KEY environment variable to enable live AI recommendations.
        """).strip()
