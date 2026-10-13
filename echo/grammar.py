import time

from google import genai

SYSTEM_INSTRUCTION = (
    "You are a careful editor. Correct spelling, grammar, punctuation, and "
    "syntax in the user's text. Do not paraphrase, reword, or change the "
    "meaning, tone, or style. Fix only what is actually wrong. If the text is "
    "already correct, return it unchanged. Return only the corrected text with "
    "no commentary, quotes, or markdown."
)

_PER_MODEL_ATTEMPTS = 1
_DEFAULT_FALLBACKS = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


class GrammarError(RuntimeError):
    pass


class GrammarChecker:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallbacks: list[str] | None = None,
    ) -> None:
        self._models = [model, * (fallbacks or _DEFAULT_FALLBACKS)]
        self._client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 45000},
        )

    def correct(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text

        last_exc: Exception | None = None
        for model in self._models:
            for attempt in range(_PER_MODEL_ATTEMPTS):
                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=text,
                        config={
                            "system_instruction": SYSTEM_INSTRUCTION,
                            "temperature": 0.2,
                        },
                    )
                    return (response.text or "").strip()
                except Exception as exc:  # noqa: BLE001 - normalize retries below
                    last_exc = exc
                    if _is_transient(exc):
                        time.sleep(2 ** attempt)
                        continue
                    break

        assert last_exc is not None
        raise GrammarError(_friendly_message(last_exc))


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code in (429, 500, 502, 503)


def _friendly_message(exc: Exception) -> str:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 404:
        return "Gemini model not found. Check GEMINI_MODEL in .env."
    if code == 429:
        return "Gemini free-tier quota exceeded. Wait a bit and try again."
    if code in (500, 502, 503, 504):
        return "Gemini is temporarily unavailable. Try again in a moment."
    return f"{type(exc).__name__}: {exc}"