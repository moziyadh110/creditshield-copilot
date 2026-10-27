"""Personal data: Presidio masks the applicant note before any LLM call, and the same analyzer
checks the LLM's output. Uses spaCy's small English model (en_core_web_sm), set up once.
"""

from functools import cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from app.schemas import CheckResult

# Identifiers that must never reach the provider or appear in a notice.
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_STREET_ADDRESS",
]
# The output check skips LOCATION: spaCy's small model tags ordinary capitalised words
# ("Plain") as places, and a false block costs a good notice. Street addresses are still caught
# by US_STREET_ADDRESS. Masking the input keeps LOCATION, where a false positive costs nothing.
OUTPUT_ENTITIES = [e for e in PII_ENTITIES if e != "LOCATION"]
MIN_SCORE = 0.5
_STREET = (
    r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b\.?"
)
_PHONE = r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)"


@cache
def _analyzer() -> AnalyzerEngine:
    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    # Presidio has no US street-address recognizer, and its phone recognizer rejects numbers
    # that fail validation (e.g. 555 numbers), so both get a plain pattern as well.
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_STREET_ADDRESS",
            patterns=[Pattern("street address", _STREET, 0.7)],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER", patterns=[Pattern("us phone", _PHONE, 0.6)]
        )
    )
    return analyzer


def find_pii(text: str, entities: list[str] = PII_ENTITIES) -> list[tuple[str, str]]:
    """(entity type, matched text) for every identifier found."""
    results = _analyzer().analyze(
        text=text, entities=entities, language="en", score_threshold=MIN_SCORE
    )
    return [(r.entity_type, text[r.start : r.end]) for r in results]


def mask_pii(text: str) -> str:
    """Replace each identifier with its type, e.g. 'call Jane on 555-0100' -> 'call <PERSON> on
    <PHONE_NUMBER>'."""
    results = _analyzer().analyze(
        text=text, entities=PII_ENTITIES, language="en", score_threshold=MIN_SCORE
    )
    return AnonymizerEngine().anonymize(text=text, analyzer_results=results).text


def check_personal_data(texts: list[str]) -> CheckResult:
    found = [f"{kind}: '{value}'" for t in texts for kind, value in find_pii(t, OUTPUT_ENTITIES)]
    # A mask token copied into the output means the model echoed the applicant's note.
    found += [f"mask token {tok}" for t in texts for tok in _mask_tokens(t)]
    return CheckResult(
        name="personal_data", passed=not found, detail="; ".join(found), on_fail="escalate"
    )


def _mask_tokens(text: str) -> list[str]:
    return [f"<{e}>" for e in PII_ENTITIES if f"<{e}>" in text]
