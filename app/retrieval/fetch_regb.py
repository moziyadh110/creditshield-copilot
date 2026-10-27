"""Download Reg B 12 CFR 1002.9 and Appendix C (forms C-1 to C-5) from eCFR into data/regb/.

The saved text files are committed, so ingest never needs the network. Re-run this only to
pick up an amendment:  uv run python -m app.retrieval.fetch_regb
"""

import re
import xml.etree.ElementTree as ET

import httpx

from app.config import get_settings

API = "https://www.ecfr.gov/api/versioner/v1"
PUBLIC = "https://www.ecfr.gov/current/title-12/chapter-X/part-1002"

DOCUMENTS = {
    "1002.9.txt": {
        "query": {"part": "1002", "section": "1002.9"},
        "source": f"{PUBLIC}/section-1002.9",
    },
    "appendix_c.txt": {
        "query": {"part": "1002", "appendix": "Appendix C to Part 1002"},
        "source": f"{PUBLIC}/appendix-Appendix%20C%20to%20Part%201002",
    },
}

# Forms C-1 to C-5 are the credit-denial notices; C-6 onwards (other notices) are not kept.
FIRST_DROPPED_FORM = "Form C-6"


def latest_issue_date(client: httpx.Client) -> str:
    titles = client.get(f"{API}/titles.json").raise_for_status().json()["titles"]
    return next(t for t in titles if t["number"] == 12)["up_to_date_as_of"]


def xml_to_lines(xml: str) -> list[str]:
    """One line per heading or paragraph, whitespace collapsed, in document order."""
    lines = []
    for el in ET.fromstring(xml).iter():
        if el.tag in {"HEAD", "HD1", "P", "FP"}:
            text = re.sub(r"\s+", " ", "".join(el.itertext())).strip()
            if text:
                lines.append(text)
    return lines


def main() -> None:
    out_dir = get_settings().regb_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30) as client:
        as_of = latest_issue_date(client)
        for filename, doc in DOCUMENTS.items():
            url = f"{API}/full/{as_of}/title-12.xml"
            lines = xml_to_lines(client.get(url, params=doc["query"]).raise_for_status().text)
            if filename == "appendix_c.txt":
                cut = next(i for i, s in enumerate(lines) if s.startswith(FIRST_DROPPED_FORM))
                lines = lines[:cut]
            header = [f"# source: {doc['source']}", f"# as_of: {as_of}", ""]
            (out_dir / filename).write_text("\n".join(header + lines) + "\n", encoding="utf-8")
            print(f"{filename}: {len(lines)} paragraphs (eCFR as of {as_of})")


if __name__ == "__main__":
    main()
