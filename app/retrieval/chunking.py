"""Split the saved Reg B text into section-aware chunks.

12 CFR 1002.9 is cut at its second level, e.g. 1002.9(a)(2), with sub-items (i), (A) kept
inside their parent. Appendix C becomes one chunk per numbered instruction and one per form
(C-1 to C-5). Every chunk starts with its citation, so a retrieved chunk is self-describing.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegChunk:
    section_id: str
    title: str
    source_url: str
    text: str


# "(a) Heading—(1) Heading. Body" or "(d) Heading. Body"; letters stop before (i), a roman numeral.
LETTER = re.compile(r"^\(([a-h])\) (.+?)(?:—\((\d+)\) (.+?))?\. ")
NUMBER = re.compile(r"^\((\d+)\) (.+?)\. ")
FORM = re.compile(r"^Form (C-\d+)—(.+)$")


def _read(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    source = next(s.removeprefix("# source: ") for s in lines if s.startswith("# source: "))
    return source, [s for s in lines if s and not s.startswith("#")]


def chunk_section_1002_9(path: Path) -> list[RegChunk]:
    source, lines = _read(path)
    groups: list[tuple[str, str, list[str]]] = []
    letter = letter_title = ""
    for line in lines[1:]:  # lines[0] is the section heading
        if m := LETTER.match(line):
            letter, letter_title, number, number_title = m.groups()
            sid = f"1002.9({letter})" + (f"({number})" if number else "")
            title = f"{letter_title} — {number_title}" if number else letter_title
            groups.append((sid, title, [line]))
        elif m := NUMBER.match(line):
            groups.append((f"1002.9({letter})({m[1]})", f"{letter_title} — {m[2]}", [line]))
        else:
            groups[-1][2].append(line)
    return [
        RegChunk(sid, title, source, "\n".join([f"12 CFR § {sid} {title}", *body]))
        for sid, title, body in groups
    ]


def chunk_appendix_c(path: Path) -> list[RegChunk]:
    source, lines = _read(path)
    groups: list[tuple[str, str, list[str]]] = []
    for line in lines[1:]:  # lines[0] is the appendix heading
        if m := FORM.match(line):
            groups.append((f"1002-appC-{m[1]}", f"Form {m[1]}: {m[2]}", []))
        elif not groups or groups[-1][0].startswith("1002-appC-instr"):
            # Each numbered instruction is its own chunk: together they exceed the
            # embedding model's 512-token input and the tail would be silently cut.
            n = line.split(".", 1)[0]
            groups.append((f"1002-appC-instr-{n}", f"Appendix C instruction {n}", [line]))
        else:
            groups[-1][2].append(line)
    return [
        RegChunk(sid, title, source, "\n".join([f"12 CFR Part 1002, Appendix C, {title}", *body]))
        for sid, title, body in groups
    ]


def chunk_regb(regb_dir: Path) -> list[RegChunk]:
    return chunk_section_1002_9(regb_dir / "1002.9.txt") + chunk_appendix_c(
        regb_dir / "appendix_c.txt"
    )
