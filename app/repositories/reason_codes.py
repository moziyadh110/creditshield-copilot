"""Reads data/reason_codes.yaml: the plain wording and banned wordings for every model feature."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from app.config import get_settings


class ReasonCode(BaseModel):
    code: str
    source: Literal["application", "bureau", "history"]
    reason: str | None
    banned: list[str]
    escalate: bool
    sensitive: bool

    @model_validator(mode="after")
    def _worded_unless_escalated(self) -> "ReasonCode":
        if self.escalate and self.reason is not None:
            raise ValueError(f"{self.code}: escalate-only features must not have a reason wording")
        if not self.escalate and not self.reason:
            raise ValueError(f"{self.code}: needs a reason wording")
        return self


class ReasonCatalog(BaseModel):
    version: int
    features: dict[str, ReasonCode]

    @model_validator(mode="after")
    def _codes_unique(self) -> "ReasonCatalog":
        codes = [rc.code for rc in self.features.values()]
        if len(codes) != len(set(codes)):
            raise ValueError("reason codes must be unique")
        return self


def load_reason_catalog(path: Path) -> ReasonCatalog:
    return ReasonCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache
def get_reason_catalog() -> ReasonCatalog:
    return load_reason_catalog(get_settings().reason_codes_path)
