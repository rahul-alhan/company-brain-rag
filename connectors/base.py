"""Connector ABC + normalized Document schema.

Every source (Slack/Notion/Drive/...) normalizes to the same Document so the
retrieval pipeline is source-agnostic. ACL principals are first-class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Iterable


@dataclass
class Document:
    doc_id: str                       # globally unique: "{source}::{source_id}"
    source: str                       # "slack" | "notion" | "drive"
    source_id: str                    # native id within the source
    title: str
    text: str
    timestamp: datetime               # last edited / posted
    author: str
    acl_principals: list[str] = field(default_factory=list)  # users + groups
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class Connector(ABC):
    """Pull from a source and yield normalized Documents."""

    source_name: str

    @abstractmethod
    def fetch(self) -> Iterable[Document]:
        ...
