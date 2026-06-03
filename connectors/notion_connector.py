"""Notion connector — reads page records, yields Documents.

ACL model for Notion:
  - workspace-wide pages → "workspace:default"
  - team-page → "team:<name>"
  - explicit shares → "user:<email>"
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .base import Connector, Document


@dataclass
class NotionConnector(Connector):
    path: str
    source_name: str = "notion"

    def fetch(self) -> Iterable[Document]:
        for line in Path(self.path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            principals = self._principals_for(r["shares"])
            yield Document(
                doc_id=f"notion::{r['page_id']}",
                source="notion",
                source_id=r["page_id"],
                title=r["title"],
                text=r["content"],
                timestamp=datetime.fromisoformat(r["last_edited"]),
                author=r["last_edited_by"],
                acl_principals=principals,
                extra={"workspace": r.get("workspace", "default"), "parent": r.get("parent")},
            )

    @staticmethod
    def _principals_for(shares: dict) -> list[str]:
        out: list[str] = []
        if shares.get("workspace_visible"):
            out.append("workspace:default")
            out.append("group:all-employees")
        for team in shares.get("teams", []):
            out.append(f"team:{team}")
        for user in shares.get("users", []):
            out.append(f"user:{user}")
        return out
