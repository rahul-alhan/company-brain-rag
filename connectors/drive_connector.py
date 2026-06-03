"""Drive connector — reads doc records, yields Documents.

ACL model for Drive:
  - "anyone in domain" → "group:all-employees"
  - link-shared to a group → "team:<name>"
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
class DriveConnector(Connector):
    path: str
    source_name: str = "drive"

    def fetch(self) -> Iterable[Document]:
        for line in Path(self.path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            principals = self._principals_for(r["permissions"])
            yield Document(
                doc_id=f"drive::{r['file_id']}",
                source="drive",
                source_id=r["file_id"],
                title=r["name"],
                text=r["content"],
                timestamp=datetime.fromisoformat(r["modified"]),
                author=r["owner"],
                acl_principals=principals,
                extra={"mime": r.get("mime", "text/plain")},
            )

    @staticmethod
    def _principals_for(perms: dict) -> list[str]:
        out: list[str] = []
        if perms.get("domain_visible"):
            out.append("group:all-employees")
        for team in perms.get("teams", []):
            out.append(f"team:{team}")
        for user in perms.get("users", []):
            out.append(f"user:{user}")
        return out
