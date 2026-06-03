"""Slack connector — reads a JSONL file of thread records, yields Documents.

ACL model for Slack:
  - public channel → principal "channel:<name>" (everyone in workspace)
  - private channel → principal "channel:<name>:private" (channel members only)
  - DM → principals are the participating user IDs
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .base import Connector, Document


@dataclass
class SlackConnector(Connector):
    path: str
    source_name: str = "slack"

    def fetch(self) -> Iterable[Document]:
        for line in Path(self.path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            channel = r["channel"]
            principals = self._principals_for(channel, r.get("members", []), r.get("is_private", False))
            yield Document(
                doc_id=f"slack::{r['thread_id']}",
                source="slack",
                source_id=r["thread_id"],
                title=f"#{channel}: {r.get('summary', r['messages'][0]['text'][:60])}",
                text=self._format_thread(r["messages"]),
                timestamp=datetime.fromisoformat(r["last_ts"]),
                author=r["messages"][0]["user"],
                acl_principals=principals,
                extra={"channel": channel, "msg_count": len(r["messages"])},
            )

    @staticmethod
    def _principals_for(channel: str, members: list[str], is_private: bool) -> list[str]:
        if channel.startswith("dm-"):
            return [f"user:{m}" for m in members]
        if is_private:
            return [f"channel:{channel}:private"] + [f"user:{m}" for m in members]
        return [f"channel:{channel}", "group:all-employees"]

    @staticmethod
    def _format_thread(messages: list[dict]) -> str:
        return "\n".join(f"{m['user']}: {m['text']}" for m in messages)
