"""On-disk index of saved configs, models and results.

Responsibility
--------------
Maintain one JSON index so a caller (and later the Streamlit UI) can list what
exists without walking the filesystem on every interaction.

Explicitly NOT this module's job
--------------------------------
Any science, and any decision about what is worth keeping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class RegistryEntry:
    """One recorded artefact."""

    kind: str          # "config" | "model" | "result"
    identifier: str    # the relevant hash
    name: str          # human-readable label
    path: str          # relative to the registry root
    details: dict[str, Any]


class Registry:
    """A flat JSON index living at `root / index.json`.

    Deliberately dumb: no caching, no locking. Runs are sequential in Phase 1, and
    a single-writer JSON file is easier to inspect by hand than a database when
    something looks wrong.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / INDEX_FILENAME

    def _read(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, entries: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(entries, indent=2, default=str) + "\n",
                                   encoding="utf-8")

    def add(self, entry: RegistryEntry) -> None:
        """Insert or replace the entry with this (kind, identifier)."""
        entries = [e for e in self._read()
                   if not (e["kind"] == entry.kind
                           and e["identifier"] == entry.identifier)]
        entries.append(vars(entry))
        self._write(entries)

    def list(self, kind: str | None = None) -> list[RegistryEntry]:
        """All entries, optionally filtered by kind."""
        return [RegistryEntry(**e) for e in self._read()
                if kind is None or e["kind"] == kind]

    def get(self, kind: str, identifier: str) -> RegistryEntry | None:
        for entry in self.list(kind):
            if entry.identifier == identifier:
                return entry
        return None

    def remove(self, kind: str, identifier: str) -> bool:
        """Drop an entry from the index. Does NOT delete the file it points at."""
        entries = self._read()
        remaining = [e for e in entries
                     if not (e["kind"] == kind and e["identifier"] == identifier)]
        self._write(remaining)
        return len(remaining) != len(entries)
