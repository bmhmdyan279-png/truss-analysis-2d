"""Machine-readable physics boundary — what this solver is and is not.

``docs/theory.md`` §12 states the validity envelope in prose, which answers the
question for a *reader*.  This module answers it for a *program*: an analysis
result can carry the boundary it was computed under, a test can assert the
boundary has not silently changed, and a downstream consumer can refuse an
analysis whose envelope does not cover its use.  Prose cannot do any of those
three (round-6 audit C13/C14).

The shipped artefact is ``data/physics_boundary.yaml``.  It is the single source
of truth; §12 of the theory document is the human rendering of the same list,
and ``tests/test_physics_boundary.py`` pins the two against each other so the
documentation cannot drift away from what actually ships in the wheel.

Every entry carries a stable ``id`` and a ``status`` drawn from a closed
vocabulary (``exact``, ``supported``, ``supported-with-limits``,
``not-supported``, ``deferred``).  Entries that are ``supported-with-limits``
list their limits explicitly, because a capability whose restriction lives only
in a docstring will eventually be used outside it.

Typical use::

    from truss_analysis.physics_boundary import physics_boundary

    boundary = physics_boundary()
    if not boundary.covers("fire_exposure_protected"):
        ...  # refuse: this analysis does not model what you need

    digest = boundary.digest()   # compact, JSON-safe, hash-pinned

Scope note: this module *declares* the envelope.  It does not enforce it at
solve time -- the individual modules raise, warn or clamp as documented, and
the boundary is the map of where they do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BOUNDARY_SCHEMA",
    "STATUS_VOCABULARY",
    "BoundaryEntry",
    "PhysicsBoundary",
    "boundary_digest",
    "physics_boundary",
]

_DATA_PATH = Path(__file__).resolve().parent / "data" / "physics_boundary.yaml"

#: Schema identifier the shipped artefact must declare.
BOUNDARY_SCHEMA = "truss-analysis/physics-boundary/v1"

#: Closed vocabulary for :attr:`BoundaryEntry.status`.  Kept in the module as
#: well as in the YAML so a caller can validate without reading the file, and
#: so the two are compared by a test rather than trusted to agree.
STATUS_VOCABULARY: tuple[str, ...] = (
    "exact",
    "supported",
    "supported-with-limits",
    "not-supported",
    "deferred",
)


@dataclass(frozen=True)
class BoundaryEntry:
    """One row of the physics boundary.

    Attributes
    ----------
    id : str
        Stable identifier; safe to reference from code and from results.
    phenomenon : str
        Human-readable name of the capability or omission.
    status : str
        One of :data:`STATUS_VOCABULARY`.
    detail : str
        What is actually modelled, and against what it is verified.
    doc_section : str
        Where in ``docs/theory.md`` the derivation lives.
    limits : tuple[str, ...]
        Explicit validity restrictions.  Non-empty exactly when ``status`` is
        ``supported-with-limits``; a capability with an unstated restriction
        is how a library ends up used outside its envelope.
    """

    id: str
    phenomenon: str
    status: str
    detail: str
    doc_section: str
    limits: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Return whether this phenomenon is modelled at all."""
        return self.status in ("exact", "supported", "supported-with-limits")


@dataclass(frozen=True)
class PhysicsBoundary:
    """The whole declared validity envelope.

    Attributes
    ----------
    schema : str
        Artefact schema; must equal :data:`BOUNDARY_SCHEMA`.
    version : str
        Library version the boundary was last revised for.
    updated : str
        ISO date of that revision.
    audit_round : int
        External audit round that last changed the envelope.
    entries : tuple[BoundaryEntry, ...]
        All rows, in document order.
    status_vocabulary : dict[str, str]
        The vocabulary's own definitions, as shipped.
    """

    schema: str
    version: str
    updated: str
    audit_round: int
    entries: tuple[BoundaryEntry, ...]
    status_vocabulary: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the artefact rather than trusting it.

        A boundary file that is internally inconsistent is worse than none,
        because a consumer that checks ``covers()`` would be relying on it.
        """
        if self.schema != BOUNDARY_SCHEMA:
            msg = (
                f"physics boundary schema mismatch: file declares "
                f"{self.schema!r}, loader expects {BOUNDARY_SCHEMA!r}"
            )
            raise ValueError(msg)
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"physics boundary has duplicate entry ids: {duplicates}"
            raise ValueError(msg)
        for entry in self.entries:
            if entry.status not in STATUS_VOCABULARY:
                msg = (
                    f"physics boundary entry {entry.id!r} has status "
                    f"{entry.status!r}, not in {STATUS_VOCABULARY}"
                )
                raise ValueError(msg)
            if entry.status == "supported-with-limits" and not entry.limits:
                msg = (
                    f"physics boundary entry {entry.id!r} is "
                    "'supported-with-limits' but declares no limits"
                )
                raise ValueError(msg)

    def entry(self, entry_id: str) -> BoundaryEntry:
        """Return one entry by id.

        Raises
        ------
        KeyError
            If the id is unknown.  The message lists the known ids, because a
            misspelled capability id that silently returned ``False`` from
            :meth:`covers` would be the dangerous failure mode.
        """
        for e in self.entries:
            if e.id == entry_id:
                return e
        known = ", ".join(sorted(e.id for e in self.entries))
        msg = f"unknown physics-boundary id {entry_id!r}; known: {known}"
        raise KeyError(msg)

    def covers(self, entry_id: str) -> bool:
        """Return whether ``entry_id`` is modelled (any supported status)."""
        return self.entry(entry_id).available

    def ids_with_status(self, status: str) -> tuple[str, ...]:
        """Return every entry id carrying ``status``, in document order."""
        return tuple(e.id for e in self.entries if e.status == status)

    def counts(self) -> dict[str, int]:
        """Return the number of entries per status."""
        out = {s: 0 for s in STATUS_VOCABULARY}
        for e in self.entries:
            out[e.status] = out.get(e.status, 0) + 1
        return out

    def content_hash(self) -> str:
        """Return a short hash pinning the boundary's content.

        Computed over the canonical rendering of every entry, so it changes
        when and only when the envelope changes -- which is what lets a result
        assert *which* boundary it was computed under, and a test detect a
        silent edit.
        """
        payload = "\n".join(
            f"{e.id}|{e.status}|{e.phenomenon}|{e.detail}|{e.doc_section}|"
            f"{';'.join(e.limits)}"
            for e in self.entries
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def digest(self) -> dict[str, Any]:
        """Return a compact, JSON-safe summary suitable for a result payload.

        The full table is too large to embed in every analysis output, and
        embedding it would invite consumers to parse prose.  The digest carries
        exactly what is needed to check compatibility later: the schema, the
        version, a content hash and the counts per status.
        """
        return {
            "schema": self.schema,
            "version": self.version,
            "updated": self.updated,
            "audit_round": self.audit_round,
            "content_hash": self.content_hash(),
            "n_entries": len(self.entries),
            "counts": self.counts(),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the whole boundary as plain JSON-safe data."""
        return {
            "schema": self.schema,
            "version": self.version,
            "updated": self.updated,
            "audit_round": self.audit_round,
            "status_vocabulary": dict(self.status_vocabulary),
            "entries": [
                {
                    "id": e.id,
                    "phenomenon": e.phenomenon,
                    "status": e.status,
                    "detail": e.detail,
                    "doc_section": e.doc_section,
                    "limits": list(e.limits),
                }
                for e in self.entries
            ],
        }


@lru_cache(maxsize=1)
def physics_boundary() -> PhysicsBoundary:
    """Load and validate the shipped physics boundary (cached).

    Returns
    -------
    PhysicsBoundary
        The declared validity envelope.

    Raises
    ------
    FileNotFoundError
        If the YAML artefact is missing from the installation -- which would
        mean the package data was not shipped, and is a packaging bug rather
        than something to paper over with a default.
    ValueError
        If the artefact fails validation; see
        :meth:`PhysicsBoundary.__post_init__`.
    """
    if not _DATA_PATH.exists():
        msg = (
            f"physics boundary artefact not found at {_DATA_PATH}; the package "
            "data was not shipped (check [tool.setuptools.package-data])"
        )
        raise FileNotFoundError(msg)
    raw: dict[str, Any] = yaml.safe_load(_DATA_PATH.read_text(encoding="utf-8"))

    entries: list[BoundaryEntry] = []
    for row in raw.get("entries", []):
        limits = row.get("limits") or ()
        entries.append(
            BoundaryEntry(
                id=str(row["id"]),
                phenomenon=str(row["phenomenon"]),
                status=str(row["status"]),
                detail=str(row.get("detail", "")).strip(),
                doc_section=str(row.get("doc_section", "")),
                limits=tuple(str(x) for x in limits),
            )
        )
    return PhysicsBoundary(
        schema=str(raw.get("schema", "")),
        version=str(raw.get("version", "")),
        updated=str(raw.get("updated", "")),
        audit_round=int(raw.get("audit_round", 0)),
        entries=tuple(entries),
        status_vocabulary=dict(raw.get("status_vocabulary") or {}),
    )


def boundary_digest() -> dict[str, Any]:
    """Return :meth:`PhysicsBoundary.digest` of the shipped boundary."""
    return physics_boundary().digest()
