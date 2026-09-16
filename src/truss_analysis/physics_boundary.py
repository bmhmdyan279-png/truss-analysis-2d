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
    "ENVIRONMENT_DEPENDENT_VERIFICATION",
    "STATUS_VOCABULARY",
    "VERIFICATION_VOCABULARY",
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

#: Closed vocabulary for :attr:`BoundaryEntry.verification` -- the evidence class
#: behind a row's status.  A status says what the solver *claims*; this says what
#: would have to be true for the claim to have been checked.
VERIFICATION_VOCABULARY: tuple[str, ...] = (
    "analytical-oracle",
    "property-invariant",
    "reference-solver",
    "declared-only",
)

#: Evidence classes that depend on an optional extra and therefore **may not have
#: run** in the environment that produced a result.  This is the set a consumer
#: has to be told about: without it a result can carry a valid ``content_hash``
#: for a boundary whose verification was silently skipped, and "verified" is
#: indistinguishable from "verification not attempted here".
ENVIRONMENT_DEPENDENT_VERIFICATION: frozenset[str] = frozenset({"reference-solver"})


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
    verification : str
        Evidence class behind :attr:`status`, from
        :data:`VERIFICATION_VOCABULARY`.  ``declared-only`` for rows that
        describe an absence, where the declaration is the content and there is
        nothing to check.
    verification_supplement : str or None
        A second evidence class the row also rests on, or ``None``.  Used where
        an always-running oracle is backed by an optional reference-solver
        cross-check, so the primary evidence stays environment-independent while
        the supplement is reported as skippable.

    Notes
    -----
    Whether a row's evidence could have run is answered by
    :meth:`PhysicsBoundary.verification_summary`, not by reading these fields:
    the point of separating them is that a consumer should not have to know which
    classes depend on an extra.
    """

    id: str
    phenomenon: str
    status: str
    detail: str
    doc_section: str
    limits: tuple[str, ...] = ()
    verification: str = "declared-only"
    verification_supplement: str | None = None

    @property
    def environment_dependent(self) -> tuple[str, ...]:
        """Evidence classes on this row that may not have run here."""
        out = [
            v
            for v in (self.verification, self.verification_supplement)
            if v in ENVIRONMENT_DEPENDENT_VERIFICATION
        ]
        return tuple(out)

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
            if entry.verification not in VERIFICATION_VOCABULARY:
                msg = (
                    f"physics boundary entry {entry.id!r} has verification "
                    f"{entry.verification!r}, not in {VERIFICATION_VOCABULARY}"
                )
                raise ValueError(msg)
            if (
                entry.verification_supplement is not None
                and entry.verification_supplement not in VERIFICATION_VOCABULARY
            ):
                msg = (
                    f"physics boundary entry {entry.id!r} has "
                    f"verification_supplement {entry.verification_supplement!r}, "
                    f"not in {VERIFICATION_VOCABULARY}"
                )
                raise ValueError(msg)
            if (
                entry.status
                in (
                    "exact",
                    "supported",
                    "supported-with-limits",
                )
                and entry.verification == "declared-only"
            ):
                msg = (
                    f"physics boundary entry {entry.id!r} claims "
                    f"{entry.status!r} but declares its evidence "
                    "'declared-only'; a modelled capability without a stated "
                    "verification is a claim nobody can check"
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

    def verification_summary(
        self, reference_solver_available: bool | None = None
    ) -> dict[str, Any]:
        """Report what evidence stands behind the boundary, and what may not have run.

        A ``content_hash`` says *which* envelope a result was computed under.  It
        cannot say whether that envelope's verification was executed in the
        environment that produced the result -- and before this method it did not
        have to, because nothing recorded the evidence class at all.  The gap was
        concrete: the OpenSeesPy bridge ships in the optional ``validation``
        extra, its tests skip where that extra is absent, and a result could
        still carry a valid hash for a boundary whose reference-solver column had
        never been run.  "Verified" and "verification not attempted here" looked
        identical to a consumer.

        Parameters
        ----------
        reference_solver_available : bool or None, optional
            Whether the optional reference solver can be imported.  ``None``
            (the default) probes it via
            :func:`truss_analysis.validation.opensees_reference.opensees_available`,
            falling back to ``False`` if that module cannot be imported at all.
            Pass an explicit value to describe an environment other than this one,
            which is what a result payload wants: the fact belongs to the machine
            that ran the analysis, not to whatever reads the digest later.

        Returns
        -------
        dict[str, Any]
            ``counts`` per verification class; ``environment_dependent``, the ids
            resting on evidence that may not have run; ``reference_solver_ran``,
            whether that evidence was available; and ``complete``, which is
            ``True`` only when no row depends on evidence that was unavailable.
            A consumer that requires complete verification checks ``complete``
            rather than reasoning about the classes itself.
        """
        available = reference_solver_available
        if available is None:
            available = _reference_solver_available()

        counts: dict[str, int] = {v: 0 for v in VERIFICATION_VOCABULARY}
        dependent: list[str] = []
        for entry in self.entries:
            counts[entry.verification] = counts.get(entry.verification, 0) + 1
            if entry.environment_dependent and not available:
                dependent.append(entry.id)
        return {
            "counts": counts,
            "reference_solver_available": bool(available),
            # A list, not a tuple: the digest is documented as JSON-safe, and a
            # tuple silently becomes a list on a round trip, so `json.loads(
            # json.dumps(digest)) == digest` would fail for a container that
            # claims to survive exactly that.
            "environment_dependent": list(dependent),
            "complete": not dependent,
        }

    def content_hash(self) -> str:
        """Return a short hash pinning the boundary's content *and* revision.

        Computed over the canonical rendering of every entry together with
        :attr:`audit_round`, so it changes when and only when the envelope
        changes -- which is what lets a result assert *which* boundary it was
        computed under, and a test detect a silent edit.

        The verification evidence class of every entry is hashed too: a boundary
        whose rows changed from "checked against an analytical oracle" to
        "declared only" has changed in a way that matters more than most wording
        edits, and a hash that ignored it would report the two as identical.

        ``audit_round`` is included because it is part of what a consumer needs
        to know.  Without it, re-certifying the same envelope against a new
        revision of EN 1993-1-2 -- bumping ``updated`` and ``audit_round`` while
        the wording stays identical, because the standard's numbers did not move
        -- leaves the hash unchanged, and a downstream consumer pinned to the
        hash cannot tell the two certifications apart.  The library version is
        deliberately *not* included: it changes on every release, which would
        make the hash useless as a statement about the envelope.
        """
        header = f"audit_round={self.audit_round}"
        payload = (
            header
            + "\n"
            + "\n".join(
                f"{e.id}|{e.status}|{e.phenomenon}|{e.detail}|{e.doc_section}|"
                f"{';'.join(e.limits)}|{e.verification}|"
                f"{e.verification_supplement or ''}"
                for e in self.entries
            )
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
            "verification": self.verification_summary(),
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
                    "verification": e.verification,
                    "verification_supplement": e.verification_supplement,
                }
                for e in self.entries
            ],
        }


def _reference_solver_available() -> bool:
    """Whether the optional reference solver can be imported, defaulting to no.

    Resolved lazily and defensively: :mod:`truss_analysis.validation` imports
    ``openseespy`` at module scope, so importing it in an environment without the
    extra raises, and the honest answer there is that the reference-solver
    evidence did not run.  Failing loudly instead would make the digest unusable
    exactly where it is most needed.
    """
    try:
        from .validation.opensees_reference import opensees_available

        return bool(opensees_available())
    except Exception:
        return False


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
                verification=str(row.get("verification", "declared-only")),
                verification_supplement=(
                    str(row["verification_supplement"])
                    if row.get("verification_supplement")
                    else None
                ),
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
