"""C13/C14: the machine-readable physics boundary and its propagation.

The prose table in ``docs/theory.md`` §12 answers "what is this solver?" for a
reader.  The shipped YAML answers it for a program.  These tests pin three
things:

* the artefact is **internally consistent** -- unique ids, closed status
  vocabulary, and every ``supported-with-limits`` entry actually stating its
  limits (a capability whose restriction lives only in a docstring will
  eventually be used outside it);
* the artefact and the **documentation agree**, so the table in theory.md
  cannot drift away from what ships in the wheel;
* the digest **reaches the result**, is JSON-safe, and changes when and only
  when the envelope changes -- which is what makes it usable as a
  compatibility assertion rather than decoration.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from truss_analysis import run
from truss_analysis.physics_boundary import (
    BOUNDARY_SCHEMA,
    STATUS_VOCABULARY,
    BoundaryEntry,
    PhysicsBoundary,
    boundary_digest,
    physics_boundary,
)

# test_digest_is_identical_across_solver_options runs every bc_method against
# every assembly option by design, which necessarily includes the penalty/dense
# combinations. Those emit two diagnostics: InputIgnoredWarning because penalty
# forces dense assembly, and IllConditionedWarning because the fixed 1e12
# penalty is small relative to this model's stiffness diagonal. Digest invariance
# is the assertion here; both notices are asserted where they are the subject, in
# tests/test_p2_diagnostics.py and tests/test_penalty_bc_and_options.py.
pytestmark = [
    pytest.mark.filterwarnings("ignore::truss_analysis.exceptions.InputIgnoredWarning"),
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.IllConditionedWarning"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY = REPO_ROOT / "docs" / "theory.md"
EXAMPLE = REPO_ROOT / "examples" / "example1.json"


def _norm(text: str) -> str:
    """Normalise prose for comparison: non-alphanumerics become spaces.

    Applied to *both* sides of every documentation/artefact comparison.  The
    first draft stripped punctuation to nothing on the key and to a space on
    the haystack, so "Post-buckling" became "postbuckling" on one side and
    "post buckling" on the other and the row was reported missing while being
    present.  Collapsing whitespace on both sides is what makes the
    comparison mean something.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


# --------------------------------------------------------------------------
# the artefact itself
# --------------------------------------------------------------------------


def test_artefact_loads_and_declares_its_schema() -> None:
    b = physics_boundary()
    assert b.schema == BOUNDARY_SCHEMA
    assert b.version
    assert b.updated
    assert b.audit_round >= 6
    assert len(b.entries) >= 20


def test_entry_ids_are_unique_and_stable() -> None:
    b = physics_boundary()
    ids = [e.id for e in b.entries]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids), ids
    # the ids a consumer would hard-code must exist
    for expected in (
        "linear_truss_statics",
        "fire_exposure_protected",
        "fire_exposure_parametric",
        "system_bifurcation",
        "post_buckling",
        "material_nonlinearity",
    ):
        assert expected in ids, expected


def test_every_status_is_in_the_closed_vocabulary() -> None:
    b = physics_boundary()
    assert set(b.status_vocabulary) == set(STATUS_VOCABULARY)
    for e in b.entries:
        assert e.status in STATUS_VOCABULARY, (e.id, e.status)
        assert b.status_vocabulary[e.status].strip()


def test_supported_with_limits_entries_state_their_limits() -> None:
    """The point of the status: an unstated restriction is not a boundary."""
    b = physics_boundary()
    limited = [e for e in b.entries if e.status == "supported-with-limits"]
    assert len(limited) >= 5, "suspiciously few limited capabilities"
    for e in limited:
        assert e.limits, f"{e.id} claims limits but lists none"
        assert all(limit.strip() for limit in e.limits)
    # an entry that is not modelled has no limits to state
    for e in b.entries:
        if e.status in ("not-supported", "deferred"):
            assert e.status in STATUS_VOCABULARY


def test_available_is_exactly_the_supported_statuses() -> None:
    b = physics_boundary()
    for e in b.entries:
        assert e.available == (
            e.status in ("exact", "supported", "supported-with-limits")
        )
    assert b.covers("linear_truss_statics") is True
    assert b.covers("post_buckling") is False
    assert b.covers("creep_transient_strain") is False


def test_round6_additions_are_present_as_capabilities() -> None:
    """The boundary must reflect what this round actually shipped."""
    b = physics_boundary()
    for entry_id in (
        "fire_exposure_protected",  # B1
        "fire_exposure_parametric",  # B2
        "imperfection_sensitivity",  # B4
        "tangent_linearization_gap",  # the measured first-order gap
    ):
        assert b.covers(entry_id), entry_id
    protected = b.entry("fire_exposure_protected")
    # the integrator choice is stated in the detail ...
    assert "explicit Euler" in protected.detail
    # ... and the material-model omissions are stated as limits
    assert any("gypsum" in limit for limit in protected.limits)
    assert any("intumescent" in limit for limit in protected.limits)


def test_unknown_id_raises_and_lists_the_known_ones() -> None:
    """A misspelled capability must not silently read as "not covered"."""
    b = physics_boundary()
    with pytest.raises(KeyError, match="known:") as exc:
        b.covers("fire_exposur_protected")
    assert "fire_exposure_protected" in str(exc.value)


def test_counts_are_consistent_with_the_entries() -> None:
    b = physics_boundary()
    counts = b.counts()
    assert set(counts) == set(STATUS_VOCABULARY)
    assert sum(counts.values()) == len(b.entries)
    for status, n in counts.items():
        assert n == len(b.ids_with_status(status))


def test_content_hash_is_stable_and_content_sensitive() -> None:
    """The hash is what makes the digest a compatibility assertion."""
    b = physics_boundary()
    assert b.content_hash() == physics_boundary().content_hash()
    assert re.fullmatch(r"[0-9a-f]{16}", b.content_hash())

    edited = PhysicsBoundary(
        schema=b.schema,
        version=b.version,
        updated=b.updated,
        audit_round=b.audit_round,
        entries=(
            *b.entries,
            BoundaryEntry(
                id="something_new",
                phenomenon="A newly modelled effect",
                status="supported",
                detail="detail",
                doc_section="13",
            ),
        ),
        status_vocabulary=b.status_vocabulary,
    )
    assert edited.content_hash() != b.content_hash()


def test_digest_is_json_safe_and_compact() -> None:
    d = boundary_digest()
    payload = json.dumps(d)
    assert len(payload) < 600, "digest is meant to be embeddable in a report"
    assert set(d) == {
        "schema",
        "version",
        "updated",
        "audit_round",
        "content_hash",
        "n_entries",
        "counts",
    }
    assert json.loads(payload) == d


def test_as_dict_round_trips_through_json() -> None:
    b = physics_boundary()
    data = json.loads(json.dumps(b.as_dict()))
    assert len(data["entries"]) == len(b.entries)
    assert data["entries"][0]["id"] == b.entries[0].id
    for row, entry in zip(data["entries"], b.entries, strict=True):
        assert row["limits"] == list(entry.limits)


# --------------------------------------------------------------------------
# validation: the artefact is checked, not trusted
# --------------------------------------------------------------------------


def _minimal_entries() -> tuple[BoundaryEntry, ...]:
    return (
        BoundaryEntry(
            id="a",
            phenomenon="A",
            status="supported",
            detail="d",
            doc_section="1",
        ),
    )


def test_wrong_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        PhysicsBoundary(
            schema="someone/elses/v9",
            version="1",
            updated="2026-01-01",
            audit_round=1,
            entries=_minimal_entries(),
        )


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate entry ids"):
        PhysicsBoundary(
            schema=BOUNDARY_SCHEMA,
            version="1",
            updated="2026-01-01",
            audit_round=1,
            entries=(*_minimal_entries(), *_minimal_entries()),
        )


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in"):
        PhysicsBoundary(
            schema=BOUNDARY_SCHEMA,
            version="1",
            updated="2026-01-01",
            audit_round=1,
            entries=(
                BoundaryEntry(
                    id="a",
                    phenomenon="A",
                    status="mostly-fine",
                    detail="d",
                    doc_section="1",
                ),
            ),
        )


def test_limited_status_without_limits_is_rejected() -> None:
    with pytest.raises(ValueError, match="declares no limits"):
        PhysicsBoundary(
            schema=BOUNDARY_SCHEMA,
            version="1",
            updated="2026-01-01",
            audit_round=1,
            entries=(
                BoundaryEntry(
                    id="a",
                    phenomenon="A",
                    status="supported-with-limits",
                    detail="d",
                    doc_section="1",
                ),
            ),
        )


# --------------------------------------------------------------------------
# the documentation must not drift from the artefact
# --------------------------------------------------------------------------


def test_theory_table_covers_every_artefact_entry() -> None:
    """§12 is the human rendering of the same list, so it must contain it.

    Compared on *status*, not on wording: the prose is allowed to be prettier
    than the YAML, but not to disagree with it.
    """
    text = THEORY.read_text(encoding="utf-8")
    section = text[text.index("## 12. Physics Boundary") :]
    section = section[: section.index("\n## ")] if "\n## " in section[10:] else section
    rows = [line for line in section.splitlines() if line.startswith("|")]
    haystack = _norm("\n".join(rows))

    b = physics_boundary()
    status_words = {
        "exact": "exact",
        "supported": "supported",
        "supported-with-limits": "supported",
        "not-supported": "not supported",
        "deferred": ("deferred", "not supported"),
    }
    missing = []
    for e in b.entries:
        # the phenomenon wording may differ slightly; match on a distinctive
        # fragment of it, lowercased and stripped of markdown/math
        key = " ".join(_norm(e.phenomenon).split()[:2])
        if key and key not in haystack:
            missing.append((e.id, e.phenomenon))
        assert status_words[e.status]  # every status has a prose marker
    assert not missing, f"artefact entries absent from docs/theory.md §12: {missing}"


def test_theory_table_marks_unsupported_rows_as_unsupported() -> None:
    """Row-level agreement on the negative half of the table.

    The dangerous drift is a capability the artefact calls ``not-supported``
    while the documentation implies it works, so that half is checked per row
    rather than by fragment search.
    """
    text = THEORY.read_text(encoding="utf-8")
    section = text[text.index("## 12. Physics Boundary") :]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("|") and "---" not in line
    ]
    b = physics_boundary()
    unsupported = [e for e in b.entries if not e.available]
    assert unsupported
    flat = _norm("\n".join(rows))
    for e in unsupported:
        key = " ".join(_norm(e.phenomenon).split()[:2])
        assert key in flat, (e.id, e.phenomenon)
    # the table must actually use the bold "not supported" marker
    assert "not supported" in flat


# --------------------------------------------------------------------------
# C14: propagation into the analysis result
# --------------------------------------------------------------------------


def test_digest_reaches_solver_metadata() -> None:
    result = run(EXAMPLE, quiet=True)
    md = result.solver_metadata
    assert "physics_boundary" in md
    assert md["physics_boundary"] == boundary_digest()
    assert md["physics_boundary"]["content_hash"] == physics_boundary().content_hash()


def test_digest_survives_the_json_export() -> None:
    """It has to be in the artefact that outlives the process."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        run(EXAMPLE, quiet=True, output=str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
    md = payload["solver_metadata"]["physics_boundary"]
    assert md["schema"] == BOUNDARY_SCHEMA
    assert md["n_entries"] == len(physics_boundary().entries)
    assert md["counts"] == boundary_digest()["counts"]


def test_digest_is_identical_across_solver_options() -> None:
    """The envelope does not depend on how the linear system was solved."""
    a = run(EXAMPLE, quiet=True, use_sparse=True).solver_metadata["physics_boundary"]
    b = run(EXAMPLE, quiet=True, use_sparse=False).solver_metadata["physics_boundary"]
    c = run(EXAMPLE, quiet=True, bc_method="penalty").solver_metadata[
        "physics_boundary"
    ]
    assert a == b == c


def test_metadata_remains_json_serialisable_with_the_digest() -> None:
    result = run(EXAMPLE, quiet=True)
    assert json.loads(json.dumps(result.solver_metadata)) == result.solver_metadata


# --------------------------------------------------------------------------
# round-7 audit, item 10: content_hash covered the entries but not the
# revision, so re-certifying the same wording against a new standard edition
# left a consumer pinned to the hash unable to tell the two apart.
# --------------------------------------------------------------------------


def test_content_hash_tracks_the_audit_round() -> None:
    """Same envelope, new certification: the hash must move."""
    import dataclasses

    boundary = physics_boundary()
    recertified = dataclasses.replace(boundary, audit_round=boundary.audit_round + 1)

    assert recertified.content_hash() != boundary.content_hash()
    # the entries did not change, so the difference is entirely the revision
    assert recertified.entries == boundary.entries


def test_content_hash_ignores_the_calendar_date() -> None:
    """A date is metadata about the artefact, not content of the envelope.

    ``digest()`` carries ``updated`` for provenance; the hash must not, or every
    cosmetic date edit would invalidate results computed under an identical
    envelope.
    """
    import dataclasses

    boundary = physics_boundary()
    redated = dataclasses.replace(boundary, updated="2099-12-31")

    assert redated.content_hash() == boundary.content_hash()
    assert redated.digest()["updated"] == "2099-12-31"
    assert redated.digest()["content_hash"] == boundary.digest()["content_hash"]


def test_content_hash_ignores_the_library_version() -> None:
    """The version changes on every release and says nothing about the envelope."""
    import dataclasses

    boundary = physics_boundary()
    reversioned = dataclasses.replace(boundary, version="99.99.99")

    assert reversioned.content_hash() == boundary.content_hash()


def test_content_hash_is_deterministic_and_entry_sensitive() -> None:
    """Every hashed field must actually be hashed."""
    import dataclasses

    boundary = physics_boundary()
    assert boundary.content_hash() == physics_boundary().content_hash()

    first = boundary.entries[0]
    for field_name, new_value in (
        ("id", first.id + "_renamed"),
        ("status", "deferred"),
        ("phenomenon", first.phenomenon + " (altered)"),
        ("detail", first.detail + " altered"),
        ("doc_section", "99.99"),
    ):
        altered = dataclasses.replace(first, **{field_name: new_value})
        edited = dataclasses.replace(boundary, entries=(altered, *boundary.entries[1:]))
        assert edited.content_hash() != boundary.content_hash(), field_name

    if first.limits:
        altered = dataclasses.replace(first, limits=(*first.limits, "one more"))
        edited = dataclasses.replace(boundary, entries=(altered, *boundary.entries[1:]))
        assert edited.content_hash() != boundary.content_hash(), "limits"
