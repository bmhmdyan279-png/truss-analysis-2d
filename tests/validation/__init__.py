"""Validation-suite package: the six levels + the global negative controls.

Self-contained by design: every expected number in these tests is either an
independently hand-derived closed form (level 1), a transcription of the
published reduction table (level 2), or an acceptance threshold fixed by the
validation protocol BEFORE measurement.  The tests only assert; bulk evidence
tables (full grids, curves, hashes) are produced by dedicated runner scripts
outside the package.
"""
