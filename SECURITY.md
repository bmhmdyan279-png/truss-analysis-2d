# Security Policy

> **فارسی:** آسیب‌پذیری‌های امنیتی را **خصوصی** گزارش دهید (تب **Security** →
> **Report a vulnerability**)؛ برای آن‌ها Issue عمومی باز نکنید.

## Reporting a Vulnerability

Use GitHub's **private vulnerability reporting** for this repository:
**Security tab → "Report a vulnerability"** (draft advisory). If that button is
not available, contact the maintainer (`bmhmdyan279-png`) privately through
GitHub rather than opening a public issue.

Please include:

* the affected version(s) (`truss-analysis version` prints the installed one);
* a minimal reproduction (input JSON / snippet) where possible;
* the impact you believe the issue has.

Expected handling: acknowledgment within about 7 days, and a target of 90 days
from report to published advisory/fix. Credit is given to reporters unless they
ask to remain anonymous.

## What Counts as a Security Issue Here

This is a computational structural-engineering library; its threat surface is
small but real. Report **privately**:

| Private (security) | Public (normal issue tracker) |
|---|---|
| Code execution or unsafe deserialization when loading untrusted input (model JSON files, unit specs) | Wrong numbers, crashes with tracebacks on valid input |
| An exploitable vulnerability in a pinned dependency reachable through this package | Missing features, documentation gaps |
| Secret/credential leakage caused by this repository's tooling or artifacts | Anything already declared out of scope in the validity envelope (`docs/theory.md` §12, `src/truss_analysis/data/physics_boundary.yaml`) |

## Engineering-Safety Note (not a vulnerability class)

Results from this library are *code-compliant computations*, verified against
EN 1993-1-2 and independent implementations (see `docs/theory.md` §8), but the
library has **not** been validated against measured behaviour of real
structures (§8.3). A disagreement between output and physical reality is a
modelling-limitation question for the issue tracker — and any life-safety
decision must involve a qualified structural engineer, per the project licence
and `docs/theory.md`.

## Supported Versions

Security fixes are applied to the latest minor series only; older series are
end-of-life (the package is pre-1.0-style fast-moving at 2.x).

| Version | Supported |
|---------|-----------|
| 2.10.x  | ✅ |
| < 2.10  | ❌ |

## Maintainer Checklist (keep this policy truthful)

* Private vulnerability reporting enabled: Settings → Security →
  "Private vulnerability reporting" → Enable.
* Dependabot alerts + security updates enabled (`.github/dependabot.yml`
  supplies the version-update PRs).
