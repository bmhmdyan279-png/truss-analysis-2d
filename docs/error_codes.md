# Error Codes Reference

## Input & Validation Errors (TRUSS-1xxx)
| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| TRUSS-1001 | "Node ID must be string" | Non-string node ID in JSON | Ensure IDs are strings or numbers |
| TRUSS-1002 | "E must be positive" | Negative/zero Young's modulus | Check element properties |
| TRUSS-1003 | "Insufficient constraints" | Less than 3 support constraints | Add supports to achieve stability |

## Computational Errors (TRUSS-2xxx)
| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| TRUSS-2001 | "Singular stiffness matrix" | Mechanism/instability detected | Check connectivity and supports |
| TRUSS-2002 | "Energy validation failed" | Thermodynamic inconsistency | Report as bug with input file |

## File I/O Errors (TRUSS-3xxx)
| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| TRUSS-3001 | "File not found" | Invalid path | Check file path |
| TRUSS-3002 | "Invalid JSON" | Malformed JSON | Validate JSON syntax |
