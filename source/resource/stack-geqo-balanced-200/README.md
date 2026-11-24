# Stack GEQO-balanced 200

Ryan Marcus Stack (Bao) workload: https://rmarcus.info/stack.html

100 q3 queries (12 relations), 50 q2 (11), and 50 q8 (9). Seed 42.
SQL bytes and query IDs match the archived Stack experiments.
Files are flattened as `<family>_<original-stem>.sql`; original paths and
SHA-256 hashes are recorded in `selection-manifest.json`.

Corpus fingerprint (normalized-SQL v1):
`19c9520671c9355a0ab41cad1ce402b66897fe83166e583a3ddaa810076a2e87`

The database dump is separate and must be restored on the target PostgreSQL
cluster. Query family and relation count are confounded in this selection.
