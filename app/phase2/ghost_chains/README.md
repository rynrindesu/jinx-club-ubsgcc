# Ghost Chains — Phase 2

This package extends the Phase 1 streaming graph engine with the two optional
identity dimensions introduced in Phase 2: `ipAddress` and `deviceId`.

## Cumulative model

The Phase 1 discounted-walk score remains unchanged.  Before each active
transaction is inserted, Phase 2 independently evaluates IP and device
evidence in three structural contexts:

- agreement or divergence along causal money paths containing the transaction;
- disappearance of an identity attribute after it appeared upstream; and
- reuse of the same identity across otherwise disconnected weak components.

IP evidence is weighted a little more cautiously than device evidence because
shared network infrastructure is common.  Reuse across disconnected components
is deliberately low on its own and saturates as components accumulate.  Path
evidence is distance-discounted, while parallel payments at the same path
distance are averaged so transaction frequency alone cannot manufacture a
signal. Identity paths follow strict `(createdAt, arrival sequence)` ordering
and the structural model's simple-route rules. This also lets a late-arriving
transaction connect to a later event that is already active without allowing
reversed event time to manufacture an identity trail.

The independent IP and device raw signals are added to the Phase 1 structural
raw score before the existing bounded risk transform.  A transaction with no
usable identity evidence therefore receives exactly its Phase 1 score.

## State behavior

Identity scoring reads the same active transaction set as graph scoring, so it
inherits the `[watermark - 24h, watermark]` window and cannot retain
expired identity evidence.  Phase 1 ordering, idempotency, conflict detection,
atomic batches, reset behavior, forward compatibility, and concurrency
semantics are preserved by the cumulative engine.
