# Ghost Chains — Phase 3

This package preserves the Phase 1 structural and Phase 2 identity models and
adds amount evidence only inside inferred, time-respecting money paths.

## Value-flow model

For each candidate, the scorer walks backward through individual active
transactions ending at its sender. Paths are simple, bounded, causally ordered
by `(createdAt, arrival sequence)`, and capped in number. Each path is a
separate hypothesis, so incoming branches remain separate at convergence.

Outgoing divergence starts a new value segment on every branch. A dramatic
drop also starts a new local regime instead of being treated as a severe
anomaly. This prevents sibling amounts, graph-wide totals, and account-level
aggregation from contaminating a branch's trajectory.

Within a segment, exact-decimal amounts determine increase/decrease and log
ratios provide scale-invariant strength. Slight consistent decay contributes a
small confirmation signal. A direct increase is stronger, and an increase
after a long, coherent decreasing run adds the strongest value evidence.
Multiple credible incoming paths are combined as a recency-, depth-, and
identity-weighted mixture; disagreement reduces confidence.

The final risk is a noisy-or combination of the unchanged Phase 2 result,
value evidence, and bounded same-flow interactions such as return plus
reversal, identity dropout plus amount continuation, and identity divergence
plus reversal. With no usable upstream value context, the result is exactly the
Phase 2 score.

All value evidence is read from the inherited active transaction set, so it
expires at the exact 24-hour boundary and shares the earlier phases' reset,
idempotency, conflict, batch-ordering, and concurrency behavior.
