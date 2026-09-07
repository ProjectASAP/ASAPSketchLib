# Experimental UnivMon-Q Evaluation

`UnivMonQ` is experimental. The measurements below are empirical results for
the current Rust implementations, not new theoretical guarantees. APIs,
estimators, configuration defaults, wire state, and performance may change.

## Reproduction

```bash
cargo run --release --example evaluate_univmon_q_skew -- 1000000 5 100000 256
cargo run --release --example evaluate_univmon_q_skew -- 1000000 5 100000 256 all equal-memory
```

The comparison uses one million observations per dataset, five independent
trials, a 100,000-value domain, Zipf exponents from 0.0 through 2.0, and an
eight-way merge. Both constructions use 14 logical levels, a top width of
4,096, depth 5, and 256 candidates. UnivMon uses terminal-only `fast_insert`.
UnivMon-Q additionally uses 32-bit counters, width halving every three levels,
and 4,096 coordinated occurrence samples.

## Comparison with UnivMon

Worst p95 error across the seven skew levels:

| Metric | UnivMon | UnivMon-Q |
| --- | ---: | ---: |
| Point-frequency normalized RMSE | 0.378% | 0.383% |
| F0 relative error | 9.52% | 12.04% |
| F2 relative error | 4.96% | 10.12% |
| Entropy relative error | 17.32% | 3.05% |
| L1 relative error | 0% (exact count) | 0% (exact count) |
| Quantile mean rank error | Not supported | 0.45% |
| Maximum CDF error | Not supported | 2.42% |

Median performance ranges across the seven skew levels:

| Measurement | UnivMon | UnivMon-Q | UnivMon-Q improvement |
| --- | ---: | ---: | ---: |
| Update | 1.12–12.03 us/item | 52–140 ns/item | 8–204x |
| Eight-way merge | 36–223 ms | 2.2–4.3 ms | 16–52x |
| All-metric query batch | 1.00–3.13 ms | 0.24–1.16 ms | 2.6–4.4x |
| In-memory core | At least 2.188 MiB of counters, before heap metadata | 0.707 MiB estimated | At least 3.1x smaller |

UnivMon-Q's point-frequency accuracy was similar while its compact F0/F2 core
was less accurate. L1 is exact in both sketches for this insertion-only
stream. Entropy uses an adaptive assisted estimator: diffuse streams and
complete candidate sets use
the original universal recurrence, while concentrated streams with incomplete
recovery use the coordinated occurrence sample and CountSketch frequency
estimates. This removes the large candidate-threshold failure observed with
the recurrence-only entropy estimator.

### Conservative equal-memory profile

The compact table above is not an equal-memory comparison. To separate the
construction from that resource difference, the `equal-memory` profile keeps
UnivMon at 4,096 columns and 256 candidates, then configures UnivMon-Q with
full-width levels and the largest candidate table that fits under the memory
occupied by UnivMon's counters alone. This gives UnivMon-Q 1,560 candidates
and 2.187 MiB of estimated reserved state versus 2.188 MiB of UnivMon counters,
before counting UnivMon's heap metadata. The comparison is therefore
conservative in UnivMon's favor.

Worst p95 error across the same seven skews and five trials:

| Metric | UnivMon | Equal-memory UnivMon-Q |
| --- | ---: | ---: |
| Point-frequency normalized RMSE | 0.378% | 0.379% |
| F0 relative error | 9.52% | 6.13% |
| F2 relative error | 4.96% | 2.63% |
| Entropy relative error | 17.32% | 3.05% |
| L1 relative error | 0% (exact count) | 0% (exact count) |
| Quantile mean rank error | Not supported | 0.45% |
| Maximum CDF error | Not supported | 2.42% |

Median performance ranges across the seven skews:

| Measurement | UnivMon | Equal-memory UnivMon-Q | UnivMon-Q improvement |
| --- | ---: | ---: | ---: |
| Update | 1.12–12.70 us/item | 51–153 ns/item | 7.4–199x |
| Eight-way merge | 35–222 ms | 2.6–12.1 ms | 13.2–18.3x |
| All-metric query batch | 0.98–3.23 ms | 0.44–2.09 ms | 1.4–2.2x |

For this insertion-only stream, both implementations answer L1 from their
exact stored observation count;
`estimate_g_sum(|f| f)` remains available as a generic-recurrence diagnostic
but is not used for L1. The assisted entropy estimator lowers the worst p95
error from 54.05% for recurrence-only equal-memory UnivMon-Q to 3.05%. The
largest residual error occurs around Zipf alpha 1.6, where recovery is
incomplete and the occurrence path has a 0.082-nat p95 absolute error.
Those 35-trial figures precede the heavy/residual variance reduction below and
are therefore conservative for the current estimator.

### Entropy stress matrix

The dedicated entropy evaluator broadens the distribution family and varies
the occurrence-sample budget:

```bash
cargo run --release --example evaluate_univmon_q_entropy -- 200000 8 50000
```

It covers four uniform supports, nine Zipf exponents from 0.5 through 2.5,
and five heavy-head/uniform-tail mixtures with head mass from 0.5 through
0.9999. Each of the 18 workloads uses eight trials and an eight-way merge.
Four memory/sample profiles produce 576 merged evaluations in total.

| Profile | Estimated memory | Occurrence samples | Worst adaptive entropy p95 |
| --- | ---: | ---: | ---: |
| Compact | 0.592 MiB | 512 | 4.86% |
| Compact default | 0.674 MiB | 4,096 | 1.75% |
| Compact large sample | 0.955 MiB | 16,384 | 1.46% |
| Equal-memory | 1.874 MiB | 4,096 | 1.36% |

A raw occurrence mean initially exposed a 22.76% p95 error for a 99% heavy
head and large diffuse tail. The final assisted estimator computes the
recovered-heavy entropy contribution directly, fixes its residual mass, and
uses occurrences only for conditional residual entropy. This reduced that
case to 0.32% p95. Every profile produced bit-identical entropy across left
fold and balanced merge trees. Extremely concentrated cases whose observed
support fits the candidate table use the complete universal recurrence and
were exact in this matrix.

UnivMon-Q saves memory through packed 32-bit counters and geometric width
reduction, whereas this UnivMon baseline uses fixed-width 64-bit counter
layers.

## Correctness checks

Across 35 large trials, both sketches preserved exact stream counts, produced
finite universal estimates, and round-tripped through their native MessagePack
state. Shared UnivMon metrics and all UnivMon-Q queries were stable across left
fold and balanced merge trees. Two uniform trials selected different
equal-frequency identities for UnivMon's top-20 result; the metric estimates
were unchanged, so these are reported as tie warnings rather than correctness
failures.

For UnivMon-Q, exact extrema, monotone CDF construction, monotone quantiles,
and ordered-query merge equivalence were also checked. These tests support the
implementation but do not replace the guarantees and caveats in the
[UnivMon-Q API documentation](./api/api_univmon_q.md).
