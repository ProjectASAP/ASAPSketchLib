# Test Coverage

## Data Input

- 100K uniform distribution i64 range from 0-10M -- (1)
- 1M uniform distribution i64 range from 0-10M -- (2)
- 100K zipf distribution i64, s=1.1, key-size=4096 -- (3)
- 1M zipf distribution i64, s=1.1, key-size=4096 -- (4)
- 100K zipf distribution i64, s=1.1, key-size=20k -- (5)
- 1M zipf distribution i64, s=1.1, key-size=20k -- (6)
- 100K uniform distribution f64 range from 0-10M -- (7)
- 1M uniform distribution f64 range from 0-10M -- (8)
- 100K zipf distribution f64, s=1.1, key-size=4096 -- (9)
- 1M zipf distribution f64, s=1.1, key-size=4096 -- (10)
- 100K zipf distribution f64, s=1.1, key-size=20k -- (11)
- 1M zipf distribution f64, s=1.1, key-size=20k -- (12)
- 100K uniform distribution string, 3 char long, within `alphabet` -- (13)
- 100K zipf distribution string, s=1.1, key-size=4096, 3 char long, within `alphabet` -- (14)
- 100K normal distribution f64, mean=1000, sd=250 -- (15)
- 1M normal distribution f64, mean=1000, sd=250 -- (16)
- 100K exponential distribution f64, lambda=1e-3 -- (17)
- 100K log-uniform f64 landing on DDSketch bucket edges, `gamma^k * (1 + frac*(gamma-1))`, k in [5, 40), gamma from the sketch's alpha -- (18)

### alphabet

`A-Za-z0-9`

## e2e_cardinality

- HyperLogLog
  - all classic, ErtlMLE, HIP
  - Precision: P=10-18
  - Input: (1) ~ (12)
  - error bound:
    - Precision 10-12: relative error 13%
    - Precision 13-14: relative error 4.6%
    - Precisioin 15-18: relative error 2.3%
    - reasoning: theoretical error bound is a standard deviation, to simplify the result, use this arbitrary number
- UnivMon
  - cardinality only
  - Configuration: heap 32, row 5, col 2048, layers 16
  - Input: (1) ~ (12)
  - error bound:
    - cardinality: -30% / +30%
- SetAggregator
  - Input: (1) ~ (12)
  - exact structure: distinct count and membership are both exact, no error bound
- checks applied to every sketch in this section
  - checkpoint accuracy: the same band must hold at 10K, 100K and 1M distinct keys, not only at the end of the stream
  - duplicate-replay invariance: re-inserting keys already seen must not move the estimate
  - shard merge: an even/odd split merged back must stay inside the same band as the single-pass sketch
  - configuration

## e2e_composition

- HashSketchEnsemble
  - cells: CountMin (row 3, col 4096, FastPath) + HyperLogLog (ErtlMLE), one hash layer shared by every cell
  - Input: (1) ~ (12)
  - error bound:
    - CountMin cell: same with sketch instance
    - HyperLogLog cell: same with sketch instance

## e2e_experimental

Feature-gated behind `--features experimental`.

- UniformSampling
  - Configuration: rate 0.1
  - Input: (1) ~ (2), (7) ~ (8)
  - error bound:
    - retained sample count within 15% of `n * rate`
    - `total_seen` is exact, and every retained sample must be a value that appeared in the stream
    - merge of two same-rate sketches unions the samples and sums `total_seen`
- EHUnivOptimized
  - Configuration: k=2, window 100, UnivMon defaults (heap 32, row 5, col 2048, layers 8)
  - map tier: interval queries fully inside the retained range are exact, both for the totals and for the per-key counts
- KMV
  - Input: (1) ~ (12)
  - configuration (k value): 32, 128
  - error bound:
    - k = 32: relative error 73%
    - k = 128: relative error 36%
    - reasoning: theoretical error bound is a probability, to simplify the result, use this arbitrary number
- CountMinHll
  - Configuration: row 4, col 512, HLL precision 10 (clean run); row 4, col 32 (contended run); row 4, col 256 (merge)
  - Input: 120 keys with 40 ~ 4443 distinct values each, spanning HyperLogLog's linear-counting regime and its harmonic regime
  - which rows are collision-free is computed from the sketch's own routing (`hash128_seeded` plus `MatrixFastHash::col_for_row`), not assumed
  - unit tests derive their bands the same way: `z * 1.04 / sqrt(2^p)` at the sketch's own precision, the closed form `m*ln(m/(m-1))` for a single distinct value, and an exact `0.0` for a key with no inserts (an all-zero bucket takes the linear-counting branch to `m*ln(m/m)`)
  - every guard in `deserialize` -- precision range, depth against `2^precision`, power-of-two cols, `cols >= 2`, and the 128-bit column budget -- is tripped by a forged payload
  - error bound:
    - a key with at least one collision-free row: `CardinalityConfidenceSpec::hll(10, z=4)`, i.e. HyperLogLog's own register model and nothing looser
    - reasoning: a value's register index and rank do not depend on the bucket, so a key writes the same register array into every bucket it touches; an unshared bucket therefore holds exactly that array, and the minimum across rows cannot fall below it
    - two collision-free rows of one key hold byte-identical buckets -- exact, no tolerance
    - contended run (200 keys over 32 columns, essentially no clean rows): every estimate stays above HyperLogLog's lower band, since a collision can only raise an estimate
  - both runs assert their own occupancy, so a routing regression that made the rows track each other fails rather than silently skipping the checks
  - merge: register-wise maximum reproduces a single pass over both halves byte for byte

## e2e_frameworks

- Hydra
  - Configuration: schema 2 dims (`region`, `user`), row 4, col 4096
  - counter heads: CountMin (row 4, col 4096, FastPath), KLL (k=200, row 4, col 512), HyperLogLog (row 4, col 512)
  - Input: (13) ~ (14) for the CountMin head, (7) ~ (8) for the KLL head, distinct keys for the HLL head
  - Query: full key, subpopulation query with one key and one `None`, unseen key
  - error bound:
    - full key: absolute error 1000
    - subpopulation query with one key and one `None`: one-sided, `est >= truth`, upper slack 20% + 1
      - reasoning: a generalized key accumulates sibling traffic through the fan-out, so it can only over-count
    - unseen key: absolute error 100
    - KLL head: median inside rank error 3%, CDF absolute error 0.03
    - HLL head: relative error 10%
- UnivMon
  - Configuration: heap 32, row 5, col 2048, layers 16
  - Input: (1) ~ (12), with weighted updates
  - both the standard insert path and the fast-insert path
  - error bound:
    - L1: exact
    - L2: relative error 11%; fast-insert path 18%
    - entropy: -12% / +15%
    - cardinality: -30% / +30%
    - reasoning: every metric above L1 comes out of the recursive g-sum over sampled layers, so the bound is empirical rather than a single-layer bound
- UnivMonPyramid
  - Configuration: defaults
  - Input: (1) ~ (12), with weighted updates
  - error bound:
    - L1: exact
    - L2: relative error 15%
    - cardinality: -30% / +30%

## e2e_frequency

- CountMin sketch
  - both vector based and fixed matrix based
  - both FastPath and RegularPath
  - size, combination of row and column:
    - row: 3, 5, 7
    - column: 2048, 4096, 8192, 16384, 32768
  - per key error bound (theoretical, `f(k)=1` worst case):
    - `eps = e/w`, `err <= eps * ||f||_1 = e*N/w`, `delta = e^-d`
    - relative error column = `err / f(k)` with `f(k)=1`; in range column = `1 - delta`

|input|row|column|per-key relative error|in range key percentage|
|---|---|---|---|---|
|(1)|3|2048|13,273%|95.02%|
|(1)|3|4096|6,636%|95.02%|
|(1)|3|8192|3,318%|95.02%|
|(1)|3|16384|1,659%|95.02%|
|(1)|3|32768|829.6%|95.02%|
|(1)|5|2048|13,273%|99.33%|
|(1)|5|4096|6,636%|99.33%|
|(1)|5|8192|3,318%|99.33%|
|(1)|5|16384|1,659%|99.33%|
|(1)|5|32768|829.6%|99.33%|
|(1)|7|2048|13,273%|99.91%|
|(1)|7|4096|6,636%|99.91%|
|(1)|7|8192|3,318%|99.91%|
|(1)|7|16384|1,659%|99.91%|
|(1)|7|32768|829.6%|99.91%|
|(2)|3|2048|132,729%|95.02%|
|(2)|3|4096|66,364%|95.02%|
|(2)|3|8192|33,182%|95.02%|
|(2)|3|16384|16,591%|95.02%|
|(2)|3|32768|8,296%|95.02%|
|(2)|5|2048|132,729%|99.33%|
|(2)|5|4096|66,364%|99.33%|
|(2)|5|8192|33,182%|99.33%|
|(2)|5|16384|16,591%|99.33%|
|(2)|5|32768|8,296%|99.33%|
|(2)|7|2048|132,729%|99.91%|
|(2)|7|4096|66,364%|99.91%|
|(2)|7|8192|33,182%|99.91%|
|(2)|7|16384|16,591%|99.91%|
|(2)|7|32768|8,296%|99.91%|
|(3)|3|2048|13,273%|95.02%|
|(3)|3|4096|6,636%|95.02%|
|(3)|3|8192|3,318%|95.02%|
|(3)|3|16384|1,659%|95.02%|
|(3)|3|32768|829.6%|95.02%|
|(3)|5|2048|13,273%|99.33%|
|(3)|5|4096|6,636%|99.33%|
|(3)|5|8192|3,318%|99.33%|
|(3)|5|16384|1,659%|99.33%|
|(3)|5|32768|829.6%|99.33%|
|(3)|7|2048|13,273%|99.91%|
|(3)|7|4096|6,636%|99.91%|
|(3)|7|8192|3,318%|99.91%|
|(3)|7|16384|1,659%|99.91%|
|(3)|7|32768|829.6%|99.91%|
|(4)|3|2048|132,729%|95.02%|
|(4)|3|4096|66,364%|95.02%|
|(4)|3|8192|33,182%|95.02%|
|(4)|3|16384|16,591%|95.02%|
|(4)|3|32768|8,296%|95.02%|
|(4)|5|2048|132,729%|99.33%|
|(4)|5|4096|66,364%|99.33%|
|(4)|5|8192|33,182%|99.33%|
|(4)|5|16384|16,591%|99.33%|
|(4)|5|32768|8,296%|99.33%|
|(4)|7|2048|132,729%|99.91%|
|(4)|7|4096|66,364%|99.91%|
|(4)|7|8192|33,182%|99.91%|
|(4)|7|16384|16,591%|99.91%|
|(4)|7|32768|8,296%|99.91%|
|(5)|3|2048|13,273%|95.02%|
|(5)|3|4096|6,636%|95.02%|
|(5)|3|8192|3,318%|95.02%|
|(5)|3|16384|1,659%|95.02%|
|(5)|3|32768|829.6%|95.02%|
|(5)|5|2048|13,273%|99.33%|
|(5)|5|4096|6,636%|99.33%|
|(5)|5|8192|3,318%|99.33%|
|(5)|5|16384|1,659%|99.33%|
|(5)|5|32768|829.6%|99.33%|
|(5)|7|2048|13,273%|99.91%|
|(5)|7|4096|6,636%|99.91%|
|(5)|7|8192|3,318%|99.91%|
|(5)|7|16384|1,659%|99.91%|
|(5)|7|32768|829.6%|99.91%|
|(6)|3|2048|132,729%|95.02%|
|(6)|3|4096|66,364%|95.02%|
|(6)|3|8192|33,182%|95.02%|
|(6)|3|16384|16,591%|95.02%|
|(6)|3|32768|8,296%|95.02%|
|(6)|5|2048|132,729%|99.33%|
|(6)|5|4096|66,364%|99.33%|
|(6)|5|8192|33,182%|99.33%|
|(6)|5|16384|16,591%|99.33%|
|(6)|5|32768|8,296%|99.33%|
|(6)|7|2048|132,729%|99.91%|
|(6)|7|4096|66,364%|99.91%|
|(6)|7|8192|33,182%|99.91%|
|(6)|7|16384|16,591%|99.91%|
|(6)|7|32768|8,296%|99.91%|
|(7)|3|2048|13,273%|95.02%|
|(7)|3|4096|6,636%|95.02%|
|(7)|3|8192|3,318%|95.02%|
|(7)|3|16384|1,659%|95.02%|
|(7)|3|32768|829.6%|95.02%|
|(7)|5|2048|13,273%|99.33%|
|(7)|5|4096|6,636%|99.33%|
|(7)|5|8192|3,318%|99.33%|
|(7)|5|16384|1,659%|99.33%|
|(7)|5|32768|829.6%|99.33%|
|(7)|7|2048|13,273%|99.91%|
|(7)|7|4096|6,636%|99.91%|
|(7)|7|8192|3,318%|99.91%|
|(7)|7|16384|1,659%|99.91%|
|(7)|7|32768|829.6%|99.91%|
|(8)|3|2048|132,729%|95.02%|
|(8)|3|4096|66,364%|95.02%|
|(8)|3|8192|33,182%|95.02%|
|(8)|3|16384|16,591%|95.02%|
|(8)|3|32768|8,296%|95.02%|
|(8)|5|2048|132,729%|99.33%|
|(8)|5|4096|66,364%|99.33%|
|(8)|5|8192|33,182%|99.33%|
|(8)|5|16384|16,591%|99.33%|
|(8)|5|32768|8,296%|99.33%|
|(8)|7|2048|132,729%|99.91%|
|(8)|7|4096|66,364%|99.91%|
|(8)|7|8192|33,182%|99.91%|
|(8)|7|16384|16,591%|99.91%|
|(8)|7|32768|8,296%|99.91%|
|(9)|3|2048|13,273%|95.02%|
|(9)|3|4096|6,636%|95.02%|
|(9)|3|8192|3,318%|95.02%|
|(9)|3|16384|1,659%|95.02%|
|(9)|3|32768|829.6%|95.02%|
|(9)|5|2048|13,273%|99.33%|
|(9)|5|4096|6,636%|99.33%|
|(9)|5|8192|3,318%|99.33%|
|(9)|5|16384|1,659%|99.33%|
|(9)|5|32768|829.6%|99.33%|
|(9)|7|2048|13,273%|99.91%|
|(9)|7|4096|6,636%|99.91%|
|(9)|7|8192|3,318%|99.91%|
|(9)|7|16384|1,659%|99.91%|
|(9)|7|32768|829.6%|99.91%|
|(10)|3|2048|132,729%|95.02%|
|(10)|3|4096|66,364%|95.02%|
|(10)|3|8192|33,182%|95.02%|
|(10)|3|16384|16,591%|95.02%|
|(10)|3|32768|8,296%|95.02%|
|(10)|5|2048|132,729%|99.33%|
|(10)|5|4096|66,364%|99.33%|
|(10)|5|8192|33,182%|99.33%|
|(10)|5|16384|16,591%|99.33%|
|(10)|5|32768|8,296%|99.33%|
|(10)|7|2048|132,729%|99.91%|
|(10)|7|4096|66,364%|99.91%|
|(10)|7|8192|33,182%|99.91%|
|(10)|7|16384|16,591%|99.91%|
|(10)|7|32768|8,296%|99.91%|
|(11)|3|2048|13,273%|95.02%|
|(11)|3|4096|6,636%|95.02%|
|(11)|3|8192|3,318%|95.02%|
|(11)|3|16384|1,659%|95.02%|
|(11)|3|32768|829.6%|95.02%|
|(11)|5|2048|13,273%|99.33%|
|(11)|5|4096|6,636%|99.33%|
|(11)|5|8192|3,318%|99.33%|
|(11)|5|16384|1,659%|99.33%|
|(11)|5|32768|829.6%|99.33%|
|(11)|7|2048|13,273%|99.91%|
|(11)|7|4096|6,636%|99.91%|
|(11)|7|8192|3,318%|99.91%|
|(11)|7|16384|1,659%|99.91%|
|(11)|7|32768|829.6%|99.91%|
|(12)|3|2048|132,729%|95.02%|
|(12)|3|4096|66,364%|95.02%|
|(12)|3|8192|33,182%|95.02%|
|(12)|3|16384|16,591%|95.02%|
|(12)|3|32768|8,296%|95.02%|
|(12)|5|2048|132,729%|99.33%|
|(12)|5|4096|66,364%|99.33%|
|(12)|5|8192|33,182%|99.33%|
|(12)|5|16384|16,591%|99.33%|
|(12)|5|32768|8,296%|99.33%|
|(12)|7|2048|132,729%|99.91%|
|(12)|7|4096|66,364%|99.91%|
|(12)|7|8192|33,182%|99.91%|
|(12)|7|16384|16,591%|99.91%|
|(12)|7|32768|8,296%|99.91%|

- Count sketch
  - both vector based and fixed matrix based
  - both FastPath and RegularPath
  - size, combination of row and column:
    - row: 3, 5, 7
    - column: 2048, 4096, 8192, 16384, 32768
  - per key error bound (theoretical, `f(k)=1` worst case):
    - `eps = sqrt(e/w)`, `err <= eps * ||f||_2`, per-row failure `<= 1/e` (Chebyshev), median of `d` rows
    - relative error column = `err / f(k)` with `f(k)=1`; in range column = median-of-rows success
    - `||f||_2`: (1)(7) 318, (2)(8) 1049, (3)(9) 19591, (4)(10) 195910, (5)(11) 17771, (6)(12) 177712

|input|row|column|per-key relative error|in range key percentage|
|---|---|---|---|---|
|(1)|3|2048|1,158%|69.36%|
|(1)|3|4096|818.7%|69.36%|
|(1)|3|8192|578.9%|69.36%|
|(1)|3|16384|409.4%|69.36%|
|(1)|3|32768|289.5%|69.36%|
|(1)|5|2048|1,158%|73.64%|
|(1)|5|4096|818.7%|73.64%|
|(1)|5|8192|578.9%|73.64%|
|(1)|5|16384|409.4%|73.64%|
|(1)|5|32768|289.5%|73.64%|
|(1)|7|2048|1,158%|76.97%|
|(1)|7|4096|818.7%|76.97%|
|(1)|7|8192|578.9%|76.97%|
|(1)|7|16384|409.4%|76.97%|
|(1)|7|32768|289.5%|76.97%|
|(2)|3|2048|3,821%|69.36%|
|(2)|3|4096|2,702%|69.36%|
|(2)|3|8192|1,911%|69.36%|
|(2)|3|16384|1,351%|69.36%|
|(2)|3|32768|955.3%|69.36%|
|(2)|5|2048|3,821%|73.64%|
|(2)|5|4096|2,702%|73.64%|
|(2)|5|8192|1,911%|73.64%|
|(2)|5|16384|1,351%|73.64%|
|(2)|5|32768|955.3%|73.64%|
|(2)|7|2048|3,821%|76.97%|
|(2)|7|4096|2,702%|76.97%|
|(2)|7|8192|1,911%|76.97%|
|(2)|7|16384|1,351%|76.97%|
|(2)|7|32768|955.3%|76.97%|
|(3)|3|2048|71,374%|69.36%|
|(3)|3|4096|50,469%|69.36%|
|(3)|3|8192|35,687%|69.36%|
|(3)|3|16384|25,234%|69.36%|
|(3)|3|32768|17,843%|69.36%|
|(3)|5|2048|71,374%|73.64%|
|(3)|5|4096|50,469%|73.64%|
|(3)|5|8192|35,687%|73.64%|
|(3)|5|16384|25,234%|73.64%|
|(3)|5|32768|17,843%|73.64%|
|(3)|7|2048|71,374%|76.97%|
|(3)|7|4096|50,469%|76.97%|
|(3)|7|8192|35,687%|76.97%|
|(3)|7|16384|25,234%|76.97%|
|(3)|7|32768|17,843%|76.97%|
|(4)|3|2048|713,738%|69.36%|
|(4)|3|4096|504,689%|69.36%|
|(4)|3|8192|356,869%|69.36%|
|(4)|3|16384|252,345%|69.36%|
|(4)|3|32768|178,435%|69.36%|
|(4)|5|2048|713,738%|73.64%|
|(4)|5|4096|504,689%|73.64%|
|(4)|5|8192|356,869%|73.64%|
|(4)|5|16384|252,345%|73.64%|
|(4)|5|32768|178,435%|73.64%|
|(4)|7|2048|713,738%|76.97%|
|(4)|7|4096|504,689%|76.97%|
|(4)|7|8192|356,869%|76.97%|
|(4)|7|16384|252,345%|76.97%|
|(4)|7|32768|178,435%|76.97%|
|(5)|3|2048|64,744%|69.36%|
|(5)|3|4096|45,781%|69.36%|
|(5)|3|8192|32,372%|69.36%|
|(5)|3|16384|22,890%|69.36%|
|(5)|3|32768|16,186%|69.36%|
|(5)|5|2048|64,744%|73.64%|
|(5)|5|4096|45,781%|73.64%|
|(5)|5|8192|32,372%|73.64%|
|(5)|5|16384|22,890%|73.64%|
|(5)|5|32768|16,186%|73.64%|
|(5)|7|2048|64,744%|76.97%|
|(5)|7|4096|45,781%|76.97%|
|(5)|7|8192|32,372%|76.97%|
|(5)|7|16384|22,890%|76.97%|
|(5)|7|32768|16,186%|76.97%|
|(6)|3|2048|647,437%|69.36%|
|(6)|3|4096|457,807%|69.36%|
|(6)|3|8192|323,719%|69.36%|
|(6)|3|16384|228,904%|69.36%|
|(6)|3|32768|161,859%|69.36%|
|(6)|5|2048|647,437%|73.64%|
|(6)|5|4096|457,807%|73.64%|
|(6)|5|8192|323,719%|73.64%|
|(6)|5|16384|228,904%|73.64%|
|(6)|5|32768|161,859%|73.64%|
|(6)|7|2048|647,437%|76.97%|
|(6)|7|4096|457,807%|76.97%|
|(6)|7|8192|323,719%|76.97%|
|(6)|7|16384|228,904%|76.97%|
|(6)|7|32768|161,859%|76.97%|
|(7)|3|2048|1,158%|69.36%|
|(7)|3|4096|818.7%|69.36%|
|(7)|3|8192|578.9%|69.36%|
|(7)|3|16384|409.4%|69.36%|
|(7)|3|32768|289.5%|69.36%|
|(7)|5|2048|1,158%|73.64%|
|(7)|5|4096|818.7%|73.64%|
|(7)|5|8192|578.9%|73.64%|
|(7)|5|16384|409.4%|73.64%|
|(7)|5|32768|289.5%|73.64%|
|(7)|7|2048|1,158%|76.97%|
|(7)|7|4096|818.7%|76.97%|
|(7)|7|8192|578.9%|76.97%|
|(7)|7|16384|409.4%|76.97%|
|(7)|7|32768|289.5%|76.97%|
|(8)|3|2048|3,821%|69.36%|
|(8)|3|4096|2,702%|69.36%|
|(8)|3|8192|1,911%|69.36%|
|(8)|3|16384|1,351%|69.36%|
|(8)|3|32768|955.3%|69.36%|
|(8)|5|2048|3,821%|73.64%|
|(8)|5|4096|2,702%|73.64%|
|(8)|5|8192|1,911%|73.64%|
|(8)|5|16384|1,351%|73.64%|
|(8)|5|32768|955.3%|73.64%|
|(8)|7|2048|3,821%|76.97%|
|(8)|7|4096|2,702%|76.97%|
|(8)|7|8192|1,911%|76.97%|
|(8)|7|16384|1,351%|76.97%|
|(8)|7|32768|955.3%|76.97%|
|(9)|3|2048|71,374%|69.36%|
|(9)|3|4096|50,469%|69.36%|
|(9)|3|8192|35,687%|69.36%|
|(9)|3|16384|25,234%|69.36%|
|(9)|3|32768|17,843%|69.36%|
|(9)|5|2048|71,374%|73.64%|
|(9)|5|4096|50,469%|73.64%|
|(9)|5|8192|35,687%|73.64%|
|(9)|5|16384|25,234%|73.64%|
|(9)|5|32768|17,843%|73.64%|
|(9)|7|2048|71,374%|76.97%|
|(9)|7|4096|50,469%|76.97%|
|(9)|7|8192|35,687%|76.97%|
|(9)|7|16384|25,234%|76.97%|
|(9)|7|32768|17,843%|76.97%|
|(10)|3|2048|713,738%|69.36%|
|(10)|3|4096|504,689%|69.36%|
|(10)|3|8192|356,869%|69.36%|
|(10)|3|16384|252,345%|69.36%|
|(10)|3|32768|178,435%|69.36%|
|(10)|5|2048|713,738%|73.64%|
|(10)|5|4096|504,689%|73.64%|
|(10)|5|8192|356,869%|73.64%|
|(10)|5|16384|252,345%|73.64%|
|(10)|5|32768|178,435%|73.64%|
|(10)|7|2048|713,738%|76.97%|
|(10)|7|4096|504,689%|76.97%|
|(10)|7|8192|356,869%|76.97%|
|(10)|7|16384|252,345%|76.97%|
|(10)|7|32768|178,435%|76.97%|
|(11)|3|2048|64,744%|69.36%|
|(11)|3|4096|45,781%|69.36%|
|(11)|3|8192|32,372%|69.36%|
|(11)|3|16384|22,890%|69.36%|
|(11)|3|32768|16,186%|69.36%|
|(11)|5|2048|64,744%|73.64%|
|(11)|5|4096|45,781%|73.64%|
|(11)|5|8192|32,372%|73.64%|
|(11)|5|16384|22,890%|73.64%|
|(11)|5|32768|16,186%|73.64%|
|(11)|7|2048|64,744%|76.97%|
|(11)|7|4096|45,781%|76.97%|
|(11)|7|8192|32,372%|76.97%|
|(11)|7|16384|22,890%|76.97%|
|(11)|7|32768|16,186%|76.97%|
|(12)|3|2048|647,437%|69.36%|
|(12)|3|4096|457,807%|69.36%|
|(12)|3|8192|323,719%|69.36%|
|(12)|3|16384|228,904%|69.36%|
|(12)|3|32768|161,859%|69.36%|
|(12)|5|2048|647,437%|73.64%|
|(12)|5|4096|457,807%|73.64%|
|(12)|5|8192|323,719%|73.64%|
|(12)|5|16384|228,904%|73.64%|
|(12)|5|32768|161,859%|73.64%|
|(12)|7|2048|647,437%|76.97%|
|(12)|7|4096|457,807%|76.97%|
|(12)|7|8192|323,719%|76.97%|
|(12)|7|16384|228,904%|76.97%|
|(12)|7|32768|161,859%|76.97%|

- CountL2HH
  - Configuration: row 4, col 2048
  - Input: (3) ~ (6), (9) ~ (12), with weighted and negative updates
  - error bound:
    - hottest key after a decrement: relative error 2%
    - `F2`: relative error 10%
- FoldCMS / FoldCS
  - Configuration: row 3, col 2048, fold level 0, top_k 32
  - counts are exact on sparse dims, including signed weighted updates through FoldCS
  - same-level merge sums disjoint contributions; hierarchical merge of level-matched sketches preserves totals

## e2e_heavy_hitters

- CountMin sketch
  - both vector based and fixed matrix based
  - both FastPath and RegularPath
  - size, combination of row and column:
    - row: 3, 5, 7
    - column: 2048, 4096, 8192, 16384, 32768
  - heavy hitter definition:
    - 1% key with highest appearance
  - per key error bound (theoretical, worst heavy hitter):
    - `eps = e/w`, `err <= eps * ||f||_1 = e*N/w`, `delta = e^-d`
    - relative error column = `err / f_HH`; in range column = `1 - delta`
    - `f_HH` (frequency of the 1% rank key): (1)(7) 2, (2)(8) 3, (3)(9) 285, (4)(10) 2700, (5)(11) 87, (6)(12) 433

|input|row|column|per-key relative error|in range key percentage|
|---|---|---|---|---|
|(1)|3|2048|6,636%|95.02%|
|(1)|3|4096|3,318%|95.02%|
|(1)|3|8192|1,659%|95.02%|
|(1)|3|16384|829.6%|95.02%|
|(1)|3|32768|414.8%|95.02%|
|(1)|5|2048|6,636%|99.33%|
|(1)|5|4096|3,318%|99.33%|
|(1)|5|8192|1,659%|99.33%|
|(1)|5|16384|829.6%|99.33%|
|(1)|5|32768|414.8%|99.33%|
|(1)|7|2048|6,636%|99.91%|
|(1)|7|4096|3,318%|99.91%|
|(1)|7|8192|1,659%|99.91%|
|(1)|7|16384|829.6%|99.91%|
|(1)|7|32768|414.8%|99.91%|
|(2)|3|2048|44,243%|95.02%|
|(2)|3|4096|22,121%|95.02%|
|(2)|3|8192|11,061%|95.02%|
|(2)|3|16384|5,530%|95.02%|
|(2)|3|32768|2,765%|95.02%|
|(2)|5|2048|44,243%|99.33%|
|(2)|5|4096|22,121%|99.33%|
|(2)|5|8192|11,061%|99.33%|
|(2)|5|16384|5,530%|99.33%|
|(2)|5|32768|2,765%|99.33%|
|(2)|7|2048|44,243%|99.91%|
|(2)|7|4096|22,121%|99.91%|
|(2)|7|8192|11,061%|99.91%|
|(2)|7|16384|5,530%|99.91%|
|(2)|7|32768|2,765%|99.91%|
|(3)|3|2048|46.5%|95.02%|
|(3)|3|4096|23.3%|95.02%|
|(3)|3|8192|11.6%|95.02%|
|(3)|3|16384|5.8%|95.02%|
|(3)|3|32768|2.9%|95.02%|
|(3)|5|2048|46.5%|99.33%|
|(3)|5|4096|23.3%|99.33%|
|(3)|5|8192|11.6%|99.33%|
|(3)|5|16384|5.8%|99.33%|
|(3)|5|32768|2.9%|99.33%|
|(3)|7|2048|46.5%|99.91%|
|(3)|7|4096|23.3%|99.91%|
|(3)|7|8192|11.6%|99.91%|
|(3)|7|16384|5.8%|99.91%|
|(3)|7|32768|2.9%|99.91%|
|(4)|3|2048|49.2%|95.02%|
|(4)|3|4096|24.6%|95.02%|
|(4)|3|8192|12.3%|95.02%|
|(4)|3|16384|6.1%|95.02%|
|(4)|3|32768|3.1%|95.02%|
|(4)|5|2048|49.2%|99.33%|
|(4)|5|4096|24.6%|99.33%|
|(4)|5|8192|12.3%|99.33%|
|(4)|5|16384|6.1%|99.33%|
|(4)|5|32768|3.1%|99.33%|
|(4)|7|2048|49.2%|99.91%|
|(4)|7|4096|24.6%|99.91%|
|(4)|7|8192|12.3%|99.91%|
|(4)|7|16384|6.1%|99.91%|
|(4)|7|32768|3.1%|99.91%|
|(5)|3|2048|152.5%|95.02%|
|(5)|3|4096|76.2%|95.02%|
|(5)|3|8192|38.1%|95.02%|
|(5)|3|16384|19.1%|95.02%|
|(5)|3|32768|9.5%|95.02%|
|(5)|5|2048|152.5%|99.33%|
|(5)|5|4096|76.2%|99.33%|
|(5)|5|8192|38.1%|99.33%|
|(5)|5|16384|19.1%|99.33%|
|(5)|5|32768|9.5%|99.33%|
|(5)|7|2048|152.5%|99.91%|
|(5)|7|4096|76.2%|99.91%|
|(5)|7|8192|38.1%|99.91%|
|(5)|7|16384|19.1%|99.91%|
|(5)|7|32768|9.5%|99.91%|
|(6)|3|2048|306.4%|95.02%|
|(6)|3|4096|153.2%|95.02%|
|(6)|3|8192|76.6%|95.02%|
|(6)|3|16384|38.3%|95.02%|
|(6)|3|32768|19.1%|95.02%|
|(6)|5|2048|306.4%|99.33%|
|(6)|5|4096|153.2%|99.33%|
|(6)|5|8192|76.6%|99.33%|
|(6)|5|16384|38.3%|99.33%|
|(6)|5|32768|19.1%|99.33%|
|(6)|7|2048|306.4%|99.91%|
|(6)|7|4096|153.2%|99.91%|
|(6)|7|8192|76.6%|99.91%|
|(6)|7|16384|38.3%|99.91%|
|(6)|7|32768|19.1%|99.91%|
|(7)|3|2048|6,636%|95.02%|
|(7)|3|4096|3,318%|95.02%|
|(7)|3|8192|1,659%|95.02%|
|(7)|3|16384|829.6%|95.02%|
|(7)|3|32768|414.8%|95.02%|
|(7)|5|2048|6,636%|99.33%|
|(7)|5|4096|3,318%|99.33%|
|(7)|5|8192|1,659%|99.33%|
|(7)|5|16384|829.6%|99.33%|
|(7)|5|32768|414.8%|99.33%|
|(7)|7|2048|6,636%|99.91%|
|(7)|7|4096|3,318%|99.91%|
|(7)|7|8192|1,659%|99.91%|
|(7)|7|16384|829.6%|99.91%|
|(7)|7|32768|414.8%|99.91%|
|(8)|3|2048|44,243%|95.02%|
|(8)|3|4096|22,121%|95.02%|
|(8)|3|8192|11,061%|95.02%|
|(8)|3|16384|5,530%|95.02%|
|(8)|3|32768|2,765%|95.02%|
|(8)|5|2048|44,243%|99.33%|
|(8)|5|4096|22,121%|99.33%|
|(8)|5|8192|11,061%|99.33%|
|(8)|5|16384|5,530%|99.33%|
|(8)|5|32768|2,765%|99.33%|
|(8)|7|2048|44,243%|99.91%|
|(8)|7|4096|22,121%|99.91%|
|(8)|7|8192|11,061%|99.91%|
|(8)|7|16384|5,530%|99.91%|
|(8)|7|32768|2,765%|99.91%|
|(9)|3|2048|46.5%|95.02%|
|(9)|3|4096|23.3%|95.02%|
|(9)|3|8192|11.6%|95.02%|
|(9)|3|16384|5.8%|95.02%|
|(9)|3|32768|2.9%|95.02%|
|(9)|5|2048|46.5%|99.33%|
|(9)|5|4096|23.3%|99.33%|
|(9)|5|8192|11.6%|99.33%|
|(9)|5|16384|5.8%|99.33%|
|(9)|5|32768|2.9%|99.33%|
|(9)|7|2048|46.5%|99.91%|
|(9)|7|4096|23.3%|99.91%|
|(9)|7|8192|11.6%|99.91%|
|(9)|7|16384|5.8%|99.91%|
|(9)|7|32768|2.9%|99.91%|
|(10)|3|2048|49.2%|95.02%|
|(10)|3|4096|24.6%|95.02%|
|(10)|3|8192|12.3%|95.02%|
|(10)|3|16384|6.1%|95.02%|
|(10)|3|32768|3.1%|95.02%|
|(10)|5|2048|49.2%|99.33%|
|(10)|5|4096|24.6%|99.33%|
|(10)|5|8192|12.3%|99.33%|
|(10)|5|16384|6.1%|99.33%|
|(10)|5|32768|3.1%|99.33%|
|(10)|7|2048|49.2%|99.91%|
|(10)|7|4096|24.6%|99.91%|
|(10)|7|8192|12.3%|99.91%|
|(10)|7|16384|6.1%|99.91%|
|(10)|7|32768|3.1%|99.91%|
|(11)|3|2048|152.5%|95.02%|
|(11)|3|4096|76.2%|95.02%|
|(11)|3|8192|38.1%|95.02%|
|(11)|3|16384|19.1%|95.02%|
|(11)|3|32768|9.5%|95.02%|
|(11)|5|2048|152.5%|99.33%|
|(11)|5|4096|76.2%|99.33%|
|(11)|5|8192|38.1%|99.33%|
|(11)|5|16384|19.1%|99.33%|
|(11)|5|32768|9.5%|99.33%|
|(11)|7|2048|152.5%|99.91%|
|(11)|7|4096|76.2%|99.91%|
|(11)|7|8192|38.1%|99.91%|
|(11)|7|16384|19.1%|99.91%|
|(11)|7|32768|9.5%|99.91%|
|(12)|3|2048|306.4%|95.02%|
|(12)|3|4096|153.2%|95.02%|
|(12)|3|8192|76.6%|95.02%|
|(12)|3|16384|38.3%|95.02%|
|(12)|3|32768|19.1%|95.02%|
|(12)|5|2048|306.4%|99.33%|
|(12)|5|4096|153.2%|99.33%|
|(12)|5|8192|76.6%|99.33%|
|(12)|5|16384|38.3%|99.33%|
|(12)|5|32768|19.1%|99.33%|
|(12)|7|2048|306.4%|99.91%|
|(12)|7|4096|153.2%|99.91%|
|(12)|7|8192|76.6%|99.91%|
|(12)|7|16384|38.3%|99.91%|
|(12)|7|32768|19.1%|99.91%|

- Count sketch
  - both vector based and fixed matrix based
  - both FastPath and RegularPath
  - size, combination of row and column:
    - row: 3, 5, 7
    - column: 2048, 4096, 8192, 16384, 32768
  - heavy hitter definition:
    - 1% key with highest appearance
  - per key error bound (theoretical, worst heavy hitter):
    - `eps = sqrt(e/w)`, `err <= eps * ||f||_2`, per-row failure `<= 1/e` (Chebyshev), median of `d` rows
    - relative error column = `err / f_HH`; in range column = median-of-rows success
    - `||f||_2`: (1)(7) 318, (2)(8) 1049, (3)(9) 19591, (4)(10) 195910, (5)(11) 17771, (6)(12) 177712
    - `f_HH` (frequency of the 1% rank key): (1)(7) 2, (2)(8) 3, (3)(9) 285, (4)(10) 2700, (5)(11) 87, (6)(12) 433

|input|row|column|per-key relative error|in range key percentage|
|---|---|---|---|---|
|(1)|3|2048|578.9%|69.36%|
|(1)|3|4096|409.4%|69.36%|
|(1)|3|8192|289.5%|69.36%|
|(1)|3|16384|204.7%|69.36%|
|(1)|3|32768|144.7%|69.36%|
|(1)|5|2048|578.9%|73.64%|
|(1)|5|4096|409.4%|73.64%|
|(1)|5|8192|289.5%|73.64%|
|(1)|5|16384|204.7%|73.64%|
|(1)|5|32768|144.7%|73.64%|
|(1)|7|2048|578.9%|76.97%|
|(1)|7|4096|409.4%|76.97%|
|(1)|7|8192|289.5%|76.97%|
|(1)|7|16384|204.7%|76.97%|
|(1)|7|32768|144.7%|76.97%|
|(2)|3|2048|1,274%|69.36%|
|(2)|3|4096|900.6%|69.36%|
|(2)|3|8192|636.8%|69.36%|
|(2)|3|16384|450.3%|69.36%|
|(2)|3|32768|318.4%|69.36%|
|(2)|5|2048|1,274%|73.64%|
|(2)|5|4096|900.6%|73.64%|
|(2)|5|8192|636.8%|73.64%|
|(2)|5|16384|450.3%|73.64%|
|(2)|5|32768|318.4%|73.64%|
|(2)|7|2048|1,274%|76.97%|
|(2)|7|4096|900.6%|76.97%|
|(2)|7|8192|636.8%|76.97%|
|(2)|7|16384|450.3%|76.97%|
|(2)|7|32768|318.4%|76.97%|
|(3)|3|2048|250.2%|69.36%|
|(3)|3|4096|176.9%|69.36%|
|(3)|3|8192|125.1%|69.36%|
|(3)|3|16384|88.5%|69.36%|
|(3)|3|32768|62.6%|69.36%|
|(3)|5|2048|250.2%|73.64%|
|(3)|5|4096|176.9%|73.64%|
|(3)|5|8192|125.1%|73.64%|
|(3)|5|16384|88.5%|73.64%|
|(3)|5|32768|62.6%|73.64%|
|(3)|7|2048|250.2%|76.97%|
|(3)|7|4096|176.9%|76.97%|
|(3)|7|8192|125.1%|76.97%|
|(3)|7|16384|88.5%|76.97%|
|(3)|7|32768|62.6%|76.97%|
|(4)|3|2048|264.4%|69.36%|
|(4)|3|4096|186.9%|69.36%|
|(4)|3|8192|132.2%|69.36%|
|(4)|3|16384|93.5%|69.36%|
|(4)|3|32768|66.1%|69.36%|
|(4)|5|2048|264.4%|73.64%|
|(4)|5|4096|186.9%|73.64%|
|(4)|5|8192|132.2%|73.64%|
|(4)|5|16384|93.5%|73.64%|
|(4)|5|32768|66.1%|73.64%|
|(4)|7|2048|264.4%|76.97%|
|(4)|7|4096|186.9%|76.97%|
|(4)|7|8192|132.2%|76.97%|
|(4)|7|16384|93.5%|76.97%|
|(4)|7|32768|66.1%|76.97%|
|(5)|3|2048|743.8%|69.36%|
|(5)|3|4096|526.0%|69.36%|
|(5)|3|8192|371.9%|69.36%|
|(5)|3|16384|263.0%|69.36%|
|(5)|3|32768|186.0%|69.36%|
|(5)|5|2048|743.8%|73.64%|
|(5)|5|4096|526.0%|73.64%|
|(5)|5|8192|371.9%|73.64%|
|(5)|5|16384|263.0%|73.64%|
|(5)|5|32768|186.0%|73.64%|
|(5)|7|2048|743.8%|76.97%|
|(5)|7|4096|526.0%|76.97%|
|(5)|7|8192|371.9%|76.97%|
|(5)|7|16384|263.0%|76.97%|
|(5)|7|32768|186.0%|76.97%|
|(6)|3|2048|1,494%|69.36%|
|(6)|3|4096|1,057%|69.36%|
|(6)|3|8192|747.2%|69.36%|
|(6)|3|16384|528.4%|69.36%|
|(6)|3|32768|373.6%|69.36%|
|(6)|5|2048|1,494%|73.64%|
|(6)|5|4096|1,057%|73.64%|
|(6)|5|8192|747.2%|73.64%|
|(6)|5|16384|528.4%|73.64%|
|(6)|5|32768|373.6%|73.64%|
|(6)|7|2048|1,494%|76.97%|
|(6)|7|4096|1,057%|76.97%|
|(6)|7|8192|747.2%|76.97%|
|(6)|7|16384|528.4%|76.97%|
|(6)|7|32768|373.6%|76.97%|
|(7)|3|2048|578.9%|69.36%|
|(7)|3|4096|409.4%|69.36%|
|(7)|3|8192|289.5%|69.36%|
|(7)|3|16384|204.7%|69.36%|
|(7)|3|32768|144.7%|69.36%|
|(7)|5|2048|578.9%|73.64%|
|(7)|5|4096|409.4%|73.64%|
|(7)|5|8192|289.5%|73.64%|
|(7)|5|16384|204.7%|73.64%|
|(7)|5|32768|144.7%|73.64%|
|(7)|7|2048|578.9%|76.97%|
|(7)|7|4096|409.4%|76.97%|
|(7)|7|8192|289.5%|76.97%|
|(7)|7|16384|204.7%|76.97%|
|(7)|7|32768|144.7%|76.97%|
|(8)|3|2048|1,274%|69.36%|
|(8)|3|4096|900.6%|69.36%|
|(8)|3|8192|636.8%|69.36%|
|(8)|3|16384|450.3%|69.36%|
|(8)|3|32768|318.4%|69.36%|
|(8)|5|2048|1,274%|73.64%|
|(8)|5|4096|900.6%|73.64%|
|(8)|5|8192|636.8%|73.64%|
|(8)|5|16384|450.3%|73.64%|
|(8)|5|32768|318.4%|73.64%|
|(8)|7|2048|1,274%|76.97%|
|(8)|7|4096|900.6%|76.97%|
|(8)|7|8192|636.8%|76.97%|
|(8)|7|16384|450.3%|76.97%|
|(8)|7|32768|318.4%|76.97%|
|(9)|3|2048|250.2%|69.36%|
|(9)|3|4096|176.9%|69.36%|
|(9)|3|8192|125.1%|69.36%|
|(9)|3|16384|88.5%|69.36%|
|(9)|3|32768|62.6%|69.36%|
|(9)|5|2048|250.2%|73.64%|
|(9)|5|4096|176.9%|73.64%|
|(9)|5|8192|125.1%|73.64%|
|(9)|5|16384|88.5%|73.64%|
|(9)|5|32768|62.6%|73.64%|
|(9)|7|2048|250.2%|76.97%|
|(9)|7|4096|176.9%|76.97%|
|(9)|7|8192|125.1%|76.97%|
|(9)|7|16384|88.5%|76.97%|
|(9)|7|32768|62.6%|76.97%|
|(10)|3|2048|264.4%|69.36%|
|(10)|3|4096|186.9%|69.36%|
|(10)|3|8192|132.2%|69.36%|
|(10)|3|16384|93.5%|69.36%|
|(10)|3|32768|66.1%|69.36%|
|(10)|5|2048|264.4%|73.64%|
|(10)|5|4096|186.9%|73.64%|
|(10)|5|8192|132.2%|73.64%|
|(10)|5|16384|93.5%|73.64%|
|(10)|5|32768|66.1%|73.64%|
|(10)|7|2048|264.4%|76.97%|
|(10)|7|4096|186.9%|76.97%|
|(10)|7|8192|132.2%|76.97%|
|(10)|7|16384|93.5%|76.97%|
|(10)|7|32768|66.1%|76.97%|
|(11)|3|2048|743.8%|69.36%|
|(11)|3|4096|526.0%|69.36%|
|(11)|3|8192|371.9%|69.36%|
|(11)|3|16384|263.0%|69.36%|
|(11)|3|32768|186.0%|69.36%|
|(11)|5|2048|743.8%|73.64%|
|(11)|5|4096|526.0%|73.64%|
|(11)|5|8192|371.9%|73.64%|
|(11)|5|16384|263.0%|73.64%|
|(11)|5|32768|186.0%|73.64%|
|(11)|7|2048|743.8%|76.97%|
|(11)|7|4096|526.0%|76.97%|
|(11)|7|8192|371.9%|76.97%|
|(11)|7|16384|263.0%|76.97%|
|(11)|7|32768|186.0%|76.97%|
|(12)|3|2048|1,494%|69.36%|
|(12)|3|4096|1,057%|69.36%|
|(12)|3|8192|747.2%|69.36%|
|(12)|3|16384|528.4%|69.36%|
|(12)|3|32768|373.6%|69.36%|
|(12)|5|2048|1,494%|73.64%|
|(12)|5|4096|1,057%|73.64%|
|(12)|5|8192|747.2%|73.64%|
|(12)|5|16384|528.4%|73.64%|
|(12)|5|32768|373.6%|73.64%|
|(12)|7|2048|1,494%|76.97%|
|(12)|7|4096|1,057%|76.97%|
|(12)|7|8192|747.2%|76.97%|
|(12)|7|16384|528.4%|76.97%|
|(12)|7|32768|373.6%|76.97%|

- elastic sketch
  - Configuration: length 64
  - Input: (3) ~ (6), (9) ~ (12)
  - heavy hitter definition:
    - 3 hot flows at ~10% of the stream each, against ~977 background flows
  - error bound: per hot flow, relative error 20%
    - reasoning: the split between the light and heavy parts is data dependent, so the bound is empirical rather than derived
- Coco
  - Configuration: table size 256, 2 ways
  - Input: disjoint key prefixes (`aaa*` over 50 keys at weight 7, `zzz*` over 30 keys at weight 3)
  - error bound: per prefix family, `est` in `[0.75 * truth, N]`, and a prefix must never pick up a bucket from another prefix
    - reasoning: eviction loses counts one-sidedly downwards while over-attribution is capped by the stream total, so only that band is guaranteed
- SpaceSaving
  - Configuration (capacity): 2, 8, 64, 2048, and the default 1024
  - Input: (1) ~ (12)
  - heavy hitter definition:
    - the monitored keys the Stream-Summary still holds
  - per key error bound (theoretical, worst case):
    - `f(k) <= est <= f(k) + min_count`, and `min_count <= N / m` with `m` the capacity
    - the table is the absolute over-estimate `N / m`, not a relative error

|capacity|N=100K -- (1)(3)(5)(7)(9)(11)|N=1M -- (2)(4)(6)(8)(10)(12)|
|---|---|---|
|2|50,000|500,000|
|8|12,500|125,000|
|64|1,563|15,625|
|1024 (default)|97.7|977|
|2048|48.8|488|

## e2e_membership

- Bloom filter
  - input: 20K distinct members, 200K disjoint probes per measured rate
  - configuration
    - both FastPath and RegularPath
    - `with_capacity(20K, target)` for targets 0.1, 0.01, 0.001; explicit `with_dimensions` geometries for the hash layouts and the degenerate cases
  - exact structure: no inserted member reads back absent, and a union equals the filter of the concatenated stream
  - error bound:
    - measured false positive rate within five binomial standard errors of `predicted_fpp`, and at or under the target
    - `estimated_fpp` within 0.75-1.25x of the measured rate
  - sizing, the default geometry and the allocation ceiling are checked as exact geometries rather than rates


## e2e_nitro

- NitroBatch
  - target sketches: CountMin (row 5, col 2048, FastPath) and Count sketch (row 5, col 2048, FastPath)
  - Configuration (sampling rate): 1.0, 0.5
  - Input: one key repeated 100K times, so the rescaling of the estimator is what the bound measures
  - error bound:
    - CountMin target: relative error 5%
    - Count sketch target: relative error 10%

## e2e_octo

- Octo
  - input: zipf(1.1) streams of 40K to 400K keys over a 2048 to 8192 key domain, dispatched to 2, 3 or 4 workers under both the `HashByKey` and `RoundRobin` partitions
  - configuration:
    - CMS: row 5, col 2048 or row 5, col 4096,  with the worker promotion threshold tau = 31
    - CS: row 5, col 4096, with the worker promotion threshold tau = 31
    - DD: alpha = 0.01, with the worker promotion threshold tau = 4
    - HLL: Classic at P14 for the protocol runs, and P12/P14/P16 across Classic and ErtlMLE for the exactness sweep, with the worker promotion threshold tau = 0 so every register improvement promotes at once
    - UnivMon: heap 64, row 5, col 1024, 12 layers, with the worker promotion threshold tau = 31, applied both flat across the pyramid and scaled per layer
    - Coco: 512 buckets, 2 ways, with the worker promotion threshold tau = 31
    - Elastic: 256 heavy buckets over a light part of row 3, col 2048, with the worker promotion threshold tau = 31
  - error bound: counts a promotion can hold back per counter on top of the standalone sketch's own bound, `workers * tau` at 4 workers
    - CMS: 124
    - CS: 124
    - DD: 16
    - HLL: 0
    - UnivMon: 124
    - Coco: 124
    - Elastic: 124

## e2e_quantiles

- KLL
  - Input: (1) ~ (12)
  - Configuration: k = 50, 200, 800
  - Query: P0, P10, P20, P30, P40, P50, P60, P70, P80, P90, P100
  - Error bound: rank error
    - k = 50: 6.11%
    - k = 200: 1.65%
    - k = 800: 0.447%
- KLL_Dynamic
  - Input: (1) ~ (12)
  - Configuration: k = 50, 200, 800
  - Query: P0, P10, P20, P30, P40, P50, P60, P70, P80, P90, P100
  - Error bound: rank error
    - k = 50: 6.11%
    - k = 200: 1.65%
    - k = 800: 0.447%
- DD
  - Input: (1) ~ (12)
  - Configuration: alpha = 0.1, 0.01, 0.001
  - Query: P0, P10, P20, P30, P40, P50, P60, P70, P80, P90, P100
  - Error bound: relative error
    - alpha = 0.1: 10%
    - alpha = 0.01: 1%
    - alpha = 0.001: 0.1%
- UnivMonQ
  - Input: (1) ~ (12)
  - Configuration: levels 10, width 4096, depth 5, candidates 1024, ordered_samples 1024
    - `ordered_samples = 0` disables rank/CDF/quantile entirely, covered as a negative case
  - Query: P0, P10, P20, P30, P40, P50, P60, P70, P80, P90, P100, plus rank(), cdf() and heavy hitters
  - Error bound:
    - quantile: rank error 4%
    - rank(): relative error 6%
    - count, min, max: exact
    - heaviest key frequency: -5% / +10%
    - distinct: -10% / +15%
    - `F2`: relative error 15%
    - entropy: -10% / +15%
    - reasoning: ordered queries ride on the top-level CountSketch (depth 5, width 4096) plus a bounded ordered sample, so the bound is empirical rather than the CountSketch bound alone
- checks applied to every sketch in this section
  - shard merge: the merged sketch keeps the same bound
  - DDSketch drops non-finite, non-positive and non-indexable values without corrupting bucket 0 and without letting one sample force a distant-bucket allocation

## e2e_topk

- CMS_heap
  - Input: (3) ~ (6), (9) ~ (12)
  - Configuration:
    - top_k: 32, 64, 128
    - cms: row: 5; col: 32768
    - reasoning: CMS accuracy is tested elsewhere, so here is to test how the heap performs E2E
  - Error bound: for items in heap, relative error should be less than 2%

- CS_heap
  - Input: (3) ~ (6), (9) ~ (12)
  - Configuration:
    - top_k: 32, 64, 128
    - cs: row: 5; col: 32768
    - reasoning: CMS accuracy is tested elsewhere, so here is to test how the heap performs E2E
  - Error bound: for items in heap, relative error should be less than 2%

- checks applied to both heaps
  - capacity: the heap must never hold more than `top_k` entries
  - recall: at least `top_k - 1` of the entries must be true top-k keys
  - consistency: every heap entry's stored count must equal the sketch's current estimate for that key

## e2e_windows

- ExponentialHistogram
  - Configuration: k=8, window 100, payload CountMin (row 3, col 2048, FastPath)
  - Input: (3) ~ (6), (14)
  - error bound: interval count relative error 21%
    - reasoning: interval merges snap to bucket boundaries in both directions, so the error is granularity, not sketch collision
  - expiry: the retained span must be covered, and anything past the observed maximum time must not be
- TumblingWindow<FoldCMS>
  - Configuration: window 10, 16 slots, FoldCMS row 3, col 2048, fold level 0, top_k 32
  - counts are exact per window
  - `query_all` and `query_recent(n)` cover exactly the expected spans; `flush` closes the active window even when it is partially filled or empty
- TumblingWindow<KLL>
  - Configuration: window 100, KLL k=200
  - Query: P50, P90
  - Error bound: rank error
    - `query_all` and `query_recent(1)`: 5%
    - active window alone: 6%
- ExponentialHistogram, variant matrix
  - Configuration: k=8, window 1,000,000 for the accuracy runs so nothing expires inside them; payload row 3, col 512; UnivMon payload row 5, col 2048
  - Input: 10K keys over a 2048 key domain
  - each variant keeps its own bound over the retained window:
    - CountMin: the CountMin model -- the merged window estimate is one-sided and its excess is held to `eps * N` over the events still retained, exactly as for a standalone CountMin at row 3, col 512
    - Count sketch: the L2 model
    - CountL2HH: the L2 bound
    - Coco and Elastic: one-sided -- on the 32 heaviest keys of the retained window a query must never read low, which is the only direction a structure that evicts flow keys can guarantee
    - HyperLogLog: the register error model
    - KLL: the rank error characterization
    - DDSketch: the relative value error contract
    - UnivMon: exact L1
    - UniformSampling: exact retention bookkeeping
  - other properties:
    - every variant selects the documented merge norm, and can merge into its own kind
    - buckets past the window expire, the retained span is reported, and expiry follows the window length the histogram was last given
    - a custom bucket update matches repeated inserts
- MicroCM (feature-gated behind `--features experimental`)
  - Configuration: row 4, col 1024, T=8, c=2, count-based clock of 1000 items per sub-window
  - Input: 40K zipf i64, s=1.1, key-size=4096
  - reference window: `DeltaStrategy::Over` charges sub-windows `n-T ..= n` in full, and a count-based clock puts item `i` in sub-window `i / L`, so the exact window is items `[(n-T) * L, seen)` -- no tolerance is spent on granularity
  - error bound: the CountMin model over that exact window, one-sided, at the simultaneous additive bound for row 4, col 1024
    - the load is sized to stay at zoom 0 and the test asserts it, because a zoom adds rounding the CountMin contract does not model
  - other properties:
    - `Linear` sits exactly at the clock's residual fraction between `Under` and `Over`, for all 4096 keys, measured part-way through a sub-window; the bare ordering is not asserted, since scaling one non-negative quantity by 0, `p` and 1 orders itself whatever the cells hold
    - a key that stops appearing leaves the window **exactly**: the filler traffic is filtered to keys that provably share no cell with it, so the assertion is `== 0.0` rather than a collision bound. Expiry comes from the sub-window boundary sweep, which visits every cell, not from writes to that key's cells

## conformance_kit

- MicroCM (feature-gated behind `--features experimental`)
  - Configuration: row 4, col 4096, T=64, c=2, count-based clock of 1000 items per sub-window
  - Input: the kit's shared 60K zipf i64 stream, s=1.1, key-size=2048
  - batteries: `frequency_battery` with `one_sided: true`, and `merge_equivalence_battery`
  - the two premises the contract rests on are asserted, not assumed:
    - the whole stream fits inside one window, so the reference is the full per-key truth
    - no cell zooms, so the error is one-sided; past a zoom the probabilistic rounding makes it two-sided and `one_sided: true` would be the wrong contract
  - CountMinHll is not run through the batteries: its query is a grouped distinct count, and no `*Ops` trait in the kit models that

## e2e_wire

- ASAPv1 envelope round trip, one kind at a time
  - kinds covered: DDSketch, Bloom (both hash paths), Coco, Elastic, SpaceSaving, KLLDynamic, the heap-backed matrix types, ExponentialHistogram, EHSketchList, Hydra (naming the counter it carries), the UnivMon family, and the experimental kinds
  - every envelope must round trip and name its own kind
  - an envelope is refused by every decoder but its own
  - the matrix and quantile envelopes carry their answers unchanged across the wire
