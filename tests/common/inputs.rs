//! The documented data inputs of `tests/TEST_COVERAGE.md`, in one place.
//!
//! The coverage document numbers eighteen streams and then writes every suite's
//! matrix in terms of those numbers. A suite that rolls its own stream is
//! therefore covering *something*, but not the row of the table it is filed
//! under. Everything here is that table, verbatim:
//!
//! | id | stream |
//! |---|---|
//! | (1) (2) | uniform `i64` over `[0, 10M)`, 100K / 1M draws |
//! | (3) (4) | Zipf(1.1) `i64` over a 4096-key domain, 100K / 1M draws |
//! | (5) (6) | Zipf(1.1) `i64` over a 20000-key domain, 100K / 1M draws |
//! | (7) ~ (12) | the same six shapes carried on `f64` keys |
//! | (13) | uniform 3-character strings over `A-Za-z0-9` |
//! | (14) | Zipf(1.1) 3-character strings over a 4096-key domain |
//! | (15) (16) | Normal(1000, 250) values, 100K / 1M draws |
//! | (17) | Exponential(1e-3) values |
//! | (18) | log-uniform values on DDSketch bucket edges for a given alpha |
//!
//! # Why the `f64` twins mirror their integer originals draw for draw
//!
//! The document's error tables give `(1)` and `(7)` the same `||f||_1`,
//! `||f||_2` and `f_HH`, and likewise for every other integer/float pair. That
//! is only true if the float stream is the *same multiset of identities*
//! carried on a different `DataInput` variant, so that is what is built here:
//! the draw is made once as an integer and then presented as `f64`. A float
//! stream drawn independently would have its own frequency vector and would
//! not be the row the table describes.
//!
//! # Identity
//!
//! Every stream is reduced to a `Vec<i64>` of canonical key identities so that
//! `FreqTruth` (which is keyed by `i64`) is exact for both encodings. For a
//! float stream the identity is the value's bit pattern, which is injective on
//! non-NaN values, and `KeyInput::data` reverses it exactly.

use asap_sketchlib::DataInput;

use super::streamgen::{exponential_f64, log_uniform_f64, normal_f64, uniform_u64, zipf_u64};
use super::truth::FreqTruth;

/// The twelve keyed streams the frequency, heavy-hitter, cardinality, top-k,
/// composition and quantile matrices are written against.
pub const KEY_INPUT_IDS: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// Draw counts: the odd ids are the 100K streams, the even ids the 1M ones.
const SMALL: usize = 100_000;
const LARGE: usize = 1_000_000;

/// Uniform key range for `(1)`, `(2)`, `(7)` and `(8)`.
const UNIFORM_RANGE: u64 = 10_000_000;

/// The Zipf exponent every skewed stream in the document uses.
const ZIPF_S: f64 = 1.1;

/// The `alphabet` section of the coverage document.
const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// One stream seed per input id, far apart so no two streams share a draw
/// sequence.
fn seed_for(id: u8) -> u64 {
    0x1_0000_0000u64 + id as u64 * 0x9E37_79B9
}

/// How a key identity is presented to a sketch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEncoding {
    /// `DataInput::I64` of the identity itself.
    I64,
    /// `DataInput::F64` of the value whose bit pattern is the identity.
    F64,
}

/// One of the twelve keyed streams, reduced to canonical key identities.
pub struct KeyInput {
    pub id: u8,
    pub label: String,
    pub encoding: KeyEncoding,
    /// Distinct keys the generator could produce, `0` for the unbounded
    /// uniform streams. Only used for context strings.
    pub domain: usize,
    pub keys: Vec<i64>,
}

impl KeyInput {
    /// The stream element for a key identity, in the encoding this input uses.
    pub fn data(&self, key: i64) -> DataInput<'static> {
        match self.encoding {
            KeyEncoding::I64 => DataInput::I64(key),
            KeyEncoding::F64 => DataInput::F64(f64::from_bits(key as u64)),
        }
    }

    /// The numeric value a key identity stands for, for the quantile suites.
    pub fn value(&self, key: i64) -> f64 {
        match self.encoding {
            KeyEncoding::I64 => key as f64,
            KeyEncoding::F64 => f64::from_bits(key as u64),
        }
    }

    /// The stream as numeric values, in arrival order.
    pub fn values(&self) -> Vec<f64> {
        self.keys.iter().map(|k| self.value(*k)).collect()
    }

    /// Exact frequency ground truth over the whole stream.
    pub fn truth(&self) -> FreqTruth {
        let mut truth = FreqTruth::default();
        for k in &self.keys {
            truth.observe(*k);
        }
        truth
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// `(id) shape n=... seed=...`, for failure messages.
    pub fn context(&self) -> String {
        format!(
            "input ({}) {} n={} seed={}",
            self.id,
            self.label,
            self.keys.len(),
            seed_for(self.id)
        )
    }
}

/// Builds one of the twelve keyed streams. Panics on any other id, so a typo
/// in a matrix is a loud failure rather than a silently different stream.
pub fn key_input(id: u8) -> KeyInput {
    let seed = seed_for(id);
    let (n, shape) = match id {
        1 | 3 | 5 | 7 | 9 | 11 => (SMALL, "100K"),
        2 | 4 | 6 | 8 | 10 | 12 => (LARGE, "1M"),
        other => panic!("({other}) is not one of the twelve keyed inputs"),
    };
    let encoding = if id <= 6 {
        KeyEncoding::I64
    } else {
        KeyEncoding::F64
    };

    // The draw is always integer, then presented in the input's own encoding.
    // See the module docs: the float twins must be the same multiset as their
    // integer originals or the document's shared error tables do not apply.
    let (draws, domain, kind) = match id {
        1 | 2 | 7 | 8 => (
            uniform_u64(n, UNIFORM_RANGE, seed),
            0usize,
            format!("uniform [0, {UNIFORM_RANGE})"),
        ),
        3 | 4 | 9 | 10 => (
            zipf_u64(n, 4_096, ZIPF_S, seed),
            4_096,
            format!("zipf({ZIPF_S}) key-size=4096"),
        ),
        5 | 6 | 11 | 12 => (
            zipf_u64(n, 20_000, ZIPF_S, seed),
            20_000,
            format!("zipf({ZIPF_S}) key-size=20k"),
        ),
        _ => unreachable!(),
    };

    let keys = match encoding {
        KeyEncoding::I64 => draws.into_iter().map(|v| v as i64).collect(),
        KeyEncoding::F64 => draws
            .into_iter()
            .map(|v| (v as f64).to_bits() as i64)
            .collect(),
    };

    KeyInput {
        id,
        label: format!("{shape} {kind} {encoding:?}"),
        encoding,
        domain,
        keys,
    }
}

/// One of the two string streams.
pub struct StringInput {
    pub id: u8,
    pub label: String,
    pub keys: Vec<String>,
}

impl StringInput {
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn context(&self) -> String {
        format!(
            "input ({}) {} n={} seed={}",
            self.id,
            self.label,
            self.keys.len(),
            seed_for(self.id)
        )
    }
}

/// The `index`-th 3-character word of the alphabet, in odometer order. The map
/// is injective for `index < 62^3`, so a draw over a domain is a draw over
/// distinct words.
fn word_of(index: usize) -> String {
    let a = ALPHABET.len();
    let bytes = [
        ALPHABET[(index / (a * a)) % a],
        ALPHABET[(index / a) % a],
        ALPHABET[index % a],
    ];
    String::from_utf8(bytes.to_vec()).expect("alphabet is ASCII")
}

/// Builds `(13)` or `(14)`.
pub fn string_input(id: u8) -> StringInput {
    let seed = seed_for(id);
    let alphabet_size = ALPHABET.len() * ALPHABET.len() * ALPHABET.len();
    let (keys, label) = match id {
        13 => (
            uniform_u64(SMALL, alphabet_size as u64, seed)
                .into_iter()
                .map(|v| word_of(v as usize))
                .collect::<Vec<_>>(),
            "100K uniform 3-char strings".to_string(),
        ),
        14 => (
            zipf_u64(SMALL, 4_096, ZIPF_S, seed)
                .into_iter()
                .map(|v| word_of(v as usize))
                .collect::<Vec<_>>(),
            format!("100K zipf({ZIPF_S}) key-size=4096 3-char strings"),
        ),
        other => panic!("({other}) is not one of the string inputs"),
    };
    StringInput { id, label, keys }
}

/// `(15)`, `(16)` and `(17)`: value streams with no key identity.
pub fn value_input(id: u8) -> (String, Vec<f64>) {
    let seed = seed_for(id);
    match id {
        15 => (
            "100K normal(1000, 250)".to_string(),
            normal_f64(SMALL, 1_000.0, 250.0, seed),
        ),
        16 => (
            "1M normal(1000, 250)".to_string(),
            normal_f64(LARGE, 1_000.0, 250.0, seed),
        ),
        17 => (
            "100K exponential(1e-3)".to_string(),
            exponential_f64(SMALL, 1e-3, seed),
        ),
        other => panic!("({other}) is not one of the value inputs"),
    }
}

/// `(18)`: values that land on the bucket edges of a DDSketch with this
/// `alpha`, mixed with interior values. `gamma` comes from the sketch under
/// test, so the stream is adversarial for that sketch specifically.
pub fn ddsketch_edge_input(alpha: f64) -> (String, Vec<f64>) {
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    (
        format!("100K log-uniform on alpha={alpha} bucket edges"),
        log_uniform_f64(SMALL, gamma, 5..40, seed_for(18) + (alpha * 1e6) as u64),
    )
}

/// The document's heavy-hitter definition: the 1% of keys with the highest
/// counts, most frequent first.
///
/// Ties at the boundary are broken by key so the set is deterministic, and at
/// least one key is always returned so a degenerate stream still asserts
/// something.
pub fn top_percent_keys(truth: &FreqTruth, percent: f64) -> Vec<(i64, i64)> {
    let take = ((truth.distinct() as f64 * percent).ceil() as usize).max(1);
    truth.top_k(take)
}
