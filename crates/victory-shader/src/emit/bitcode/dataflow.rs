//! Whole-function DXIL `exec`-mask, known-bits, and bitfield dataflow
//! super-optimizer (`DATAFLOW`).
//!
//! Operates over the full SSA module after lowering and before bitcode
//! serialization, using three interlocking whole-function analyses:
//!
//! 1. **Sparse Conditional Known-Bits Propagation (SCKP):** a worklist
//!    fixed-point over executable CFG blocks/edges and a bit-level lattice
//!    (`KnownBits { width, zero, one }`) across `phi` nodes, loops, integer
//!    arithmetic (exact ripple-carry add/sub, shifts, multiplies, unsigned
//!    div/rem), casts, selects, integer comparisons, and pure DXIL bit/clamp
//!    intrinsics (`UMin`, `UMax`, `IMin`, `IMax`, `Countbits`, `FirstbitLo`,
//!    `FirstbitHi`, `Bfrev`, `legacyF32ToF16`).
//! 2. **Correlated Dominator-Tree & Multi-Predecessor Predicate/Range
//!    Refinement:** propagates branch-implied boolean facts, canonical
//!    comparisons, constant equalities, bit-mask constraints, and
//!    unsigned/signed ranges through dominated subtrees (and intersected
//!    multi-predecessor edges), collapsing redundant `exec`/guard `select`
//!    chains, `v_cndmask` cascades, constant branches, and range-checked
//!    `UMin`/`Tag::Clamp` intrinsics.
//! 3. **Whole-Function Backward Structural Demanded-Bits:** computes a
//!    sound bit-level liveness fixed-point across all CFG blocks and `phi`
//!    nodes so dead upper/lower bitfield packs, redundant shift/mask pairs
//!    (`v_bfe_*`, `v_bfi_b32`, `v_mul_u32_u24`, SDWA, 64-bit packs,
//!    `zext`/`trunc` roundtrips) collapse from first principles.
//!
//! Soundness invariants:
//! - Strictly bit-exact on all defined integer and IEEE-754 float values
//!   (no fast-math, no `fcmp one` emission, shift amounts >= width left
//!   alone, division by zero / signed `MIN / -1` left alone, `undef` never
//!   treated as a constant or self-equal).
//! - Never forwards groupshared (`addrspace(3)`) LDS stores to loads.
//! - Never coalesces or reuses `dx.op.getDimensions` resource size queries.

use std::collections::{HashMap, HashSet, VecDeque};

use super::opt::{Effect, Tree, map_operands, mask, opposite, resolve, signed};
use super::{Bin, Cast, Const, Module, Op, Tag, Term, Ty, TypeDef, V, for_each_operand, pred};

/// Bit-level knowledge of an integer value of `width` bits (`1..=64`).
///
/// Invariants:
/// - `(zero | one) & !mask(width) == 0`
/// - `(zero & one) == 0`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KnownBits {
    pub(super) width: u32,
    pub(super) zero: u64,
    pub(super) one: u64,
}

impl KnownBits {
    #[inline]
    pub(super) fn unknown(width: u32) -> Self {
        debug_assert!((1..=64).contains(&width));
        Self {
            width,
            zero: 0,
            one: 0,
        }
    }

    #[inline]
    pub(super) fn constant(width: u32, val: u64) -> Self {
        debug_assert!((1..=64).contains(&width));
        let m = mask(width);
        let v = val & m;
        Self {
            width,
            zero: (!v) & m,
            one: v,
        }
    }

    /// Tightest `KnownBits` covering every integer in `[min_u, max_u]`.
    pub(super) fn from_unsigned_range(width: u32, min_u: u64, max_u: u64) -> Self {
        let m = mask(width);
        let lo = min_u & m;
        let hi = max_u & m;
        if lo > hi {
            return Self::unknown(width);
        }
        let diff = lo ^ hi;
        if diff == 0 {
            return Self::constant(width, lo);
        }
        let vary_bits = 64 - diff.leading_zeros();
        let common = (!mask(vary_bits)) & m;
        Self {
            width,
            zero: (!lo) & common,
            one: lo & common,
        }
    }

    #[inline]
    pub(super) fn is_constant(self) -> bool {
        (self.zero | self.one) == mask(self.width)
    }

    #[inline]
    pub(super) fn const_val(self) -> Option<u64> {
        self.is_constant().then_some(self.one)
    }

    #[inline]
    pub(super) fn min_u(self) -> u64 {
        self.one
    }

    #[inline]
    pub(super) fn max_u(self) -> u64 {
        (!self.zero) & mask(self.width)
    }

    #[inline]
    pub(super) fn min_s(self) -> i64 {
        let sign = 1u64 << (self.width - 1);
        let raw = self.one | if self.zero & sign == 0 { sign } else { 0 };
        signed(self.width, raw & mask(self.width))
    }

    #[inline]
    pub(super) fn max_s(self) -> i64 {
        let sign = 1u64 << (self.width - 1);
        let hi = self.max_u();
        let raw = if self.one & sign != 0 { hi } else { hi & !sign };
        signed(self.width, raw & mask(self.width))
    }

    #[inline]
    pub(super) fn sign_is_zero(self) -> bool {
        (self.zero >> (self.width - 1)) & 1 != 0
    }

    #[inline]
    pub(super) fn sign_is_one(self) -> bool {
        (self.one >> (self.width - 1)) & 1 != 0
    }

    #[inline]
    pub(super) fn trailing_zeros(self) -> u32 {
        self.zero.trailing_ones().min(self.width)
    }

    #[inline]
    pub(super) fn leading_zeros_u(self) -> u32 {
        self.max_u().leading_zeros() - (64 - self.width)
    }

    /// Lattice join (union of possible concrete values, e.g. at PHI / select).
    #[inline]
    pub(super) fn join(self, other: Self) -> Self {
        debug_assert_eq!(self.width, other.width);
        Self {
            width: self.width,
            zero: self.zero & other.zero,
            one: self.one & other.one,
        }
    }

    /// Lattice meet (intersection of constraints, e.g. branch refinement).
    /// Returns `None` if contradictory (`zero & one != 0`).
    #[inline]
    pub(super) fn refine(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.width, other.width);
        let m = mask(self.width);
        let zero = (self.zero | other.zero) & m;
        let one = (self.one | other.one) & m;
        if zero & one != 0 {
            return None;
        }
        Some(Self {
            width: self.width,
            zero,
            one,
        })
    }

    pub(super) fn and(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let m = mask(self.width);
        Self {
            width: self.width,
            zero: (self.zero | b.zero) & m,
            one: (self.one & b.one) & m,
        }
    }

    pub(super) fn or(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let m = mask(self.width);
        Self {
            width: self.width,
            zero: (self.zero & b.zero) & m,
            one: (self.one | b.one) & m,
        }
    }

    pub(super) fn xor(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let m = mask(self.width);
        Self {
            width: self.width,
            zero: ((self.zero & b.zero) | (self.one & b.one)) & m,
            one: ((self.zero & b.one) | (self.one & b.zero)) & m,
        }
    }

    pub(super) fn not(self) -> Self {
        let m = mask(self.width);
        Self {
            width: self.width,
            zero: self.one & m,
            one: self.zero & m,
        }
    }

    /// Exact ripple-carry addition (`carry_in = false`) or subtraction (`carry_in = true` with `~b`).
    pub(super) fn add_sub(self, b: Self, is_sub: bool) -> Self {
        debug_assert_eq!(self.width, b.width);
        let m = mask(self.width);
        let rhs = if is_sub { b.not() } else { b };
        let cin = u64::from(is_sub);
        let a_min = self.one;
        let a_max = self.max_u();
        let b_min = rhs.one;
        let b_max = rhs.max_u();
        let sum_min = a_min.wrapping_add(b_min).wrapping_add(cin);
        let sum_max = a_max.wrapping_add(b_max).wrapping_add(cin);
        let c_min = sum_min ^ a_min ^ b_min;
        let c_max = sum_max ^ a_max ^ b_max;
        let carry_known = !(c_min ^ c_max);
        let known = (self.zero | self.one) & (rhs.zero | rhs.one) & carry_known;
        Self {
            width: self.width,
            zero: (!sum_min) & known & m,
            one: sum_min & known & m,
        }
    }

    pub(super) fn shl_const(self, k: u32) -> Self {
        debug_assert!(k < self.width);
        let m = mask(self.width);
        Self {
            width: self.width,
            zero: ((self.zero << k) | mask(k)) & m,
            one: (self.one << k) & m,
        }
    }

    pub(super) fn lshr_const(self, k: u32) -> Self {
        debug_assert!(k < self.width);
        let m = mask(self.width);
        let hi_zero = if k == 0 {
            0
        } else {
            (!mask(self.width - k)) & m
        };
        Self {
            width: self.width,
            zero: ((self.zero >> k) | hi_zero) & m,
            one: (self.one >> k) & m,
        }
    }

    pub(super) fn ashr_const(self, k: u32) -> Self {
        debug_assert!(k < self.width);
        let m = mask(self.width);
        let hi_mask = if k == 0 {
            0
        } else {
            (!mask(self.width - k)) & m
        };
        let hi_zero = if self.sign_is_zero() { hi_mask } else { 0 };
        let hi_one = if self.sign_is_one() { hi_mask } else { 0 };
        Self {
            width: self.width,
            zero: ((self.zero >> k) | hi_zero) & m,
            one: ((self.one >> k) | hi_one) & m,
        }
    }

    /// Shift with possibly non-constant shift amount `b`.
    /// Conservatively returns `unknown` whenever `b` could be `>= width`.
    pub(super) fn shift(self, op: Bin, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let w = self.width;
        if b.max_u() >= u64::from(w) {
            return Self::unknown(w);
        }
        if let Some(k) = b.const_val() {
            let k = k as u32;
            return match op {
                Bin::Shl => self.shl_const(k),
                Bin::LShr => self.lshr_const(k),
                Bin::AShr => self.ashr_const(k),
                _ => Self::unknown(w),
            };
        }
        // When the shift amount has at most 3 unknown bits (<= 8 concrete values),
        // take the exact join across all possible shift amounts.
        let unknown = (!(b.zero | b.one)) & mask(w);
        let n_unk = unknown.count_ones();
        if n_unk <= 3 {
            let mut bits = [0u32; 3];
            let mut idx = 0;
            let mut u = unknown;
            while u != 0 {
                bits[idx] = u.trailing_zeros();
                idx += 1;
                u &= u - 1;
            }
            let mut acc: Option<Self> = None;
            for combo in 0u32..(1u32 << n_unk) {
                let mut sh = b.one;
                for (j, &bit) in bits[..idx].iter().enumerate() {
                    if (combo >> j) & 1 != 0 {
                        sh |= 1u64 << bit;
                    }
                }
                if sh >= u64::from(w) {
                    return Self::unknown(w);
                }
                let k = sh as u32;
                let one = match op {
                    Bin::Shl => self.shl_const(k),
                    Bin::LShr => self.lshr_const(k),
                    Bin::AShr => self.ashr_const(k),
                    _ => Self::unknown(w),
                };
                acc = Some(match acc {
                    None => one,
                    Some(prev) => prev.join(one),
                });
            }
            return acc.unwrap_or_else(|| Self::unknown(w));
        }
        let b_min = b.min_u() as u32;
        let b_max = b.max_u() as u32;
        let m = mask(w);
        match op {
            Bin::Shl => {
                let low_zeros = (self.trailing_zeros() + b_min).min(w);
                let mut zero = mask(low_zeros) & m;
                if self.max_u() == 0 {
                    return Self::constant(w, 0);
                }
                let hi_a = w - 1 - self.leading_zeros_u();
                let hi_res = hi_a + b_max;
                if hi_res + 1 < w {
                    zero |= (!mask(hi_res + 1)) & m;
                }
                Self {
                    width: w,
                    zero,
                    one: 0,
                }
            }
            Bin::LShr => {
                if self.max_u() == 0 {
                    return Self::constant(w, 0);
                }
                let hi_a = w - 1 - self.leading_zeros_u();
                if hi_a < b_min {
                    return Self::constant(w, 0);
                }
                let hi_res = hi_a - b_min;
                let zero = if hi_res + 1 < w {
                    (!mask(hi_res + 1)) & m
                } else {
                    0
                };
                Self {
                    width: w,
                    zero,
                    one: 0,
                }
            }
            Bin::AShr => {
                if self.sign_is_zero() {
                    return self.shift(Bin::LShr, b);
                }
                if self.sign_is_one() {
                    let top_ones = (b_min + 1).min(w);
                    let one = if top_ones == w {
                        m
                    } else {
                        (!mask(w - top_ones)) & m
                    };
                    return Self {
                        width: w,
                        zero: 0,
                        one,
                    };
                }
                Self::unknown(w)
            }
            _ => Self::unknown(w),
        }
    }

    pub(super) fn mul(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let w = self.width;
        let m = mask(w);
        if let (Some(x), Some(y)) = (self.const_val(), b.const_val()) {
            return Self::constant(w, x.wrapping_mul(y));
        }
        if self.max_u() == 0 || b.max_u() == 0 {
            return Self::constant(w, 0);
        }
        // Small-popcount constant multiplier: decompose as exact shifted sums.
        if let Some(k) = b.const_val()
            && k.count_ones() <= 4
        {
            return mul_by_small_const(self, k);
        }
        if let Some(k) = self.const_val()
            && k.count_ones() <= 4
        {
            return mul_by_small_const(b, k);
        }
        let tz = (self.trailing_zeros() + b.trailing_zeros()).min(w);
        let mut zero = mask(tz) & m;
        let mut one = 0u64;
        if tz == 0 && (self.one & b.one & 1) != 0 {
            one = 1;
        }
        let prod_max = u128::from(self.max_u()) * u128::from(b.max_u());
        if w < 64 && prod_max < (1u128 << w) {
            let max_u = prod_max as u64;
            let lz = max_u.leading_zeros() - (64 - w);
            if lz > 0 {
                let keep = w - lz;
                zero |= (!mask(keep)) & m;
            }
        }
        Self {
            width: w,
            zero: zero & !one & m,
            one: one & m,
        }
    }

    pub(super) fn udiv(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let w = self.width;
        if b.min_u() == 0 {
            return Self::unknown(w);
        }
        let min_q = self.min_u() / b.max_u();
        let max_q = self.max_u() / b.min_u();
        Self::from_unsigned_range(w, min_q, max_q)
    }

    pub(super) fn urem(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let w = self.width;
        if b.max_u() == 0 {
            return Self::unknown(w);
        }
        if let Some(k) = b.const_val() {
            if k == 0 {
                return Self::unknown(w);
            }
            if k.is_power_of_two() {
                return self.and(Self::constant(w, k - 1));
            }
            if let Some(x) = self.const_val() {
                return Self::constant(w, x % k);
            }
        }
        let max_r = self.max_u().min(b.max_u() - 1);
        let mut kb = Self::from_unsigned_range(w, 0, max_r);
        // If a < b is guaranteed, a % b == a.
        if self.max_u() < b.min_u() {
            kb = self;
        }
        kb
    }

    pub(super) fn sdiv(self, b: Self) -> Self {
        debug_assert_eq!(self.width, b.width);
        let w = self.width;
        if let (Some(x), Some(y)) = (self.const_val(), b.const_val()) {
            let (sx, sy) = (signed(w, x), signed(w, y));
            if sy == 0 || (sy == -1 && x == (1u64 << (w - 1))) {
                return Self::unknown(w);
            }
            return Self::constant(w, (sx / sy) as u64);
        }
        if self.sign_is_zero() && b.sign_is_zero() && b.min_u() > 0 {
            return self.udiv(b);
        }
        Self::unknown(w)
    }

    pub(super) fn zext(self, to: u32) -> Self {
        debug_assert!(to >= self.width && to <= 64);
        let m = mask(to);
        let hi_zero = (!mask(self.width)) & m;
        Self {
            width: to,
            zero: (self.zero | hi_zero) & m,
            one: self.one & m,
        }
    }

    pub(super) fn sext(self, to: u32) -> Self {
        debug_assert!(to >= self.width && to <= 64);
        let m = mask(to);
        let hi = (!mask(self.width)) & m;
        let hi_zero = if self.sign_is_zero() { hi } else { 0 };
        let hi_one = if self.sign_is_one() { hi } else { 0 };
        Self {
            width: to,
            zero: (self.zero | hi_zero) & m,
            one: (self.one | hi_one) & m,
        }
    }

    pub(super) fn trunc(self, to: u32) -> Self {
        debug_assert!(to >= 1 && to <= self.width);
        let m = mask(to);
        Self {
            width: to,
            zero: self.zero & m,
            one: self.one & m,
        }
    }

    /// Evaluates an integer comparison `p(self, b)` if its outcome is provably constant.
    pub(super) fn cmp(self, p: u8, b: Self) -> Option<bool> {
        debug_assert_eq!(self.width, b.width);
        match p {
            pred::EQ => {
                if let (Some(x), Some(y)) = (self.const_val(), b.const_val()) {
                    Some(x == y)
                } else if (self.zero & b.one) | (self.one & b.zero) != 0 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::NE => self.cmp(pred::EQ, b).map(|v| !v),
            pred::IULT => {
                if self.max_u() < b.min_u() {
                    Some(true)
                } else if self.min_u() >= b.max_u() {
                    Some(false)
                } else {
                    None
                }
            }
            pred::IULE => {
                if self.max_u() <= b.min_u() {
                    Some(true)
                } else if self.min_u() > b.max_u() {
                    Some(false)
                } else {
                    None
                }
            }
            pred::IUGT => b.cmp(pred::IULT, self),
            pred::IUGE => b.cmp(pred::IULE, self),
            pred::SLT => {
                if self.max_s() < b.min_s() {
                    Some(true)
                } else if self.min_s() >= b.max_s() {
                    Some(false)
                } else {
                    None
                }
            }
            pred::SLE => {
                if self.max_s() <= b.min_s() {
                    Some(true)
                } else if self.min_s() > b.max_s() {
                    Some(false)
                } else {
                    None
                }
            }
            pred::SGT => b.cmp(pred::SLT, self),
            pred::SGE => b.cmp(pred::SLE, self),
            _ => None,
        }
    }
}

fn mul_by_small_const(a: KnownBits, mut k: u64) -> KnownBits {
    let mut acc = KnownBits::constant(a.width, 0);
    while k != 0 {
        let sh = k.trailing_zeros();
        if sh < a.width {
            acc = acc.add_sub(a.shl_const(sh), false);
        }
        k &= k - 1;
    }
    acc
}

/// Sparse conditional lattice state per instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lattice {
    /// Not yet reached on any executable path.
    Top,
    /// Reachable integer value with sound `KnownBits`.
    Bits(KnownBits),
    /// Reachable non-integer value.
    Other,
}

impl Lattice {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Top, x) | (x, Self::Top) => x,
            (Self::Bits(a), Self::Bits(b)) if a.width == b.width => Self::Bits(a.join(b)),
            _ => Self::Other,
        }
    }
}

/// Canonical `(pred, a, b)` for comparison equivalence (IEEE-754 NaN safe).
fn canonical_cmp(p: u8, a: V, b: V) -> (u8, V, V) {
    match p {
        pred::EQ
        | pred::NE
        | pred::OEQ
        | pred::UEQ
        | pred::ONE
        | pred::UNE
        | pred::ORD
        | pred::UNO => {
            if b < a {
                (p, b, a)
            } else {
                (p, a, b)
            }
        }
        pred::IUGT => (pred::IULT, b, a),
        pred::IUGE => (pred::IULE, b, a),
        pred::SGT => (pred::SLT, b, a),
        pred::SGE => (pred::SLE, b, a),
        pred::OGT => (pred::OLT, b, a),
        pred::OGE => (pred::OLE, b, a),
        pred::UGT => (pred::ULT, b, a),
        pred::UGE => (pred::ULE, b, a),
        _ => (p, a, b),
    }
}

/// Flips operand order of an integer comparison predicate (`p(a, b) <=> swap_int_pred(p)(b, a)`).
fn swap_int_pred(p: u8) -> u8 {
    match p {
        pred::IULT => pred::IUGT,
        pred::IULE => pred::IUGE,
        pred::IUGT => pred::IULT,
        pred::IUGE => pred::IULE,
        pred::SLT => pred::SGT,
        pred::SLE => pred::SGE,
        pred::SGT => pred::SLT,
        pred::SGE => pred::SLE,
        other => other,
    }
}

fn dominates(idom: &[usize], a: usize, b: usize) -> bool {
    if a >= idom.len() || b >= idom.len() || idom[a] == usize::MAX || idom[b] == usize::MAX {
        return false;
    }
    let mut cur = b;
    loop {
        if cur == a {
            return true;
        }
        if cur == 0 {
            return false;
        }
        cur = idom[cur];
    }
}

/// Bitmask of the lowest `hi_bit(d)` bits, where `hi_bit(0) = 0`.
#[inline]
fn low_mask_up_to(d: u64) -> u64 {
    if d == 0 {
        0
    } else {
        mask(64 - d.leading_zeros())
    }
}

/// Output of whole-function Sparse Conditional Known-Bits Propagation.
struct SckpResult {
    exec_block: Vec<bool>,
    exec_edge: HashSet<(usize, usize)>,
    lat: Vec<Lattice>,
}

/// Scoped branch-correlated environment along the dominator tree.
#[derive(Default)]
struct PathEnv {
    bits: HashMap<V, bool>,
    compares: HashMap<(u8, V, V), bool>,
    kb: HashMap<V, KnownBits>,
    urange: HashMap<V, (u64, u64)>,
    srange: HashMap<V, (i64, i64)>,
    const_eq: HashMap<V, V>,
}

#[derive(Default)]
struct ScopeUndo {
    bits: Vec<(V, Option<bool>)>,
    compares: Vec<((u8, V, V), Option<bool>)>,
    kb: Vec<(V, Option<KnownBits>)>,
    urange: Vec<(V, Option<(u64, u64)>)>,
    srange: Vec<(V, Option<(i64, i64)>)>,
    const_eq: Vec<(V, Option<V>)>,
}

impl PathEnv {
    fn set_bit(&mut self, undo: &mut ScopeUndo, v: V, val: bool) -> bool {
        if self.bits.get(&v) == Some(&val) {
            return false;
        }
        let prev = self.bits.insert(v, val);
        undo.bits.push((v, prev));
        true
    }

    fn set_cmp(&mut self, undo: &mut ScopeUndo, key: (u8, V, V), val: bool) {
        if self.compares.get(&key) == Some(&val) {
            return;
        }
        let prev = self.compares.insert(key, val);
        undo.compares.push((key, prev));
    }

    fn set_const_eq(&mut self, undo: &mut ScopeUndo, v: V, k: V) {
        if self.const_eq.contains_key(&v) {
            return;
        }
        let prev = self.const_eq.insert(v, k);
        undo.const_eq.push((v, prev));
    }

    fn refine_kb(&mut self, undo: &mut ScopeUndo, v: V, base: KnownBits, add: KnownBits) {
        let cur = self.kb.get(&v).copied().unwrap_or(base);
        let Some(next) = cur.refine(add) else {
            return;
        };
        if next != cur {
            let prev = self.kb.insert(v, next);
            undo.kb.push((v, prev));
        }
    }

    fn refine_urange(
        &mut self,
        undo: &mut ScopeUndo,
        v: V,
        base_kb: KnownBits,
        lo: u64,
        hi: u64,
    ) {
        let cur = self
            .urange
            .get(&v)
            .copied()
            .unwrap_or((base_kb.min_u(), base_kb.max_u()));
        let nlo = cur.0.max(lo);
        let nhi = cur.1.min(hi);
        if nlo > nhi {
            return;
        }
        if (nlo, nhi) != cur {
            let prev = self.urange.insert(v, (nlo, nhi));
            undo.urange.push((v, prev));
        }
        let from_range = KnownBits::from_unsigned_range(base_kb.width, nlo, nhi);
        self.refine_kb(undo, v, base_kb, from_range);
    }

    fn refine_srange(
        &mut self,
        undo: &mut ScopeUndo,
        v: V,
        base_kb: KnownBits,
        lo: i64,
        hi: i64,
    ) {
        let cur = self
            .srange
            .get(&v)
            .copied()
            .unwrap_or((base_kb.min_s(), base_kb.max_s()));
        let nlo = cur.0.max(lo);
        let nhi = cur.1.min(hi);
        if nlo > nhi {
            return;
        }
        if (nlo, nhi) != cur {
            let prev = self.srange.insert(v, (nlo, nhi));
            undo.srange.push((v, prev));
        }
        let w = base_kb.width;
        let sign = 1u64 << (w - 1);
        if nlo >= 0 {
            self.refine_kb(
                undo,
                v,
                base_kb,
                KnownBits {
                    width: w,
                    zero: sign,
                    one: 0,
                },
            );
            self.refine_urange(undo, v, base_kb, nlo as u64, nhi as u64);
        } else if nhi < 0 {
            self.refine_kb(
                undo,
                v,
                base_kb,
                KnownBits {
                    width: w,
                    zero: 0,
                    one: sign,
                },
            );
            let m = mask(w);
            self.refine_urange(undo, v, base_kb, (nlo as u64) & m, (nhi as u64) & m);
        }
    }

    fn rollback(&mut self, undo: ScopeUndo) {
        for (v, prev) in undo.bits.into_iter().rev() {
            match prev {
                Some(x) => {
                    self.bits.insert(v, x);
                }
                None => {
                    self.bits.remove(&v);
                }
            }
        }
        for (k, prev) in undo.compares.into_iter().rev() {
            match prev {
                Some(x) => {
                    self.compares.insert(k, x);
                }
                None => {
                    self.compares.remove(&k);
                }
            }
        }
        for (v, prev) in undo.kb.into_iter().rev() {
            match prev {
                Some(x) => {
                    self.kb.insert(v, x);
                }
                None => {
                    self.kb.remove(&v);
                }
            }
        }
        for (v, prev) in undo.urange.into_iter().rev() {
            match prev {
                Some(x) => {
                    self.urange.insert(v, x);
                }
                None => {
                    self.urange.remove(&v);
                }
            }
        }
        for (v, prev) in undo.srange.into_iter().rev() {
            match prev {
                Some(x) => {
                    self.srange.insert(v, x);
                }
                None => {
                    self.srange.remove(&v);
                }
            }
        }
        for (v, prev) in undo.const_eq.into_iter().rev() {
            match prev {
                Some(x) => {
                    self.const_eq.insert(v, x);
                }
                None => {
                    self.const_eq.remove(&v);
                }
            }
        }
    }
}

impl Module {
    /// Whole-function `exec`-mask, known-bits, and bitfield dataflow pass.
    /// Returns whether any instruction, phi, or terminator changed.
    pub(super) fn optimise_dataflow(&mut self) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let mut changed = false;
        changed |= self.dataflow_forward();
        changed |= self.dataflow_demanded_bits();
        changed
    }

    fn is_undef(&self, v: V) -> bool {
        matches!(v, V::Const(c) if matches!(self.consts[c as usize].1, Const::Undef))
    }

    fn int_width_of(&self, v: V) -> Option<u32> {
        match self.types[self.type_of(v).0 as usize] {
            TypeDef::Int(bits) if (1..=64).contains(&bits) => Some(bits),
            _ => None,
        }
    }

    fn ty_int_bits(&self, ty: Ty) -> Option<u32> {
        match self.types[ty.0 as usize] {
            TypeDef::Int(bits) if (1..=64).contains(&bits) => Some(bits),
            _ => None,
        }
    }

    fn const_int(&mut self, ty: Ty, bits: u32, val: u64) -> V {
        self.int_const(ty, signed(bits, val & mask(bits)))
    }

    fn is_bool_not_v(&self, v: V) -> Option<V> {
        let V::Inst(j) = v else { return None };
        let inst = &self.insts[j as usize];
        if self.ty_int_bits(inst.ty) != Some(1) {
            return None;
        }
        let Op::Bin(Bin::Xor, a, b) = inst.op else {
            return None;
        };
        match (self.int_k(a), self.int_k(b)) {
            (None, Some((1, 1))) => Some(a),
            (Some((1, 1)), None) => Some(b),
            _ => None,
        }
    }

    /// Evaluates a value `v` to `Lattice` using current `lat`.
    fn eval_lattice(&self, lat: &[Lattice], v: V) -> Lattice {
        match v {
            V::Inst(i) => lat[i as usize],
            V::Const(c) => {
                let (ty, val) = &self.consts[c as usize];
                match (val, self.ty_int_bits(*ty)) {
                    (Const::Int(x), Some(w)) => Lattice::Bits(KnownBits::constant(w, *x as u64)),
                    (Const::Undef, Some(w)) => Lattice::Bits(KnownBits::unknown(w)),
                    _ => Lattice::Other,
                }
            }
            _ => Lattice::Other,
        }
    }

    fn eval_kb(&self, lat: &[Lattice], v: V) -> Option<KnownBits> {
        match self.eval_lattice(lat, v) {
            Lattice::Bits(kb) => Some(kb),
            _ => None,
        }
    }

    /// Evaluates a value `v` to `Lattice` using current `lat`, optionally
    /// sharpened by static branch-dominated `block_kb[at_blk]`.
    fn eval_lattice_at(
        &self,
        lat: &[Lattice],
        block_kb: &[HashMap<V, KnownBits>],
        at_blk: usize,
        v: V,
    ) -> Lattice {
        let base = self.eval_lattice(lat, v);
        match base {
            Lattice::Bits(kb) => {
                if let Some(&refined) = block_kb[at_blk].get(&v)
                    && refined.width == kb.width
                    && let Some(r) = kb.refine(refined)
                {
                    Lattice::Bits(r)
                } else {
                    base
                }
            }
            _ => base,
        }
    }

    /// Precomputes static branch-implied `KnownBits` refinements at each block
    /// by walking the dominator tree once from entry.
    fn compute_block_kb(&self, tree: &Tree) -> Vec<HashMap<V, KnownBits>> {
        let nb = self.blocks.len();
        let mut out: Vec<HashMap<V, KnownBits>> = vec![HashMap::new(); nb];
        let unknown_lat: Vec<Lattice> = self
            .insts
            .iter()
            .map(|inst| match self.ty_int_bits(inst.ty) {
                Some(w) => Lattice::Bits(KnownBits::unknown(w)),
                None => Lattice::Other,
            })
            .collect();
        let empty_repl: Vec<Option<V>> = vec![None; self.insts.len()];
        let mut env = PathEnv::default();
        enum Step {
            Enter(usize),
            Leave(ScopeUndo),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(undo) => {
                    env.rollback(undo);
                    continue;
                }
            };
            let mut undo = ScopeUndo::default();
            if let [p] = tree.preds[b][..]
                && let Some(Term::CondBr(c, t, f)) = self.blocks[p].term
                && t != f
            {
                self.learn_path(&empty_repl, &unknown_lat, c, b == t, &mut env, &mut undo, 8);
            }
            if !env.kb.is_empty() || !env.bits.is_empty() {
                let mut snap = env.kb.clone();
                for (&v, &bit) in &env.bits {
                    snap.insert(v, KnownBits::constant(1, u64::from(bit)));
                }
                out[b] = snap;
            }
            steps.push(Step::Leave(undo));
            for &child in tree.children[b].iter().rev() {
                steps.push(Step::Enter(child));
            }
        }
        out
    }

    /// Evaluates instruction `i` under current SCKP lattice state `lat` and
    /// edge-sensitive `block_kb`.
    fn transfer_inst(
        &self,
        lat: &[Lattice],
        block_kb: &[HashMap<V, KnownBits>],
        exec_edge: &HashSet<(usize, usize)>,
        blk: usize,
        i: u32,
    ) -> Lattice {
        let inst = &self.insts[i as usize];
        let Some(w) = self.ty_int_bits(inst.ty) else {
            return Lattice::Other;
        };
        match &inst.op {
            Op::Phi(incoming) => {
                let mut acc = Lattice::Top;
                for &(v, pred_b) in incoming {
                    if !exec_edge.contains(&(pred_b, blk)) {
                        continue;
                    }
                    acc = acc.join(self.eval_lattice_at(lat, block_kb, pred_b, v));
                }
                acc
            }
            Op::Bin(op, a, b) => {
                let (la, lb) = (
                    self.eval_lattice_at(lat, block_kb, blk, *a),
                    self.eval_lattice_at(lat, block_kb, blk, *b),
                );
                let (Lattice::Bits(ka), Lattice::Bits(kb)) = (la, lb) else {
                    return if matches!(la, Lattice::Top) || matches!(lb, Lattice::Top) {
                        Lattice::Top
                    } else {
                        Lattice::Bits(KnownBits::unknown(w))
                    };
                };
                if *a == *b && !self.is_undef(*a) {
                    match op {
                        Bin::And | Bin::Or => return Lattice::Bits(ka),
                        Bin::Sub | Bin::Xor => return Lattice::Bits(KnownBits::constant(w, 0)),
                        _ => {}
                    }
                }
                // Exact bitmask pattern `sub(shl(1, sh), 1)` (emitted by `bfe` / `s_bfm_b32`):
                // when `sh.max_u() < w`, `(1 << sh) - 1` has `sh` trailing ones and never underflows.
                if *op == Bin::Sub
                    && kb.const_val() == Some(1)
                    && let V::Inst(sh_i) = *a
                    && let Op::Bin(Bin::Shl, sh_base, sh_amt) = self.insts[sh_i as usize].op
                    && self
                        .eval_lattice_at(lat, block_kb, blk, sh_base)
                        == Lattice::Bits(KnownBits::constant(w, 1))
                    && let Lattice::Bits(ks) = self.eval_lattice_at(lat, block_kb, blk, sh_amt)
                    && ks.max_u() < u64::from(w)
                {
                    let m = mask(w);
                    let lo_ones = mask(ks.min_u() as u32) & m;
                    let hi_zeros = (!mask(ks.max_u() as u32)) & m;
                    return Lattice::Bits(KnownBits {
                        width: w,
                        zero: hi_zeros,
                        one: lo_ones,
                    });
                }
                let res = match op {
                    Bin::And => ka.and(kb),
                    Bin::Or => ka.or(kb),
                    Bin::Xor => ka.xor(kb),
                    Bin::Add => ka.add_sub(kb, false),
                    Bin::Sub => ka.add_sub(kb, true),
                    Bin::Shl | Bin::LShr | Bin::AShr => ka.shift(*op, kb),
                    Bin::Mul => ka.mul(kb),
                    Bin::UDiv => ka.udiv(kb),
                    Bin::URem => ka.urem(kb),
                    Bin::SDiv => ka.sdiv(kb),
                };
                Lattice::Bits(res)
            }
            Op::Cast(op, x) => match op {
                Cast::ZExt => match self.eval_lattice_at(lat, block_kb, blk, *x) {
                    Lattice::Top => Lattice::Top,
                    Lattice::Bits(kx) => Lattice::Bits(kx.zext(w)),
                    Lattice::Other => Lattice::Bits(KnownBits::unknown(w)),
                },
                Cast::SExt => match self.eval_lattice_at(lat, block_kb, blk, *x) {
                    Lattice::Top => Lattice::Top,
                    Lattice::Bits(kx) => Lattice::Bits(kx.sext(w)),
                    Lattice::Other => Lattice::Bits(KnownBits::unknown(w)),
                },
                Cast::Trunc => match self.eval_lattice_at(lat, block_kb, blk, *x) {
                    Lattice::Top => Lattice::Top,
                    Lattice::Bits(kx) => Lattice::Bits(kx.trunc(w)),
                    Lattice::Other => Lattice::Bits(KnownBits::unknown(w)),
                },
                Cast::Bitcast => {
                    if let V::Const(c) = *x
                        && let Const::Float(bits) = self.consts[c as usize].1
                        && w == 32
                    {
                        return Lattice::Bits(KnownBits::constant(32, u64::from(bits)));
                    }
                    if let V::Inst(j) = *x
                        && let Op::Cast(Cast::Bitcast, inner) = self.insts[j as usize].op
                        && self.type_of(inner) == inst.ty
                    {
                        return self.eval_lattice_at(lat, block_kb, blk, inner);
                    }
                    Lattice::Bits(KnownBits::unknown(w))
                }
                _ => Lattice::Bits(KnownBits::unknown(w)),
            },
            Op::Select(c, t, f) => match self.eval_lattice_at(lat, block_kb, blk, *c) {
                Lattice::Top => Lattice::Top,
                Lattice::Bits(kc) => match kc.const_val() {
                    Some(1) => self.eval_lattice_at(lat, block_kb, blk, *t),
                    Some(0) => self.eval_lattice_at(lat, block_kb, blk, *f),
                    _ => self
                        .eval_lattice_at(lat, block_kb, blk, *t)
                        .join(self.eval_lattice_at(lat, block_kb, blk, *f)),
                },
                Lattice::Other => Lattice::Bits(KnownBits::unknown(w)),
            },
            Op::Cmp(p, a, b) => {
                if *p <= pred::TRUE {
                    if *p == pred::FALSE {
                        return Lattice::Bits(KnownBits::constant(1, 0));
                    }
                    if *p == pred::TRUE {
                        return Lattice::Bits(KnownBits::constant(1, 1));
                    }
                    if let (Some(x), Some(y)) = (self.float_k(*a), self.float_k(*b)) {
                        let holds = if x.is_nan() || y.is_nan() {
                            p & 8 != 0
                        } else {
                            (p & 1 != 0 && x == y)
                                || (p & 2 != 0 && x > y)
                                || (p & 4 != 0 && x < y)
                        };
                        return Lattice::Bits(KnownBits::constant(1, u64::from(holds)));
                    }
                    return Lattice::Bits(KnownBits::unknown(1));
                }
                if !(pred::EQ..=pred::SLE).contains(p) {
                    return Lattice::Bits(KnownBits::unknown(1));
                }
                if *a == *b && !self.is_undef(*a) {
                    let holds = matches!(
                        *p,
                        pred::EQ | pred::IUGE | pred::IULE | pred::SGE | pred::SLE
                    );
                    return Lattice::Bits(KnownBits::constant(1, u64::from(holds)));
                }
                let (la, lb) = (
                    self.eval_lattice_at(lat, block_kb, blk, *a),
                    self.eval_lattice_at(lat, block_kb, blk, *b),
                );
                match (la, lb) {
                    (Lattice::Top, _) | (_, Lattice::Top) => Lattice::Top,
                    (Lattice::Bits(ka), Lattice::Bits(kb)) => match ka.cmp(*p, kb) {
                        Some(holds) => Lattice::Bits(KnownBits::constant(1, u64::from(holds))),
                        None => Lattice::Bits(KnownBits::unknown(1)),
                    },
                    _ => Lattice::Bits(KnownBits::unknown(1)),
                }
            }
            Op::Call(f, args) => self.transfer_pure_call(lat, block_kb, blk, *f, args, w),
            // Loads (including LDS space-3), atomics, extracts: always unknown.
            _ => Lattice::Bits(KnownBits::unknown(w)),
        }
    }

    /// Sound KnownBits transfer for pure DXIL integer/bit/clamp intrinsics.
    /// Never inspects or coalesces `dx.op.getDimensions`.
    fn transfer_pure_call(
        &self,
        lat: &[Lattice],
        block_kb: &[HashMap<V, KnownBits>],
        blk: usize,
        f: u32,
        args: &[V],
        w: u32,
    ) -> Lattice {
        let name = self.funcs[f as usize].name.as_str();
        let name = name.strip_prefix("dx.op.").unwrap_or(name);
        if name.starts_with("getDimensions") || w != 32 {
            return Lattice::Bits(KnownBits::unknown(w));
        }
        let Some((_, opcode)) = args.first().and_then(|&v| self.int_k(v)) else {
            return Lattice::Bits(KnownBits::unknown(w));
        };
        let eval_arg = |v: V| self.eval_lattice_at(lat, block_kb, blk, v);
        match (name, opcode, &args[1..]) {
            // `dx.op.legacyF32ToF16` (130): f16 in low 16 bits, upper 16 bits guaranteed 0.
            ("legacyF32ToF16", 130, [_]) => Lattice::Bits(KnownBits {
                width: 32,
                zero: 0xFFFF_0000,
                one: 0,
            }),
            // `dx.op.unary.i32(30, a)` = Bfrev (bit reverse).
            ("unary.i32", 30, [a]) => match eval_arg(*a) {
                Lattice::Top => Lattice::Top,
                Lattice::Bits(ka) => Lattice::Bits(KnownBits {
                    width: 32,
                    zero: u64::from((ka.zero as u32).reverse_bits()),
                    one: u64::from((ka.one as u32).reverse_bits()),
                }),
                Lattice::Other => Lattice::Bits(KnownBits::unknown(32)),
            },
            // `dx.op.unaryBits.i32`: 31 = Countbits, 32 = FirstbitLo, 33 = FirstbitHi (u32), 34 = FirstbitHi (i32).
            ("unaryBits.i32", 31, [a]) => match eval_arg(*a) {
                Lattice::Top => Lattice::Top,
                Lattice::Bits(ka) => {
                    let min_pop = u64::from(ka.one.count_ones());
                    let max_pop = u64::from(32 - ka.zero.count_ones());
                    Lattice::Bits(KnownBits::from_unsigned_range(32, min_pop, max_pop))
                }
                Lattice::Other => Lattice::Bits(KnownBits::from_unsigned_range(32, 0, 32)),
            },
            ("unaryBits.i32", 32, [a]) => match eval_arg(*a) {
                Lattice::Top => Lattice::Top,
                Lattice::Bits(ka) => {
                    if let Some(x) = ka.const_val() {
                        let r = if x == 0 {
                            u32::MAX
                        } else {
                            (x as u32).trailing_zeros()
                        };
                        Lattice::Bits(KnownBits::constant(32, u64::from(r)))
                    } else if ka.one != 0 {
                        let min_tz = u64::from(ka.trailing_zeros());
                        let max_tz = u64::from((ka.one as u32).trailing_zeros());
                        Lattice::Bits(KnownBits::from_unsigned_range(32, min_tz, max_tz))
                    } else {
                        Lattice::Bits(KnownBits::unknown(32))
                    }
                }
                Lattice::Other => Lattice::Bits(KnownBits::unknown(32)),
            },
            ("unaryBits.i32", 33, [a]) => match eval_arg(*a) {
                Lattice::Top => Lattice::Top,
                Lattice::Bits(ka) => {
                    if let Some(x) = ka.const_val() {
                        let r = if x == 0 {
                            u32::MAX
                        } else {
                            (x as u32).leading_zeros()
                        };
                        Lattice::Bits(KnownBits::constant(32, u64::from(r)))
                    } else if ka.one != 0 {
                        let min_lz = u64::from((ka.max_u() as u32).leading_zeros());
                        let max_lz = u64::from((ka.one as u32).leading_zeros());
                        Lattice::Bits(KnownBits::from_unsigned_range(32, min_lz, max_lz))
                    } else {
                        Lattice::Bits(KnownBits::unknown(32))
                    }
                }
                Lattice::Other => Lattice::Bits(KnownBits::unknown(32)),
            },
            ("unaryBits.i32", 34, [a]) => match eval_arg(*a) {
                Lattice::Top => Lattice::Top,
                Lattice::Bits(ka) => {
                    if let Some(x) = ka.const_val() {
                        let x = x as u32;
                        let r = if x == 0 || x == u32::MAX {
                            u32::MAX
                        } else if (x as i32) < 0 {
                            (!x).leading_zeros()
                        } else {
                            x.leading_zeros()
                        };
                        Lattice::Bits(KnownBits::constant(32, u64::from(r)))
                    } else {
                        Lattice::Bits(KnownBits::unknown(32))
                    }
                }
                Lattice::Other => Lattice::Bits(KnownBits::unknown(32)),
            },
            // `dx.op.binary.i32`: 37 = IMax, 38 = IMin, 39 = UMax, 40 = UMin.
            ("binary.i32", 37..=40, [a, b]) => {
                let (la, lb) = (eval_arg(*a), eval_arg(*b));
                let (Lattice::Bits(ka), Lattice::Bits(kb)) = (la, lb) else {
                    return if matches!(la, Lattice::Top) || matches!(lb, Lattice::Top) {
                        Lattice::Top
                    } else {
                        Lattice::Bits(KnownBits::unknown(32))
                    };
                };
                let kb_res = match opcode {
                    40 => {
                        // UMin(a, b)
                        if ka.max_u() <= kb.min_u() {
                            ka
                        } else if kb.max_u() <= ka.min_u() {
                            kb
                        } else {
                            let lo = ka.min_u().min(kb.min_u());
                            let hi = ka.max_u().min(kb.max_u());
                            KnownBits::from_unsigned_range(32, lo, hi)
                        }
                    }
                    39 => {
                        // UMax(a, b)
                        if ka.min_u() >= kb.max_u() {
                            ka
                        } else if kb.min_u() >= ka.max_u() {
                            kb
                        } else {
                            let lo = ka.min_u().max(kb.min_u());
                            let hi = ka.max_u().max(kb.max_u());
                            KnownBits::from_unsigned_range(32, lo, hi)
                        }
                    }
                    38 => {
                        // IMin(a, b)
                        if ka.max_s() <= kb.min_s() {
                            ka
                        } else if kb.max_s() <= ka.min_s() {
                            kb
                        } else if ka.sign_is_zero() && kb.sign_is_zero() {
                            KnownBits::from_unsigned_range(
                                32,
                                ka.min_u().min(kb.min_u()),
                                ka.max_u().min(kb.max_u()),
                            )
                        } else {
                            ka.join(kb)
                        }
                    }
                    37 => {
                        // IMax(a, b)
                        if ka.min_s() >= kb.max_s() {
                            ka
                        } else if kb.min_s() >= ka.max_s() {
                            kb
                        } else if ka.sign_is_zero() && kb.sign_is_zero() {
                            KnownBits::from_unsigned_range(
                                32,
                                ka.min_u().max(kb.min_u()),
                                ka.max_u().max(kb.max_u()),
                            )
                        } else {
                            ka.join(kb)
                        }
                    }
                    _ => KnownBits::unknown(32),
                };
                Lattice::Bits(kb_res)
            }
            _ => Lattice::Bits(KnownBits::unknown(w)),
        }
    }

    /// Runs whole-function Sparse Conditional Known-Bits Propagation.
    fn run_sckp(&self) -> SckpResult {
        let nb = self.blocks.len();
        let ni = self.insts.len();
        let tree = self.dominator_tree();
        let block_kb = self.compute_block_kb(&tree);
        let mut block_of = vec![usize::MAX; ni];
        let mut users: Vec<Vec<u32>> = vec![Vec::new(); ni];
        let mut term_users: Vec<Vec<usize>> = vec![Vec::new(); ni];
        for (b, blk) in self.blocks.iter().enumerate() {
            for &i in blk.phis.iter().chain(&blk.body) {
                block_of[i as usize] = b;
                for_each_operand(&self.insts[i as usize].op, |v| {
                    if let V::Inst(src) = v {
                        users[src as usize].push(i);
                    }
                });
                // Register transitive 2-hop SSA dependencies inspected by
                // `transfer_inst` so widening a 2-hop source re-enqueues `i`
                // even when the intermediate instruction's lattice is unchanged.
                match self.insts[i as usize].op {
                    Op::Bin(Bin::Sub, V::Inst(sh_i), _) => {
                        if let Op::Bin(Bin::Shl, sh_base, sh_amt) = self.insts[sh_i as usize].op {
                            if let V::Inst(src) = sh_base {
                                users[src as usize].push(i);
                            }
                            if let V::Inst(src) = sh_amt {
                                users[src as usize].push(i);
                            }
                        }
                    }
                    Op::Cast(Cast::Bitcast, V::Inst(j)) => {
                        if let Op::Cast(Cast::Bitcast, V::Inst(inner)) = self.insts[j as usize].op {
                            users[inner as usize].push(i);
                        }
                    }
                    _ => {}
                }
            }
            if let Some(Term::CondBr(V::Inst(c), ..)) = blk.term {
                term_users[c as usize].push(b);
            }
        }
        for u in &mut users {
            u.sort_unstable();
            u.dedup();
        }
        for u in &mut term_users {
            u.sort_unstable();
            u.dedup();
        }

        let mut exec_block = vec![false; nb];
        let mut exec_edge: HashSet<(usize, usize)> = HashSet::new();
        let mut lat = vec![Lattice::Top; ni];
        let mut cfg_work: VecDeque<usize> = VecDeque::new();
        let mut ssa_work: VecDeque<u32> = VecDeque::new();
        let mut in_ssa = vec![false; ni];

        let enqueue_inst = |i: u32, ssa_work: &mut VecDeque<u32>, in_ssa: &mut [bool]| {
            if !in_ssa[i as usize] {
                in_ssa[i as usize] = true;
                ssa_work.push_back(i);
            }
        };

        let mark_edge = |from: usize,
                         to: usize,
                         exec_block: &mut [bool],
                         exec_edge: &mut HashSet<(usize, usize)>,
                         cfg_work: &mut VecDeque<usize>,
                         ssa_work: &mut VecDeque<u32>,
                         in_ssa: &mut [bool],
                         blocks: &[super::Block]| {
            if exec_edge.insert((from, to)) {
                if !exec_block[to] {
                    exec_block[to] = true;
                    cfg_work.push_back(to);
                } else {
                    for &p in &blocks[to].phis {
                        enqueue_inst(p, ssa_work, in_ssa);
                    }
                }
            }
        };

        exec_block[0] = true;
        cfg_work.push_back(0);

        let eval_term = |b: usize,
                         lat: &[Lattice],
                         exec_block: &mut [bool],
                         exec_edge: &mut HashSet<(usize, usize)>,
                         cfg_work: &mut VecDeque<usize>,
                         ssa_work: &mut VecDeque<u32>,
                         in_ssa: &mut [bool]| {
            match self.blocks[b].term {
                Some(Term::Br(t)) => mark_edge(
                    b,
                    t,
                    exec_block,
                    exec_edge,
                    cfg_work,
                    ssa_work,
                    in_ssa,
                    &self.blocks,
                ),
                Some(Term::CondBr(c, t, f)) => {
                    if t == f {
                        mark_edge(
                            b,
                            t,
                            exec_block,
                            exec_edge,
                            cfg_work,
                            ssa_work,
                            in_ssa,
                            &self.blocks,
                        );
                        return;
                    }
                    match self.eval_lattice_at(lat, &block_kb, b, c) {
                        Lattice::Top => {}
                        Lattice::Bits(kc) => match kc.const_val() {
                            Some(1) => mark_edge(
                                b,
                                t,
                                exec_block,
                                exec_edge,
                                cfg_work,
                                ssa_work,
                                in_ssa,
                                &self.blocks,
                            ),
                            Some(0) => mark_edge(
                                b,
                                f,
                                exec_block,
                                exec_edge,
                                cfg_work,
                                ssa_work,
                                in_ssa,
                                &self.blocks,
                            ),
                            _ => {
                                mark_edge(
                                    b,
                                    t,
                                    exec_block,
                                    exec_edge,
                                    cfg_work,
                                    ssa_work,
                                    in_ssa,
                                    &self.blocks,
                                );
                                mark_edge(
                                    b,
                                    f,
                                    exec_block,
                                    exec_edge,
                                    cfg_work,
                                    ssa_work,
                                    in_ssa,
                                    &self.blocks,
                                );
                            }
                        },
                        Lattice::Other => {
                            mark_edge(
                                b,
                                t,
                                exec_block,
                                exec_edge,
                                cfg_work,
                                ssa_work,
                                in_ssa,
                                &self.blocks,
                            );
                            mark_edge(
                                b,
                                f,
                                exec_block,
                                exec_edge,
                                cfg_work,
                                ssa_work,
                                in_ssa,
                                &self.blocks,
                            );
                        }
                    }
                }
                _ => {}
            }
        };

        while !cfg_work.is_empty() || !ssa_work.is_empty() {
            while let Some(b) = cfg_work.pop_front() {
                for &i in self.blocks[b].phis.iter().chain(&self.blocks[b].body) {
                    let next = self.transfer_inst(&lat, &block_kb, &exec_edge, b, i);
                    let merged = lat[i as usize].join(next);
                    if merged != lat[i as usize] {
                        lat[i as usize] = merged;
                        for &u in &users[i as usize] {
                            let ub = block_of[u as usize];
                            if ub != usize::MAX && exec_block[ub] {
                                enqueue_inst(u, &mut ssa_work, &mut in_ssa);
                            }
                        }
                        for &tb in &term_users[i as usize] {
                            if exec_block[tb] {
                                eval_term(
                                    tb,
                                    &lat,
                                    &mut exec_block,
                                    &mut exec_edge,
                                    &mut cfg_work,
                                    &mut ssa_work,
                                    &mut in_ssa,
                                );
                            }
                        }
                    }
                }
                eval_term(
                    b,
                    &lat,
                    &mut exec_block,
                    &mut exec_edge,
                    &mut cfg_work,
                    &mut ssa_work,
                    &mut in_ssa,
                );
            }
            while let Some(i) = ssa_work.pop_front() {
                in_ssa[i as usize] = false;
                let b = block_of[i as usize];
                if b == usize::MAX || !exec_block[b] {
                    continue;
                }
                let next = self.transfer_inst(&lat, &block_kb, &exec_edge, b, i);
                let merged = lat[i as usize].join(next);
                if merged != lat[i as usize] {
                    lat[i as usize] = merged;
                    for &u in &users[i as usize] {
                        let ub = block_of[u as usize];
                        if ub != usize::MAX && exec_block[ub] {
                            enqueue_inst(u, &mut ssa_work, &mut in_ssa);
                        }
                    }
                    for &tb in &term_users[i as usize] {
                        if exec_block[tb] {
                            eval_term(
                                tb,
                                &lat,
                                &mut exec_block,
                                &mut exec_edge,
                                &mut cfg_work,
                                &mut ssa_work,
                                &mut in_ssa,
                            );
                        }
                    }
                }
            }
        }

        SckpResult {
            exec_block,
            exec_edge,
            lat,
        }
    }

    fn effective_kb(&self, lat: &[Lattice], env: &PathEnv, v: V) -> Option<KnownBits> {
        let base = self.eval_kb(lat, v)?;
        Some(match env.kb.get(&v).copied() {
            Some(refined) => base.refine(refined).unwrap_or(base),
            None => base,
        })
    }

    fn effective_urange(&self, lat: &[Lattice], env: &PathEnv, v: V) -> Option<(u64, u64)> {
        let kb = self.effective_kb(lat, env, v)?;
        let (mut lo, mut hi) = (kb.min_u(), kb.max_u());
        if let Some(&(rlo, rhi)) = env.urange.get(&v) {
            let nlo = lo.max(rlo);
            let nhi = hi.min(rhi);
            if nlo <= nhi {
                (lo, hi) = (nlo, nhi);
            }
        }
        Some((lo, hi))
    }

    fn effective_srange(&self, lat: &[Lattice], env: &PathEnv, v: V) -> Option<(i64, i64)> {
        let kb = self.effective_kb(lat, env, v)?;
        let (mut lo, mut hi) = (kb.min_s(), kb.max_s());
        if let Some(&(rlo, rhi)) = env.srange.get(&v) {
            let nlo = lo.max(rlo);
            let nhi = hi.min(rhi);
            if nlo <= nhi {
                (lo, hi) = (nlo, nhi);
            }
        }
        Some((lo, hi))
    }

    /// Evaluates integer comparison `p(a, b)` using both `KnownBits` and scoped ranges.
    fn eval_int_cmp(&self, lat: &[Lattice], env: &PathEnv, p: u8, a: V, b: V) -> Option<bool> {
        if !(pred::EQ..=pred::SLE).contains(&p) || self.is_undef(a) || self.is_undef(b) {
            return None;
        }
        if a == b {
            return Some(matches!(
                p,
                pred::EQ | pred::IUGE | pred::IULE | pred::SGE | pred::SLE
            ));
        }
        let (ka, kb) = (
            self.effective_kb(lat, env, a)?,
            self.effective_kb(lat, env, b)?,
        );
        if let Some(res) = ka.cmp(p, kb) {
            return Some(res);
        }
        let (ua, ub) = (
            self.effective_urange(lat, env, a)?,
            self.effective_urange(lat, env, b)?,
        );
        let (sa, sb) = (
            self.effective_srange(lat, env, a)?,
            self.effective_srange(lat, env, b)?,
        );
        match p {
            pred::EQ => {
                if ua.1 < ub.0 || ub.1 < ua.0 || sa.1 < sb.0 || sb.1 < sa.0 {
                    Some(false)
                } else if ua.0 == ua.1 && ub.0 == ub.1 {
                    Some(ua.0 == ub.0)
                } else {
                    None
                }
            }
            pred::NE => self.eval_int_cmp(lat, env, pred::EQ, a, b).map(|v| !v),
            pred::IULT => {
                if ua.1 < ub.0 {
                    Some(true)
                } else if ua.0 >= ub.1 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::IULE => {
                if ua.1 <= ub.0 {
                    Some(true)
                } else if ua.0 > ub.1 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::IUGT => {
                if ua.0 > ub.1 {
                    Some(true)
                } else if ua.1 <= ub.0 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::IUGE => {
                if ua.0 >= ub.1 {
                    Some(true)
                } else if ua.1 < ub.0 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::SLT => {
                if sa.1 < sb.0 {
                    Some(true)
                } else if sa.0 >= sb.1 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::SLE => {
                if sa.1 <= sb.0 {
                    Some(true)
                } else if sa.0 > sb.1 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::SGT => {
                if sa.0 > sb.1 {
                    Some(true)
                } else if sa.1 <= sb.0 {
                    Some(false)
                } else {
                    None
                }
            }
            pred::SGE => {
                if sa.0 >= sb.1 {
                    Some(true)
                } else if sa.1 < sb.0 {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Learns boolean, comparison, bit-mask, equality, and range facts from `v == holds`.
    #[allow(clippy::too_many_arguments)]
    fn learn_path(
        &self,
        repl: &[Option<V>],
        lat: &[Lattice],
        v: V,
        holds: bool,
        env: &mut PathEnv,
        undo: &mut ScopeUndo,
        depth: u32,
    ) {
        let v = resolve(repl, v);
        let V::Inst(i) = v else { return };
        if !env.set_bit(undo, v, holds) || depth == 0 {
            return;
        }
        let known_bool = |env: &PathEnv, x: V| -> Option<bool> {
            let x = resolve(repl, x);
            if let Some((1, k)) = self.int_k(x) {
                return Some(k != 0);
            }
            env.bits.get(&x).copied()
        };
        match self.insts[i as usize].op {
            Op::Bin(Bin::And, a, b) => {
                if holds {
                    self.learn_path(repl, lat, a, true, env, undo, depth - 1);
                    self.learn_path(repl, lat, b, true, env, undo, depth - 1);
                } else {
                    if known_bool(env, a) == Some(true) {
                        self.learn_path(repl, lat, b, false, env, undo, depth - 1);
                    }
                    if known_bool(env, b) == Some(true) {
                        self.learn_path(repl, lat, a, false, env, undo, depth - 1);
                    }
                }
            }
            Op::Bin(Bin::Or, a, b) => {
                if !holds {
                    self.learn_path(repl, lat, a, false, env, undo, depth - 1);
                    self.learn_path(repl, lat, b, false, env, undo, depth - 1);
                } else {
                    if known_bool(env, a) == Some(false) {
                        self.learn_path(repl, lat, b, true, env, undo, depth - 1);
                    }
                    if known_bool(env, b) == Some(false) {
                        self.learn_path(repl, lat, a, true, env, undo, depth - 1);
                    }
                }
            }
            Op::Bin(Bin::Xor, a, b) => {
                if let Some(ka) = known_bool(env, a) {
                    self.learn_path(repl, lat, b, holds != ka, env, undo, depth - 1);
                } else if let Some(kb) = known_bool(env, b) {
                    self.learn_path(repl, lat, a, holds != kb, env, undo, depth - 1);
                }
            }
            Op::Select(s, t, f) => {
                if let Some(ks) = known_bool(env, s) {
                    if ks {
                        self.learn_path(repl, lat, t, holds, env, undo, depth - 1);
                    } else {
                        self.learn_path(repl, lat, f, holds, env, undo, depth - 1);
                    }
                } else if known_bool(env, t) == Some(!holds) {
                    self.learn_path(repl, lat, s, false, env, undo, depth - 1);
                    self.learn_path(repl, lat, f, holds, env, undo, depth - 1);
                } else if known_bool(env, f) == Some(!holds) {
                    self.learn_path(repl, lat, s, true, env, undo, depth - 1);
                    self.learn_path(repl, lat, t, holds, env, undo, depth - 1);
                }
            }
            Op::Cmp(p, a, b) => {
                let (a, b) = (resolve(repl, a), resolve(repl, b));
                env.set_cmp(undo, canonical_cmp(p, a, b), holds);
                env.set_cmp(undo, canonical_cmp(opposite(p), a, b), !holds);
                if !(pred::EQ..=pred::SLE).contains(&p) || self.is_undef(a) || self.is_undef(b) {
                    return;
                }
                let eff_p = if holds { p } else { opposite(p) };
                let (x, eff_p, kv, w, k) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((w, k))) => (a, eff_p, b, w, k),
                    (Some((w, k)), None) => (b, swap_int_pred(eff_p), a, w, k),
                    _ => return,
                };
                let Some(base_kb) = self.effective_kb(lat, env, x) else {
                    return;
                };
                let all = mask(w);
                match eff_p {
                    pred::EQ => {
                        env.set_const_eq(undo, x, kv);
                        let ckb = KnownBits::constant(w, k);
                        env.refine_kb(undo, x, base_kb, ckb);
                        env.refine_urange(undo, x, base_kb, k, k);
                        let sk = signed(w, k);
                        env.refine_srange(undo, x, base_kb, sk, sk);
                        if w == 1 {
                            self.learn_path(repl, lat, x, k != 0, env, undo, depth - 1);
                        }
                        self.learn_eq_const_inner(repl, lat, x, w, k, env, undo, depth - 1);
                    }
                    pred::NE => {
                        if w == 1 {
                            self.learn_path(repl, lat, x, k == 0, env, undo, depth - 1);
                        }
                        if k == 0 {
                            if base_kb.min_u() == 0 && all > 0 {
                                env.refine_urange(undo, x, base_kb, 1, all);
                            }
                            self.learn_ne_zero_inner(repl, lat, x, w, env, undo, depth - 1);
                        }
                    }
                    pred::IULT if k > 0 => {
                        env.refine_urange(undo, x, base_kb, 0, k - 1);
                    }
                    pred::IULE => {
                        env.refine_urange(undo, x, base_kb, 0, k);
                    }
                    pred::IUGT if k < all => {
                        env.refine_urange(undo, x, base_kb, k + 1, all);
                    }
                    pred::IUGE => {
                        env.refine_urange(undo, x, base_kb, k, all);
                    }
                    pred::SLT => {
                        let sk = signed(w, k);
                        let min_s = signed(w, 1u64 << (w - 1));
                        if sk > min_s {
                            env.refine_srange(undo, x, base_kb, min_s, sk - 1);
                        }
                    }
                    pred::SLE => {
                        let sk = signed(w, k);
                        let min_s = signed(w, 1u64 << (w - 1));
                        env.refine_srange(undo, x, base_kb, min_s, sk);
                    }
                    pred::SGT => {
                        let sk = signed(w, k);
                        let max_s = signed(w, (1u64 << (w - 1)) - 1);
                        if sk < max_s {
                            env.refine_srange(undo, x, base_kb, sk + 1, max_s);
                        }
                    }
                    pred::SGE => {
                        let sk = signed(w, k);
                        let max_s = signed(w, (1u64 << (w - 1)) - 1);
                        env.refine_srange(undo, x, base_kb, sk, max_s);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Propagates `x == k` backward into bit-mask/shift/cast producers of `x`.
    #[allow(clippy::too_many_arguments)]
    fn learn_eq_const_inner(
        &self,
        repl: &[Option<V>],
        lat: &[Lattice],
        x: V,
        w: u32,
        k: u64,
        env: &mut PathEnv,
        undo: &mut ScopeUndo,
        depth: u32,
    ) {
        if depth == 0 {
            return;
        }
        let V::Inst(i) = x else { return };
        match self.insts[i as usize].op {
            Op::Bin(Bin::And, a, b) => {
                let (a, b) = (resolve(repl, a), resolve(repl, b));
                let (y, m_val) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((_, m))) => (a, m),
                    (Some((_, m)), None) => (b, m),
                    _ => return,
                };
                if (k & !m_val) != 0 || self.is_undef(y) {
                    return;
                }
                if let Some(y_base) = self.effective_kb(lat, env, y) {
                    env.refine_kb(
                        undo,
                        y,
                        y_base,
                        KnownBits {
                            width: w,
                            zero: m_val & !k,
                            one: m_val & k,
                        },
                    );
                }
            }
            Op::Bin(Bin::LShr, a, b) => {
                let (a, b) = (resolve(repl, a), resolve(repl, b));
                if !self.is_undef(a)
                    && let Some((_, sh)) = self.int_k(b)
                    && sh < u64::from(w)
                {
                    let sh = sh as u32;
                    if k <= mask(w - sh)
                        && let Some(a_base) = self.effective_kb(lat, env, a)
                    {
                        let hi_mask = (!mask(sh)) & mask(w);
                        let shifted = (k << sh) & hi_mask;
                        env.refine_kb(
                            undo,
                            a,
                            a_base,
                            KnownBits {
                                width: w,
                                zero: hi_mask & !shifted,
                                one: shifted,
                            },
                        );
                    }
                }
            }
            Op::Cast(Cast::ZExt | Cast::SExt, src) if k == 0 => {
                let src = resolve(repl, src);
                if !self.is_undef(src) && self.int_width_of(src) == Some(1) {
                    self.learn_path(repl, lat, src, false, env, undo, depth - 1);
                }
            }
            Op::Cast(Cast::ZExt, src) => {
                let src = resolve(repl, src);
                if !self.is_undef(src)
                    && let Some(sw) = self.int_width_of(src)
                    && k <= mask(sw)
                {
                    if sw == 1 {
                        self.learn_path(repl, lat, src, k != 0, env, undo, depth - 1);
                    } else if let Some(s_base) = self.effective_kb(lat, env, src) {
                        env.refine_kb(undo, src, s_base, KnownBits::constant(sw, k));
                        env.refine_urange(undo, src, s_base, k, k);
                    }
                }
            }
            Op::Cast(Cast::Trunc, src) => {
                let src = resolve(repl, src);
                if !self.is_undef(src)
                    && let Some(sw) = self.int_width_of(src)
                    && let Some(s_base) = self.effective_kb(lat, env, src)
                {
                    let low = mask(w);
                    env.refine_kb(
                        undo,
                        src,
                        s_base,
                        KnownBits {
                            width: sw,
                            zero: low & !k,
                            one: low & k,
                        },
                    );
                }
            }
            _ => {}
        }
    }

    /// Propagates `x != 0` backward into single-bit mask tests and boolean extensions.
    #[allow(clippy::too_many_arguments)]
    fn learn_ne_zero_inner(
        &self,
        repl: &[Option<V>],
        lat: &[Lattice],
        x: V,
        w: u32,
        env: &mut PathEnv,
        undo: &mut ScopeUndo,
        depth: u32,
    ) {
        if depth == 0 {
            return;
        }
        let V::Inst(i) = x else { return };
        match self.insts[i as usize].op {
            Op::Bin(Bin::And, a, b) => {
                let (a, b) = (resolve(repl, a), resolve(repl, b));
                let (y, m_val) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((_, m))) => (a, m),
                    (Some((_, m)), None) => (b, m),
                    _ => return,
                };
                if !self.is_undef(y)
                    && m_val.is_power_of_two()
                    && let Some(y_base) = self.effective_kb(lat, env, y)
                {
                    env.refine_kb(
                        undo,
                        y,
                        y_base,
                        KnownBits {
                            width: w,
                            zero: 0,
                            one: m_val,
                        },
                    );
                }
            }
            Op::Cast(Cast::ZExt | Cast::SExt, src) => {
                let src = resolve(repl, src);
                if !self.is_undef(src) && self.int_width_of(src) == Some(1) {
                    self.learn_path(repl, lat, src, true, env, undo, depth - 1);
                }
            }
            _ => {}
        }
    }

    /// Intersects facts proven on every reachable incoming conditional branch
    /// edge of a multi-predecessor block `b`, retaining only facts on values
    /// that strictly dominate `b`.
    #[allow(clippy::too_many_arguments)]
    fn learn_multi_pred(
        &self,
        b: usize,
        preds: &[usize],
        orig_terms: &[Option<Term>],
        idom: &[usize],
        def_block: &[usize],
        repl: &[Option<V>],
        lat: &[Lattice],
        env: &mut PathEnv,
        undo: &mut ScopeUndo,
    ) {
        if preds.len() < 2 || preds.len() > 8 {
            return;
        }
        let mut common_bits: Option<HashMap<V, bool>> = None;
        let mut common_eq: Option<HashMap<V, V>> = None;
        for &p in preds {
            let Some(Term::CondBr(c, t, f)) = orig_terms[p] else {
                return;
            };
            if t == f {
                return;
            }
            let mut local_env = PathEnv::default();
            let mut local_undo = ScopeUndo::default();
            self.learn_path(
                repl,
                lat,
                c,
                b == t,
                &mut local_env,
                &mut local_undo,
                6,
            );
            common_bits = Some(match common_bits {
                None => local_env.bits,
                Some(mut acc) => {
                    acc.retain(|k, v| local_env.bits.get(k) == Some(v));
                    acc
                }
            });
            common_eq = Some(match common_eq {
                None => local_env.const_eq,
                Some(mut acc) => {
                    acc.retain(|k, v| local_env.const_eq.get(k) == Some(v));
                    acc
                }
            });
            if common_bits.as_ref().is_none_or(HashMap::is_empty)
                && common_eq.as_ref().is_none_or(HashMap::is_empty)
            {
                return;
            }
        }
        let dom_v = |v: V| match v {
            V::Inst(j) => {
                let db = def_block[j as usize];
                db != usize::MAX && db != b && dominates(idom, db, b)
            }
            _ => true,
        };
        if let Some(bits) = common_bits {
            let mut sorted: Vec<(V, bool)> = bits.into_iter().filter(|(v, _)| dom_v(*v)).collect();
            sorted.sort_unstable_by_key(|(v, _)| *v);
            for (v, holds) in sorted {
                self.learn_path(repl, lat, v, holds, env, undo, 4);
            }
        }
        if let Some(eqs) = common_eq {
            let mut sorted: Vec<(V, V)> = eqs.into_iter().filter(|(v, _)| dom_v(*v)).collect();
            sorted.sort_unstable_by_key(|(v, _)| *v);
            for (v, kv) in sorted {
                if let Some((w, k)) = self.int_k(kv)
                    && let Some(base) = self.effective_kb(lat, env, v)
                {
                    env.set_const_eq(undo, v, kv);
                    env.refine_kb(undo, v, base, KnownBits::constant(w, k));
                    env.refine_urange(undo, v, base, k, k);
                    let sk = signed(w, k);
                    env.refine_srange(undo, v, base, sk, sk);
                }
            }
        }
    }

    /// Evaluates pure expression `v` under a temporary assumption `cond == holds`,
    /// returning an already-existing dominating `V` (or `V::Const`) when `v`
    /// collapses to a simpler value.
    #[allow(clippy::too_many_arguments)]
    fn eval_assuming(
        &self,
        repl: &[Option<V>],
        lat: &[Lattice],
        env: &mut PathEnv,
        v: V,
        cond: V,
        holds: bool,
        yes: V,
        no: V,
        depth: u32,
    ) -> Option<V> {
        let mut undo = ScopeUndo::default();
        self.learn_path(repl, lat, cond, holds, env, &mut undo, 6);
        let res = self.eval_under_env(repl, lat, env, v, yes, no, depth);
        env.rollback(undo);
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_under_env(
        &self,
        repl: &[Option<V>],
        lat: &[Lattice],
        env: &PathEnv,
        v: V,
        yes: V,
        no: V,
        depth: u32,
    ) -> Option<V> {
        let v = resolve(repl, v);
        if let Some(&b) = env.bits.get(&v) {
            return Some(if b { yes } else { no });
        }
        if let Some(&k) = env.const_eq.get(&v) {
            return Some(k);
        }
        if depth == 0 {
            return None;
        }
        let V::Inst(i) = v else { return None };
        match self.insts[i as usize].op {
            Op::Cmp(p, a, b) => {
                let (a, b) = (resolve(repl, a), resolve(repl, b));
                if let Some(&h) = env.compares.get(&canonical_cmp(p, a, b)) {
                    return Some(if h { yes } else { no });
                }
                if let Some(h) = self.eval_int_cmp(lat, env, p, a, b) {
                    return Some(if h { yes } else { no });
                }
                None
            }
            Op::Bin(op, a, b) if self.ty_int_bits(self.insts[i as usize].ty) == Some(1) => {
                let ea = self
                    .eval_under_env(repl, lat, env, a, yes, no, depth - 1)
                    .unwrap_or_else(|| resolve(repl, a));
                let eb = self
                    .eval_under_env(repl, lat, env, b, yes, no, depth - 1)
                    .unwrap_or_else(|| resolve(repl, b));
                let ba = self.int_k(ea).map(|(_, k)| k != 0);
                let bb = self.int_k(eb).map(|(_, k)| k != 0);
                match op {
                    Bin::And => match (ba, bb) {
                        (Some(false), _) | (_, Some(false)) => Some(no),
                        (Some(true), _) => Some(eb),
                        (_, Some(true)) => Some(ea),
                        _ if ea == eb && !self.is_undef(ea) => Some(ea),
                        _ => None,
                    },
                    Bin::Or => match (ba, bb) {
                        (Some(true), _) | (_, Some(true)) => Some(yes),
                        (Some(false), _) => Some(eb),
                        (_, Some(false)) => Some(ea),
                        _ if ea == eb && !self.is_undef(ea) => Some(ea),
                        _ => None,
                    },
                    Bin::Xor => match (ba, bb) {
                        (Some(x), Some(y)) => Some(if x != y { yes } else { no }),
                        (Some(false), _) => Some(eb),
                        (_, Some(false)) => Some(ea),
                        _ if ea == eb && !self.is_undef(ea) => Some(no),
                        _ => None,
                    },
                    _ => None,
                }
            }
            Op::Select(c, t, f) => {
                let ec = self
                    .eval_under_env(repl, lat, env, c, yes, no, depth - 1)
                    .unwrap_or_else(|| resolve(repl, c));
                match self.int_k(ec).map(|(_, k)| k != 0) {
                    Some(true) => Some(
                        self.eval_under_env(repl, lat, env, t, yes, no, depth - 1)
                            .unwrap_or_else(|| resolve(repl, t)),
                    ),
                    Some(false) => Some(
                        self.eval_under_env(repl, lat, env, f, yes, no, depth - 1)
                            .unwrap_or_else(|| resolve(repl, f)),
                    ),
                    None => None,
                }
            }
            _ => None,
        }
    }

    /// Sub-phase A: Whole-function SCKP + dominator-tree correlated predicate,
    /// known-bits, bitfield, select-implication, and clamp simplification.
    fn dataflow_forward(&mut self) -> bool {
        let sckp = self.run_sckp();
        let tree: Tree = self.dominator_tree();
        let effects = self.effects();
        let ni = self.insts.len();
        let mut def_block = vec![usize::MAX; ni];
        for (b, blk) in self.blocks.iter().enumerate() {
            for &i in blk.phis.iter().chain(&blk.body) {
                def_block[i as usize] = b;
            }
        }

        let mut repl: Vec<Option<V>> = vec![None; ni];
        let mut changed = false;
        let (yes, no) = (self.bool(true), self.bool(false));

        // 1. Fold branches whose condition SCKP proved constant (even across loop PHI cycles).
        for &b in &tree.order {
            if !sckp.exec_block[b] {
                continue;
            }
            if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term
                && t != f
                && self.int_k(c).is_none()
                && let Some(kb) = self.eval_kb(&sckp.lat, c)
                && let Some(k) = kb.const_val()
            {
                self.blocks[b].term = Some(Term::CondBr(if k != 0 { yes } else { no }, t, f));
                changed = true;
            }
        }

        // 2. Collapse trivial PHIs using only SCKP-executable incoming edges.
        let mut phi_again = true;
        while phi_again {
            phi_again = false;
            for &b in &tree.order {
                if !sckp.exec_block[b] {
                    continue;
                }
                for pi in 0..self.blocks[b].phis.len() {
                    let p = self.blocks[b].phis[pi];
                    if repl[p as usize].is_some() {
                        continue;
                    }
                    // Constant PHI from SCKP.
                    if let Lattice::Bits(kb) = sckp.lat[p as usize]
                        && let Some(k) = kb.const_val()
                    {
                        let ty = self.insts[p as usize].ty;
                        repl[p as usize] = Some(self.const_int(ty, kb.width, k));
                        phi_again = true;
                        changed = true;
                        continue;
                    }
                    let Op::Phi(incoming) = &self.insts[p as usize].op else {
                        continue;
                    };
                    let mut one = None;
                    let mut same = true;
                    for &(v, pred_b) in incoming {
                        if !sckp.exec_edge.contains(&(pred_b, b)) {
                            continue;
                        }
                        let v = resolve(&repl, v);
                        if v == V::Inst(p) {
                            continue;
                        }
                        match one {
                            None => one = Some(v),
                            Some(o) if o == v => {}
                            Some(_) => same = false,
                        }
                    }
                    if let (true, Some(v)) = (same, one) {
                        let ok_dom = match v {
                            V::Inst(j) => {
                                let db = def_block[j as usize];
                                db != usize::MAX && dominates(&tree.idom, db, b)
                            }
                            _ => true,
                        };
                        if ok_dom {
                            repl[p as usize] = Some(v);
                            phi_again = true;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 3. Dominator-tree walk with correlated PathEnv + KnownBits + Bitfield rewrites.
        // Snapshot pre-mutation block terminators so child blocks and multi-pred
        // joins inspect the original branch conditions consistently with SCKP.
        let orig_terms: Vec<Option<Term>> = self.blocks.iter().map(|blk| blk.term).collect();
        let mut env = PathEnv::default();
        enum Step {
            Enter(usize),
            Leave(ScopeUndo),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(undo) => {
                    env.rollback(undo);
                    continue;
                }
            };
            if !sckp.exec_block[b] {
                continue;
            }
            let mut undo = ScopeUndo::default();
            let exec_preds: Vec<usize> = tree.preds[b]
                .iter()
                .copied()
                .filter(|&p| sckp.exec_edge.contains(&(p, b)))
                .collect();
            if let [p] = exec_preds[..]
                && let Some(Term::CondBr(c, t, f)) = orig_terms[p]
                && t != f
            {
                self.learn_path(
                    &repl,
                    &sckp.lat,
                    c,
                    b == t,
                    &mut env,
                    &mut undo,
                    8,
                );
            } else if exec_preds.len() >= 2 {
                self.learn_multi_pred(
                    b,
                    &exec_preds,
                    &orig_terms,
                    &tree.idom,
                    &def_block,
                    &repl,
                    &sckp.lat,
                    &mut env,
                    &mut undo,
                );
            }

            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                // Substitute resolved values, scoped known booleans, and scoped constant equalities.
                map_operands(&mut self.insts[i as usize].op, |v| {
                    let r = resolve(&repl, v);
                    let r = if let Some(&bit) = env.bits.get(&r) {
                        if bit { yes } else { no }
                    } else if let Some(&k) = env.const_eq.get(&r) {
                        k
                    } else {
                        r
                    };
                    if r != v {
                        changed = true;
                    }
                    r
                });

                let pure = self.is_pure_inst(&effects, i);
                if pure
                    && let Some(r) =
                        self.simplify_forward_inst(i, &repl, &sckp.lat, &mut env, yes, no, &mut changed)
                {
                    repl[i as usize] = Some(r);
                    changed = true;
                    continue;
                }
                kept.push(i);
            }
            self.blocks[b].body = kept;

            if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term {
                let mut nc = resolve(&repl, c);
                if let Some(&bit) = env.bits.get(&nc) {
                    nc = if bit { yes } else { no };
                } else if let Some(kb) = self.effective_kb(&sckp.lat, &env, nc)
                    && let Some(k) = kb.const_val()
                {
                    nc = if k != 0 { yes } else { no };
                }
                if nc != c {
                    self.blocks[b].term = Some(Term::CondBr(nc, t, f));
                    changed = true;
                }
            }

            steps.push(Step::Leave(undo));
            for &child in tree.children[b].iter().rev() {
                steps.push(Step::Enter(child));
            }
        }

        if !changed {
            return false;
        }
        self.apply_replacements(&repl);
        true
    }

    fn is_pure_inst(&self, effects: &[Effect], i: u32) -> bool {
        match &self.insts[i as usize].op {
            Op::Bin(..)
            | Op::Cast(..)
            | Op::Cmp(..)
            | Op::Select(..)
            | Op::Extract(..)
            | Op::Gep(..) => true,
            Op::Call(f, _) => {
                // Never treat `dx.op.getDimensions` as rewritable/coalescable.
                let name = self.funcs[*f as usize].name.as_str();
                let name = name.strip_prefix("dx.op.").unwrap_or(name);
                !name.starts_with("getDimensions") && effects[*f as usize] == Effect::Pure
            }
            _ => false,
        }
    }

    fn apply_replacements(&mut self, repl: &[Option<V>]) {
        for blk in &mut self.blocks {
            blk.phis.retain(|&p| repl[p as usize].is_none());
            if let Some(Term::CondBr(c, t, f)) = blk.term {
                blk.term = Some(Term::CondBr(resolve(repl, c), t, f));
            }
        }
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for blk in blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                map_operands(&mut insts[i as usize].op, |v| resolve(repl, v));
            }
        }
    }

    /// Forward KnownBits, bitfield, exec-select, cast, and clamp simplification
    /// for a single pure instruction `i`.
    #[allow(clippy::too_many_arguments)]
    fn simplify_forward_inst(
        &mut self,
        i: u32,
        repl: &[Option<V>],
        lat: &[Lattice],
        env: &mut PathEnv,
        yes: V,
        no: V,
        changed: &mut bool,
    ) -> Option<V> {
        let ty = self.insts[i as usize].ty;
        // Global SCKP or scoped PathEnv proved `i` is a constant integer.
        if let Some(w) = self.ty_int_bits(ty)
            && let Some(kb) = self.effective_kb(lat, env, V::Inst(i))
            && let Some(k) = kb.const_val()
        {
            return Some(self.const_int(ty, w, k));
        }

        match self.insts[i as usize].op.clone() {
            Op::Cmp(p, a, b) => {
                if let Some(&holds) = env.compares.get(&canonical_cmp(p, a, b)) {
                    return Some(if holds { yes } else { no });
                }
                if p <= pred::TRUE {
                    if p == pred::FALSE {
                        return Some(no);
                    }
                    if p == pred::TRUE {
                        return Some(yes);
                    }
                    return None;
                }
                if let Some(holds) = self.eval_int_cmp(lat, env, p, a, b) {
                    return Some(if holds { yes } else { no });
                }
                // Comparing `zext i1 c` / `sext i1 c` / `select(c, k1, k2)` against a constant.
                let holds_k = |x: u64, y: u64, bits: u32| -> bool {
                    let (sx, sy) = (signed(bits, x), signed(bits, y));
                    match p {
                        pred::EQ => x == y,
                        pred::NE => x != y,
                        pred::IUGT => x > y,
                        pred::IUGE => x >= y,
                        pred::IULT => x < y,
                        pred::IULE => x <= y,
                        pred::SGT => sx > sy,
                        pred::SGE => sx >= sy,
                        pred::SLT => sx < sy,
                        _ => sx <= sy,
                    }
                };
                let two_val = |v: V, bits: u32| -> Option<(V, u64, u64)> {
                    if bits == 1 && self.int_k(v).is_none() && !self.is_undef(v) {
                        return Some((v, 1, 0));
                    }
                    let V::Inst(j) = v else { return None };
                    match self.insts[j as usize].op {
                        Op::Select(c, t, f) => {
                            Some((c, self.int_k(t)?.1, self.int_k(f)?.1))
                        }
                        Op::Cast(Cast::ZExt, x) if self.int_width_of(x) == Some(1) => {
                            Some((x, 1, 0))
                        }
                        Op::Cast(Cast::SExt, x) if self.int_width_of(x) == Some(1) => {
                            Some((x, mask(bits), 0))
                        }
                        _ => None,
                    }
                };
                let reduced = if let Some((bits, k)) = self.int_k(b)
                    && let Some((c, k1, k2)) = two_val(a, bits)
                {
                    Some((c, holds_k(k1, k, bits), holds_k(k2, k, bits)))
                } else if let Some((bits, k)) = self.int_k(a)
                    && let Some((c, k1, k2)) = two_val(b, bits)
                {
                    Some((c, holds_k(k, k1, bits), holds_k(k, k2, bits)))
                } else {
                    None
                };
                if let Some((c, wt, wf)) = reduced {
                    return match (wt, wf) {
                        (true, true) => Some(yes),
                        (false, false) => Some(no),
                        (true, false) => Some(c),
                        (false, true) => self.negate_bool_inst(i, c, yes, no, changed),
                    };
                }
                None
            }
            Op::Select(mut c, mut t, mut f) => {
                let mut mutated = false;
                while let Some(inner_c) = self.is_bool_not_v(c) {
                    c = inner_c;
                    std::mem::swap(&mut t, &mut f);
                    mutated = true;
                }
                if let Some((_, k)) = self.int_k(c) {
                    return Some(if k != 0 { t } else { f });
                }
                // Correlated select-arm simplification (`exec`-mask / `v_cndmask` collapse):
                // simplify `t` assuming `c == true`, and `f` assuming `c == false`.
                if let Some(nt) = self.eval_assuming(repl, lat, env, t, c, true, yes, no, 4)
                    && nt != t
                {
                    t = nt;
                    mutated = true;
                }
                if let Some(nf) = self.eval_assuming(repl, lat, env, f, c, false, yes, no, 4)
                    && nf != f
                {
                    f = nf;
                    mutated = true;
                }
                if t == f && !self.is_undef(t) {
                    return Some(t);
                }
                if self.ty_int_bits(ty) == Some(1) {
                    let kt = self.int_k(t).map(|(_, k)| k != 0);
                    let kf = self.int_k(f).map(|(_, k)| k != 0);
                    match (kt, kf) {
                        (Some(true), Some(false)) => return Some(c),
                        (Some(false), Some(true)) => {
                            return self.negate_bool_inst(i, c, yes, no, changed);
                        }
                        (Some(true), _) => {
                            if f == c {
                                return Some(c);
                            }
                            self.insts[i as usize].op = Op::Bin(Bin::Or, c, f);
                            *changed = true;
                            return None;
                        }
                        (_, Some(false)) => {
                            if t == c {
                                return Some(c);
                            }
                            self.insts[i as usize].op = Op::Bin(Bin::And, c, t);
                            *changed = true;
                            return None;
                        }
                        _ => {}
                    }
                    if t == c {
                        self.insts[i as usize].op = Op::Bin(Bin::Or, c, f);
                        *changed = true;
                        return None;
                    }
                    if f == c {
                        self.insts[i as usize].op = Op::Bin(Bin::And, c, t);
                        *changed = true;
                        return None;
                    }
                }
                if mutated {
                    self.insts[i as usize].op = Op::Select(c, t, f);
                    *changed = true;
                }
                None
            }
            Op::Bin(op, a, b) => {
                // Never fold floating-point arithmetic.
                let w = self.ty_int_bits(ty)?;
                let m = mask(w);
                let ka = self.effective_kb(lat, env, a)?;
                let kb = self.effective_kb(lat, env, b)?;

                // Evaluate binary transfer under refined PathEnv KnownBits.
                let res_kb = match op {
                    Bin::And => ka.and(kb),
                    Bin::Or => ka.or(kb),
                    Bin::Xor => ka.xor(kb),
                    Bin::Add => ka.add_sub(kb, false),
                    Bin::Sub => ka.add_sub(kb, true),
                    Bin::Shl | Bin::LShr | Bin::AShr => ka.shift(op, kb),
                    Bin::Mul => ka.mul(kb),
                    Bin::UDiv => ka.udiv(kb),
                    Bin::URem => ka.urem(kb),
                    Bin::SDiv => ka.sdiv(kb),
                };
                if let Some(k) = res_kb.const_val() {
                    return Some(self.const_int(ty, w, k));
                }

                // Boolean `i1` implication collapse for `exec`-mask logic.
                if w == 1 {
                    match op {
                        Bin::And => {
                            if let Some(eb) =
                                self.eval_assuming(repl, lat, env, b, a, true, yes, no, 4)
                            {
                                if eb == yes {
                                    return Some(a);
                                }
                                if eb == no {
                                    return Some(no);
                                }
                                if eb != b {
                                    self.insts[i as usize].op = Op::Bin(Bin::And, a, eb);
                                    *changed = true;
                                    return None;
                                }
                            }
                            if let Some(ea) =
                                self.eval_assuming(repl, lat, env, a, b, true, yes, no, 4)
                            {
                                if ea == yes {
                                    return Some(b);
                                }
                                if ea == no {
                                    return Some(no);
                                }
                                if ea != a {
                                    self.insts[i as usize].op = Op::Bin(Bin::And, ea, b);
                                    *changed = true;
                                    return None;
                                }
                            }
                        }
                        Bin::Or => {
                            if let Some(eb) =
                                self.eval_assuming(repl, lat, env, b, a, false, yes, no, 4)
                            {
                                if eb == no {
                                    return Some(a);
                                }
                                if eb == yes {
                                    return Some(yes);
                                }
                                if eb != b {
                                    self.insts[i as usize].op = Op::Bin(Bin::Or, a, eb);
                                    *changed = true;
                                    return None;
                                }
                            }
                            if let Some(ea) =
                                self.eval_assuming(repl, lat, env, a, b, false, yes, no, 4)
                            {
                                if ea == no {
                                    return Some(b);
                                }
                                if ea == yes {
                                    return Some(yes);
                                }
                                if ea != a {
                                    self.insts[i as usize].op = Op::Bin(Bin::Or, ea, b);
                                    *changed = true;
                                    return None;
                                }
                            }
                        }
                        Bin::Xor => {
                            if kb.const_val() == Some(1) {
                                return self.negate_bool_inst(i, a, yes, no, changed);
                            }
                            if ka.const_val() == Some(1) {
                                return self.negate_bool_inst(i, b, yes, no, changed);
                            }
                        }
                        _ => {}
                    }
                }

                // KnownBits-driven identities across all integer widths.
                match op {
                    Bin::And => {
                        if ka.max_u() & !kb.one == 0 {
                            return Some(a);
                        }
                        if kb.max_u() & !ka.one == 0 {
                            return Some(b);
                        }
                        if a == b && !self.is_undef(a) {
                            return Some(a);
                        }
                        // Absorption: `and(or(x, y), x) => x`.
                        if let Some(absorbed) = self.bitwise_absorb(Bin::Or, a, b) {
                            return Some(absorbed);
                        }
                        // Per-use bit-demand bypass (sequential so stripping `a`
                        // using `b`'s known zeros cannot simultaneously strip `b`
                        // using `a`'s pre-strip known zeros, e.g. `(x & 0xFF) & (y & 0xFF)`).
                        let na = self.strip_unused_bits(lat, env, a, kb.max_u(), 6);
                        let nka = if na != a {
                            self.effective_kb(lat, env, na)
                                .unwrap_or_else(|| KnownBits::unknown(w))
                        } else {
                            ka
                        };
                        let nb = self.strip_unused_bits(lat, env, b, nka.max_u(), 6);
                        if na != a || nb != b {
                            self.insts[i as usize].op = Op::Bin(Bin::And, na, nb);
                            *changed = true;
                            if na == nb && !self.is_undef(na) {
                                return Some(na);
                            }
                            if let Some(nkb) = self.effective_kb(lat, env, nb) {
                                if nka.max_u() & !nkb.one == 0 {
                                    return Some(na);
                                }
                                if nkb.max_u() & !nka.one == 0 {
                                    return Some(nb);
                                }
                            }
                        }
                    }
                    Bin::Or => {
                        if kb.max_u() & !ka.one == 0 {
                            return Some(a);
                        }
                        if ka.max_u() & !kb.one == 0 {
                            return Some(b);
                        }
                        if a == b && !self.is_undef(a) {
                            return Some(a);
                        }
                        // Absorption: `or(and(x, y), x) => x`.
                        if let Some(absorbed) = self.bitwise_absorb(Bin::And, a, b) {
                            return Some(absorbed);
                        }
                        // `v_bfi_b32`: `or(and(m, x), and(xor(m, -1), x)) => x`.
                        if let Some(same) = self.simplify_bfi_same(a, b, w) {
                            return Some(same);
                        }
                        // Sequential per-use bit-demand bypass (prevents mutual strip
                        // of `(x | 0xFF00) | (y | 0xFF00)`).
                        let na = self.strip_unused_bits(lat, env, a, (!kb.one) & m, 6);
                        let nka = if na != a {
                            self.effective_kb(lat, env, na)
                                .unwrap_or_else(|| KnownBits::unknown(w))
                        } else {
                            ka
                        };
                        let nb = self.strip_unused_bits(lat, env, b, (!nka.one) & m, 6);
                        if na != a || nb != b {
                            self.insts[i as usize].op = Op::Bin(Bin::Or, na, nb);
                            *changed = true;
                            if na == nb && !self.is_undef(na) {
                                return Some(na);
                            }
                            if let Some(nkb) = self.effective_kb(lat, env, nb) {
                                if nkb.max_u() & !nka.one == 0 {
                                    return Some(na);
                                }
                                if nka.max_u() & !nkb.one == 0 {
                                    return Some(nb);
                                }
                            }
                        }
                    }
                    Bin::Xor => {
                        if kb.zero == m {
                            return Some(a);
                        }
                        if ka.zero == m {
                            return Some(b);
                        }
                        if a == b && !self.is_undef(a) {
                            return Some(self.const_int(ty, w, 0));
                        }
                    }
                    Bin::Add => {
                        if kb.zero == m {
                            return Some(a);
                        }
                        if ka.zero == m {
                            return Some(b);
                        }
                    }
                    Bin::Sub => {
                        if kb.zero == m {
                            return Some(a);
                        }
                        if a == b && !self.is_undef(a) {
                            return Some(self.const_int(ty, w, 0));
                        }
                    }
                    Bin::Mul => {
                        if kb.const_val() == Some(1) {
                            return Some(a);
                        }
                        if ka.const_val() == Some(1) {
                            return Some(b);
                        }
                    }
                    Bin::UDiv => {
                        if kb.const_val() == Some(1) {
                            return Some(a);
                        }
                        if kb.min_u() > 0 && ka.max_u() < kb.min_u() {
                            return Some(self.const_int(ty, w, 0));
                        }
                    }
                    Bin::URem => {
                        if kb.const_val() == Some(1) {
                            return Some(self.const_int(ty, w, 0));
                        }
                        if kb.min_u() > 0 && ka.max_u() < kb.min_u() {
                            return Some(a);
                        }
                    }
                    Bin::Shl | Bin::LShr | Bin::AShr => {
                        if kb.zero == m {
                            return Some(a);
                        }
                        if let Some(c2) = kb.const_val()
                            && c2 < u64::from(w)
                        {
                            let c2_u32 = c2 as u32;
                            let a = match op {
                                Bin::Shl => {
                                    let na = self.strip_unused_bits(
                                        lat,
                                        env,
                                        a,
                                        mask(w - c2_u32),
                                        6,
                                    );
                                    if na != a {
                                        self.insts[i as usize].op = Op::Bin(Bin::Shl, na, b);
                                        *changed = true;
                                    }
                                    na
                                }
                                Bin::LShr => {
                                    let na = self.strip_unused_bits(
                                        lat,
                                        env,
                                        a,
                                        (!mask(c2_u32)) & m,
                                        6,
                                    );
                                    if na != a {
                                        self.insts[i as usize].op = Op::Bin(Bin::LShr, na, b);
                                        *changed = true;
                                    }
                                    na
                                }
                                _ => a,
                            };
                            return self.simplify_shift_pair(i, ty, w, op, a, c2_u32, lat, env, changed);
                        }
                    }
                    _ => {}
                }
                None
            }
            Op::Cast(op, x) => {
                let from = self.type_of(x);
                match op {
                    Cast::Bitcast => {
                        if from == ty {
                            return Some(x);
                        }
                        if let V::Inst(j) = x
                            && let Op::Cast(Cast::Bitcast, z) = self.insts[j as usize].op
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        None
                    }
                    Cast::Trunc => {
                        let to_w = self.ty_int_bits(ty)?;
                        let nx = self.strip_unused_bits(lat, env, x, mask(to_w), 6);
                        let x = if nx != x {
                            self.insts[i as usize].op = Op::Cast(Cast::Trunc, nx);
                            *changed = true;
                            nx
                        } else {
                            x
                        };
                        if let V::Inst(j) = x
                            && let Op::Cast(inner_op @ (Cast::ZExt | Cast::SExt), z) =
                                self.insts[j as usize].op
                        {
                            let zw = self.int_width_of(z)?;
                            if zw == to_w {
                                return Some(z);
                            }
                            if to_w < zw {
                                self.insts[i as usize].op = Op::Cast(Cast::Trunc, z);
                                *changed = true;
                                return None;
                            }
                            self.insts[i as usize].op = Op::Cast(inner_op, z);
                            *changed = true;
                            return None;
                        }
                        None
                    }
                    Cast::ZExt => {
                        let to_w = self.ty_int_bits(ty)?;
                        let from_w = self.ty_int_bits(from)?;
                        // `zext(trunc(z))` -> `z` when upper bits of `z` are already known 0.
                        if let V::Inst(j) = x
                            && let Op::Cast(Cast::Trunc, z) = self.insts[j as usize].op
                            && self.type_of(z) == ty
                            && let Some(kz) = self.effective_kb(lat, env, z)
                        {
                            let hi = (!mask(from_w)) & mask(to_w);
                            if (kz.zero & hi) == hi {
                                return Some(z);
                            }
                        }
                        None
                    }
                    Cast::SExt => {
                        let to_w = self.ty_int_bits(ty)?;
                        let from_w = self.ty_int_bits(from)?;
                        // `sext(trunc(z))` -> `z` when top `(to_w - from_w + 1)` bits of `z`
                        // are all known 0 or all known 1.
                        if let V::Inst(j) = x
                            && let Op::Cast(Cast::Trunc, z) = self.insts[j as usize].op
                            && self.type_of(z) == ty
                            && let Some(kz) = self.effective_kb(lat, env, z)
                        {
                            let top = (!mask(from_w - 1)) & mask(to_w);
                            if (kz.zero & top) == top || (kz.one & top) == top {
                                return Some(z);
                            }
                        }
                        // Non-negative `sext(x)` is identical to `zext(x)`.
                        if let Some(kx) = self.effective_kb(lat, env, x)
                            && kx.sign_is_zero()
                        {
                            self.insts[i as usize].op = Op::Cast(Cast::ZExt, x);
                            *changed = true;
                        }
                        None
                    }
                    _ => None,
                }
            }
            Op::Call(f, args) => {
                let name = self.funcs[f as usize].name.as_str();
                let name = name.strip_prefix("dx.op.").unwrap_or(name);
                // Never touch `dx.op.getDimensions`.
                if name != "binary.i32" || args.len() != 3 {
                    return None;
                }
                let (_, opcode) = self.int_k(args[0])?;
                let (a, b) = (args[1], args[2]);
                let is_clamp = self.tags.get(&i) == Some(&Tag::Clamp);
                if !is_clamp && a == b && !self.is_undef(a) && (37..=40).contains(&opcode) {
                    return Some(a);
                }
                match opcode {
                    // UMin (including `Tag::Clamp`): for tagged per-handle clamps
                    // (RDNA4 / Belobug#324), never merge/fold onto another clamp
                    // or `UMin` and require strictly smaller non-clamp provenance.
                    40 => {
                        let (ua, ub) = (
                            self.effective_urange(lat, env, a)?,
                            self.effective_urange(lat, env, b)?,
                        );
                        if is_clamp {
                            if ua.1 < ub.0 && !self.is_clamp_or_umin(a) {
                                Some(a)
                            } else if ub.1 < ua.0 && !self.is_clamp_or_umin(b) {
                                Some(b)
                            } else {
                                None
                            }
                        } else if ua.1 <= ub.0 {
                            Some(a)
                        } else if ub.1 <= ua.0 {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    // UMax.
                    39 => {
                        let (ua, ub) = (
                            self.effective_urange(lat, env, a)?,
                            self.effective_urange(lat, env, b)?,
                        );
                        if ua.0 >= ub.1 {
                            Some(a)
                        } else if ub.0 >= ua.1 {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    // IMin.
                    38 => {
                        let (sa, sb) = (
                            self.effective_srange(lat, env, a)?,
                            self.effective_srange(lat, env, b)?,
                        );
                        if sa.1 <= sb.0 {
                            Some(a)
                        } else if sb.1 <= sa.0 {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    // IMax.
                    37 => {
                        let (sa, sb) = (
                            self.effective_srange(lat, env, a)?,
                            self.effective_srange(lat, env, b)?,
                        );
                        if sa.0 >= sb.1 {
                            Some(a)
                        } else if sb.0 >= sa.1 {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn is_clamp_or_umin(&self, v: V) -> bool {
        let V::Inst(j) = v else { return false };
        if self.tags.get(&j) == Some(&Tag::Clamp) {
            return true;
        }
        if let Op::Call(f, args) = &self.insts[j as usize].op {
            let name = self.funcs[*f as usize].name.as_str();
            let name = name.strip_prefix("dx.op.").unwrap_or(name);
            if name == "binary.i32"
                && args.len() == 3
                && self.int_k(args[0]).is_some_and(|(_, op)| op == 40)
            {
                return true;
            }
        }
        false
    }

    fn negate_bool_inst(
        &mut self,
        i: u32,
        x: V,
        yes: V,
        no: V,
        changed: &mut bool,
    ) -> Option<V> {
        if let Some((1, k)) = self.int_k(x) {
            return Some(if k == 0 { yes } else { no });
        }
        if let Some(orig) = self.is_bool_not_v(x) {
            return Some(orig);
        }
        if let V::Inst(j) = x
            && let Op::Cmp(p, a, b) = self.insts[j as usize].op
            && (p <= pred::TRUE || (pred::EQ..=pred::SLE).contains(&p))
        {
            let opp = opposite(p);
            if opp == pred::FALSE {
                return Some(no);
            }
            if opp == pred::TRUE {
                return Some(yes);
            }
            // Never emit `fcmp one` (`pred::ONE`): AMD D3D12 miscompiles it as `une`.
            if opp != pred::ONE {
                self.insts[i as usize].op = Op::Cmp(opp, a, b);
                *changed = true;
                return None;
            }
        }
        let already = matches!(
            self.insts[i as usize].op,
            Op::Bin(Bin::Xor, a, b) if (a == x && b == yes) || (a == yes && b == x)
        );
        if !already {
            self.insts[i as usize].op = Op::Bin(Bin::Xor, x, yes);
            *changed = true;
        }
        None
    }

    /// Bypasses pure non-PHI bit operations in `v` that do not affect any bit
    /// in `use_mask`, returning an already-dominating `V` agreeing with `v` on
    /// every bit of `use_mask`.
    fn strip_unused_bits(
        &self,
        lat: &[Lattice],
        env: &PathEnv,
        mut v: V,
        use_mask: u64,
        depth: u32,
    ) -> V {
        if use_mask == 0 || depth == 0 {
            return v;
        }
        for _ in 0..depth {
            let V::Inst(j) = v else { break };
            let Some(w) = self.ty_int_bits(self.insts[j as usize].ty) else {
                break;
            };
            let m = mask(w);
            let u = use_mask & m;
            let next = match self.insts[j as usize].op {
                Op::Bin(Bin::Or, a, b) => {
                    if self.effective_kb(lat, env, b).is_some_and(|k| (k.zero & u) == u) {
                        Some(a)
                    } else if self.effective_kb(lat, env, a).is_some_and(|k| (k.zero & u) == u) {
                        Some(b)
                    } else {
                        None
                    }
                }
                Op::Bin(Bin::And, a, b) => {
                    if self.effective_kb(lat, env, b).is_some_and(|k| (k.one & u) == u) {
                        Some(a)
                    } else if self.effective_kb(lat, env, a).is_some_and(|k| (k.one & u) == u) {
                        Some(b)
                    } else {
                        None
                    }
                }
                Op::Bin(Bin::Xor, a, b) => {
                    if self.effective_kb(lat, env, b).is_some_and(|k| (k.zero & u) == u) {
                        Some(a)
                    } else if self.effective_kb(lat, env, a).is_some_and(|k| (k.zero & u) == u) {
                        Some(b)
                    } else {
                        None
                    }
                }
                Op::Bin(Bin::Add, a, b) => {
                    let low = low_mask_up_to(u) & m;
                    if self.effective_kb(lat, env, b).is_some_and(|k| (k.zero & low) == low) {
                        Some(a)
                    } else if self.effective_kb(lat, env, a).is_some_and(|k| (k.zero & low) == low)
                    {
                        Some(b)
                    } else {
                        None
                    }
                }
                Op::Bin(Bin::Sub, a, b) => {
                    let low = low_mask_up_to(u) & m;
                    if self.effective_kb(lat, env, b).is_some_and(|k| (k.zero & low) == low) {
                        Some(a)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            match next {
                Some(n) if n != v => v = n,
                _ => break,
            }
        }
        v
    }

    fn bitwise_absorb(&self, inner_op: Bin, a: V, b: V) -> Option<V> {
        let check = |cand: V, other: V| -> Option<V> {
            let V::Inst(j) = cand else { return None };
            let Op::Bin(op, x, y) = self.insts[j as usize].op else {
                return None;
            };
            (op == inner_op && (x == other || y == other) && !self.is_undef(other)).then_some(other)
        };
        check(a, b).or_else(|| check(b, a))
    }

    /// Recognizes `or(and(m, x), and(xor(m, -1), x))` (from `v_bfi_b32`) -> `x`.
    fn simplify_bfi_same(&self, a: V, b: V, w: u32) -> Option<V> {
        let all = mask(w);
        let (V::Inst(ia), V::Inst(ib)) = (a, b) else {
            return None;
        };
        let (Op::Bin(Bin::And, a0, a1), Op::Bin(Bin::And, b0, b1)) =
            (self.insts[ia as usize].op.clone(), self.insts[ib as usize].op.clone())
        else {
            return None;
        };
        let is_not = |u: V, m: V| -> bool {
            let V::Inst(k) = u else { return false };
            let Op::Bin(Bin::Xor, x, y) = self.insts[k as usize].op else {
                return false;
            };
            (x == m && self.int_k(y).is_some_and(|(_, v)| v == all))
                || (y == m && self.int_k(x).is_some_and(|(_, v)| v == all))
        };
        for (ma, xa) in [(a0, a1), (a1, a0)] {
            for (mb, xb) in [(b0, b1), (b1, b0)] {
                if xa == xb && !self.is_undef(xa) && (is_not(ma, mb) || is_not(mb, ma)) {
                    return Some(xa);
                }
            }
        }
        None
    }

    /// Collapses nested shift pairs (`(x << c1) >>u c2`, `(x << c1) >>s c2`,
    /// `(x >>u c1) << c2`, and same-direction shift chains) using KnownBits.
    #[allow(clippy::too_many_arguments)]
    fn simplify_shift_pair(
        &mut self,
        i: u32,
        ty: Ty,
        w: u32,
        op: Bin,
        a: V,
        c2: u32,
        lat: &[Lattice],
        env: &PathEnv,
        changed: &mut bool,
    ) -> Option<V> {
        let V::Inst(j) = a else { return None };
        let Op::Bin(inner_op, x, sh1) = self.insts[j as usize].op else {
            return None;
        };
        let (_, c1_u64) = self.int_k(sh1)?;
        if c1_u64 >= u64::from(w) {
            return None;
        }
        let c1 = c1_u64 as u32;
        let m = mask(w);
        let kx = self.effective_kb(lat, env, x)?;

        // Same-direction shifts combine.
        if inner_op == op {
            let sum = c1 + c2;
            if sum < w {
                let k = self.const_int(ty, w, u64::from(sum));
                self.insts[i as usize].op = Op::Bin(op, x, k);
                *changed = true;
                return None;
            }
            return match op {
                Bin::Shl | Bin::LShr => Some(self.const_int(ty, w, 0)),
                Bin::AShr => {
                    let max_s = w - 1;
                    if c1 == max_s {
                        return Some(a);
                    }
                    let k = self.const_int(ty, w, u64::from(max_s));
                    self.insts[i as usize].op = Op::Bin(Bin::AShr, x, k);
                    *changed = true;
                    None
                }
                _ => None,
            };
        }

        match (inner_op, op) {
            // `lshr(shl(x, c1), c2)` — SDWA `select_bits`, `v_mul_u32_u24`, `v_bfe_u32`.
            (Bin::Shl, Bin::LShr) => {
                let top = (!mask(w - c1)) & m;
                let top_zero = (kx.zero & top) == top;
                if c1 == c2 {
                    if top_zero {
                        return Some(x);
                    }
                    let keep = self.const_int(ty, w, mask(w - c1));
                    self.insts[i as usize].op = Op::Bin(Bin::And, x, keep);
                    *changed = true;
                    return None;
                }
                if top_zero {
                    if c1 < c2 {
                        let d = self.const_int(ty, w, u64::from(c2 - c1));
                        self.insts[i as usize].op = Op::Bin(Bin::LShr, x, d);
                    } else {
                        let d = self.const_int(ty, w, u64::from(c1 - c2));
                        self.insts[i as usize].op = Op::Bin(Bin::Shl, x, d);
                    }
                    *changed = true;
                    return None;
                }
            }
            // `ashr(shl(x, c1), c2)` — signed SDWA / `v_mul_i32_i24` / `v_bfe_i32`.
            (Bin::Shl, Bin::AShr) => {
                let sign_bit_pos = w - 1 - c1;
                let top_with_sign = (!mask(sign_bit_pos)) & m;
                let already_sext =
                    (kx.zero & top_with_sign) == top_with_sign || (kx.one & top_with_sign) == top_with_sign;
                if already_sext {
                    if c1 == c2 {
                        return Some(x);
                    }
                    if c1 < c2 {
                        let d = self.const_int(ty, w, u64::from(c2 - c1));
                        self.insts[i as usize].op = Op::Bin(Bin::AShr, x, d);
                    } else {
                        let d = self.const_int(ty, w, u64::from(c1 - c2));
                        self.insts[i as usize].op = Op::Bin(Bin::Shl, x, d);
                    }
                    *changed = true;
                    return None;
                }
                // Sign bit of `shl(x, c1)` is bit `w - 1 - c1` of `x`. If known 0,
                // `ashr(shl(x, c1), c1)` is `x & mask(w - c1)`.
                if c1 == c2 && ((kx.zero >> sign_bit_pos) & 1 != 0) {
                    let keep = self.const_int(ty, w, mask(w - c1));
                    self.insts[i as usize].op = Op::Bin(Bin::And, x, keep);
                    *changed = true;
                    return None;
                }
            }
            // `shl(lshr(x, c1), c2)` — alignment clearing / address scaling.
            (Bin::LShr, Bin::Shl) => {
                let low = mask(c1);
                let low_zero = (kx.zero & low) == low;
                if c1 == c2 {
                    if low_zero {
                        return Some(x);
                    }
                    let keep = self.const_int(ty, w, (!low) & m);
                    self.insts[i as usize].op = Op::Bin(Bin::And, x, keep);
                    *changed = true;
                    return None;
                }
                if low_zero {
                    if c1 < c2 {
                        let d = self.const_int(ty, w, u64::from(c2 - c1));
                        self.insts[i as usize].op = Op::Bin(Bin::Shl, x, d);
                    } else {
                        let d = self.const_int(ty, w, u64::from(c1 - c2));
                        self.insts[i as usize].op = Op::Bin(Bin::LShr, x, d);
                    }
                    *changed = true;
                    return None;
                }
            }
            _ => {}
        }
        None
    }

    /// Sub-phase B: Whole-function backward structural Demanded-Bits pass.
    ///
    /// Computes purely structural bit-demands (depending only on opcodes and
    /// immutable constant masks/shifts/widths) to fixed point across all CFG
    /// blocks and PHIs, paired with fresh SCKP `KnownBits`. Every rewrite
    /// preserves `(new & demanded[i]) == (old & demanded[i])`, and every
    /// `KnownBits` query inspects only bits within `demanded[operand]`, so
    /// simultaneous rewrites across the function are provably sound.
    fn dataflow_demanded_bits(&mut self) -> bool {
        let sckp = self.run_sckp();
        let effects = self.effects();
        let ni = self.insts.len();
        let mut demanded = vec![0u64; ni];
        let mut work: VecDeque<u32> = VecDeque::new();
        let mut in_work = vec![false; ni];

        let add_demand = |v: V, bits: u64, demanded: &mut [u64], work: &mut VecDeque<u32>, in_work: &mut [bool]| {
            let V::Inst(j) = v else { return };
            let Some(w) = self.ty_int_bits(self.insts[j as usize].ty) else {
                return;
            };
            let bits = bits & mask(w);
            let next = demanded[j as usize] | bits;
            if next != demanded[j as usize] {
                demanded[j as usize] = next;
                if !in_work[j as usize] {
                    in_work[j as usize] = true;
                    work.push_back(j);
                }
            }
        };

        let mark_all = |v: V, demanded: &mut [u64], work: &mut VecDeque<u32>, in_work: &mut [bool]| {
            if let Some(w) = self.int_width_of(v) {
                add_demand(v, mask(w), demanded, work, in_work);
            }
        };

        // Seed roots: terminators, side-effecting instructions, and non-integer ops
        // that consume integer operands (float casts, bitcasts, GEPs, loads, calls).
        for (b, blk) in self.blocks.iter().enumerate() {
            if !sckp.exec_block[b] {
                continue;
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                add_demand(c, 1, &mut demanded, &mut work, &mut in_work);
            }
            for &i in &blk.body {
                let inst = &self.insts[i as usize];
                let is_root = !self.is_pure_inst(&effects, i)
                    || self.ty_int_bits(inst.ty).is_none();
                if is_root {
                    if let Some(w) = self.ty_int_bits(inst.ty) {
                        add_demand(V::Inst(i), mask(w), &mut demanded, &mut work, &mut in_work);
                    }
                    for_each_operand(&inst.op, |v| {
                        mark_all(v, &mut demanded, &mut work, &mut in_work);
                    });
                }
            }
        }
        // Also seed integer instructions whose result is consumed by a non-integer
        // PHI or Extract (so float/aggregate PHIs keep their inputs demanded).
        for (b, blk) in self.blocks.iter().enumerate() {
            if !sckp.exec_block[b] {
                continue;
            }
            for &p in &blk.phis {
                if self.ty_int_bits(self.insts[p as usize].ty).is_none() {
                    for_each_operand(&self.insts[p as usize].op, |v| {
                        mark_all(v, &mut demanded, &mut work, &mut in_work);
                    });
                }
            }
        }

        // Backward fixed-point propagation (structural constant-only shrinking).
        while let Some(i) = work.pop_front() {
            in_work[i as usize] = false;
            let d = demanded[i as usize];
            if d == 0 {
                continue;
            }
            let inst = &self.insts[i as usize];
            let Some(w) = self.ty_int_bits(inst.ty) else {
                continue;
            };
            let m = mask(w);
            match &inst.op {
                Op::Bin(Bin::And, a, b) => {
                    let da = match self.int_k(*b) {
                        Some((_, k)) => d & k,
                        None => d,
                    };
                    let db = match self.int_k(*a) {
                        Some((_, k)) => d & k,
                        None => d,
                    };
                    add_demand(*a, da, &mut demanded, &mut work, &mut in_work);
                    add_demand(*b, db, &mut demanded, &mut work, &mut in_work);
                }
                Op::Bin(Bin::Or, a, b) => {
                    let da = match self.int_k(*b) {
                        Some((_, k)) => d & !k,
                        None => d,
                    };
                    let db = match self.int_k(*a) {
                        Some((_, k)) => d & !k,
                        None => d,
                    };
                    add_demand(*a, da, &mut demanded, &mut work, &mut in_work);
                    add_demand(*b, db, &mut demanded, &mut work, &mut in_work);
                }
                Op::Bin(Bin::Xor, a, b) => {
                    add_demand(*a, d, &mut demanded, &mut work, &mut in_work);
                    add_demand(*b, d, &mut demanded, &mut work, &mut in_work);
                }
                Op::Bin(Bin::Add | Bin::Sub | Bin::Mul, a, b) => {
                    let low = low_mask_up_to(d) & m;
                    add_demand(*a, low, &mut demanded, &mut work, &mut in_work);
                    add_demand(*b, low, &mut demanded, &mut work, &mut in_work);
                }
                Op::Bin(Bin::Shl, a, b) => {
                    if let Some((_, k)) = self.int_k(*b)
                        && k < u64::from(w)
                    {
                        add_demand(*a, d >> (k as u32), &mut demanded, &mut work, &mut in_work);
                    } else {
                        mark_all(*a, &mut demanded, &mut work, &mut in_work);
                        mark_all(*b, &mut demanded, &mut work, &mut in_work);
                    }
                }
                Op::Bin(Bin::LShr, a, b) => {
                    if let Some((_, k)) = self.int_k(*b)
                        && k < u64::from(w)
                    {
                        add_demand(
                            *a,
                            (d << (k as u32)) & m,
                            &mut demanded,
                            &mut work,
                            &mut in_work,
                        );
                    } else {
                        mark_all(*a, &mut demanded, &mut work, &mut in_work);
                        mark_all(*b, &mut demanded, &mut work, &mut in_work);
                    }
                }
                Op::Bin(Bin::AShr, a, b) => {
                    if let Some((_, k)) = self.int_k(*b)
                        && k < u64::from(w)
                    {
                        let k = k as u32;
                        let mut da = (d << k) & m;
                        if k > 0 && (d >> (w - k)) != 0 {
                            da |= 1u64 << (w - 1);
                        }
                        add_demand(*a, da, &mut demanded, &mut work, &mut in_work);
                    } else {
                        mark_all(*a, &mut demanded, &mut work, &mut in_work);
                        mark_all(*b, &mut demanded, &mut work, &mut in_work);
                    }
                }
                Op::Cast(Cast::Trunc, x) => {
                    add_demand(*x, d & m, &mut demanded, &mut work, &mut in_work);
                }
                Op::Cast(Cast::ZExt, x) => {
                    if let Some(fw) = self.int_width_of(*x) {
                        add_demand(*x, d & mask(fw), &mut demanded, &mut work, &mut in_work);
                    }
                }
                Op::Cast(Cast::SExt, x) => {
                    if let Some(fw) = self.int_width_of(*x) {
                        let mut dx = d & mask(fw);
                        if (d & !mask(fw)) != 0 {
                            dx |= 1u64 << (fw - 1);
                        }
                        add_demand(*x, dx, &mut demanded, &mut work, &mut in_work);
                    }
                }
                Op::Select(c, t, f) => {
                    add_demand(*c, 1, &mut demanded, &mut work, &mut in_work);
                    add_demand(*t, d, &mut demanded, &mut work, &mut in_work);
                    add_demand(*f, d, &mut demanded, &mut work, &mut in_work);
                }
                Op::Phi(incoming) => {
                    for &(v, _) in incoming {
                        add_demand(v, d, &mut demanded, &mut work, &mut in_work);
                    }
                }
                op => {
                    for_each_operand(op, |v| {
                        mark_all(v, &mut demanded, &mut work, &mut in_work);
                    });
                }
            }
        }

        // Apply sound demanded-bits simplifications.
        let mut repl: Vec<Option<V>> = vec![None; ni];
        let mut changed = false;

        for b in 0..self.blocks.len() {
            if !sckp.exec_block[b] {
                continue;
            }
            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                let d = demanded[i as usize];
                if d == 0 || !self.is_pure_inst(&effects, i) {
                    kept.push(i);
                    continue;
                }
                let ty = self.insts[i as usize].ty;
                let Some(w) = self.ty_int_bits(ty) else {
                    kept.push(i);
                    continue;
                };
                // If every demanded bit of `i` is already known constant, materialize it.
                if let Some(ki) = self.eval_kb(&sckp.lat, V::Inst(i))
                    && ((ki.zero | ki.one) & d) == d
                {
                    repl[i as usize] = Some(self.const_int(ty, w, ki.one & d));
                    changed = true;
                    continue;
                }
                let m = mask(w);
                let simplified = match self.insts[i as usize].op.clone() {
                    Op::Bin(Bin::And, a, b) => {
                        let ka = self.eval_kb(&sckp.lat, a);
                        let kb = self.eval_kb(&sckp.lat, b);
                        if kb.is_some_and(|k| (k.one & d) == d) {
                            Some(a)
                        } else if ka.is_some_and(|k| (k.one & d) == d) {
                            Some(b)
                        } else if ka.is_some_and(|k| (k.zero & d) == d)
                            || kb.is_some_and(|k| (k.zero & d) == d)
                        {
                            Some(self.const_int(ty, w, 0))
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::Or, a, b) => {
                        let ka = self.eval_kb(&sckp.lat, a);
                        let kb = self.eval_kb(&sckp.lat, b);
                        if kb.is_some_and(|k| (k.zero & d) == d) {
                            Some(a)
                        } else if ka.is_some_and(|k| (k.zero & d) == d) {
                            Some(b)
                        } else if ka.is_some_and(|k| (k.one & d) == d)
                            || kb.is_some_and(|k| (k.one & d) == d)
                        {
                            Some(self.const_int(ty, w, d))
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::Xor, a, b) => {
                        let ka = self.eval_kb(&sckp.lat, a);
                        let kb = self.eval_kb(&sckp.lat, b);
                        if kb.is_some_and(|k| (k.zero & d) == d) {
                            Some(a)
                        } else if ka.is_some_and(|k| (k.zero & d) == d) {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::Add, a, b) => {
                        let low = low_mask_up_to(d) & m;
                        let ka = self.eval_kb(&sckp.lat, a);
                        let kb = self.eval_kb(&sckp.lat, b);
                        if kb.is_some_and(|k| (k.zero & low) == low) {
                            Some(a)
                        } else if ka.is_some_and(|k| (k.zero & low) == low) {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::Sub, a, b) => {
                        let low = low_mask_up_to(d) & m;
                        let kb = self.eval_kb(&sckp.lat, b);
                        if kb.is_some_and(|k| (k.zero & low) == low) {
                            Some(a)
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::Shl, _, b) => {
                        if let Some((_, k)) = self.int_k(b)
                            && k < u64::from(w)
                            && (d >> (k as u32)) == 0
                        {
                            Some(self.const_int(ty, w, 0))
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::LShr, _, b) => {
                        if let Some((_, k)) = self.int_k(b)
                            && k < u64::from(w)
                            && ((d << (k as u32)) & m) == 0
                        {
                            Some(self.const_int(ty, w, 0))
                        } else {
                            None
                        }
                    }
                    Op::Bin(Bin::AShr, a, b) => {
                        if let Some((_, k)) = self.int_k(b)
                            && k < u64::from(w)
                        {
                            let k_u32 = k as u32;
                            if k_u32 > 0 && (d >> (w - k_u32)) == 0 {
                                // No sign-extended bits are demanded: rewrite to LShr.
                                self.insts[i as usize].op = Op::Bin(Bin::LShr, a, b);
                                changed = true;
                            }
                        }
                        None
                    }
                    Op::Cast(Cast::SExt, x) => {
                        if let Some(fw) = self.int_width_of(x)
                            && (d & !mask(fw)) == 0
                        {
                            // No sign-extended bits are demanded: rewrite to ZExt.
                            self.insts[i as usize].op = Op::Cast(Cast::ZExt, x);
                            changed = true;
                        }
                        None
                    }
                    _ => None,
                };
                if let Some(r) = simplified {
                    repl[i as usize] = Some(r);
                    changed = true;
                    continue;
                }
                kept.push(i);
            }
            self.blocks[b].body = kept;
        }

        if !changed {
            return false;
        }
        self.apply_replacements(&repl);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::bitcode::opt::{ALL, DATAFLOW, DEAD, DEFAULT, FACTS, FOLD, LOADS, REUSE};
    use crate::emit::bitcode::{Attr, Module};

    // ---- Exhaustive soundness oracle for KnownBits abstract transfers ----

    /// Enumerates every valid 4-bit `KnownBits` (3^4 = 81 abstract values).
    fn all_4bit_kb() -> Vec<KnownBits> {
        let mut out = Vec::with_capacity(81);
        for zero in 0..16u64 {
            for one in 0..16u64 {
                if zero & one == 0 {
                    out.push(KnownBits {
                        width: 4,
                        zero,
                        one,
                    });
                }
            }
        }
        assert_eq!(out.len(), 81);
        out
    }

    /// All concrete 4-bit values matching `kb`.
    fn concretize_4bit(kb: KnownBits) -> Vec<u64> {
        (0..16u64)
            .filter(|&x| (x & kb.one) == kb.one && ((!x) & kb.zero) == kb.zero)
            .collect()
    }

    fn assert_sound_4bit(kb: KnownBits, actual: u64, label: &str) {
        let a = actual & 0xF;
        assert_eq!(kb.zero & kb.one, 0, "{label}: contradictory {kb:?}");
        assert_eq!((kb.zero | kb.one) & !0xF, 0, "{label}: high bits set {kb:?}");
        assert_eq!(
            a & kb.one,
            kb.one,
            "{label}: concrete {a:#x} violated known-ones {kb:?}"
        );
        assert_eq!(
            (!a) & kb.zero,
            kb.zero,
            "{label}: concrete {a:#x} violated known-zeros {kb:?}"
        );
    }

    #[test]
    fn exhaustive_4bit_known_bits_transfers_are_100_percent_sound() {
        let all = all_4bit_kb();
        // Verify unary casts, signed/unsigned bounds, and `(1 << w) - 1` mask transfer on every 4-bit KnownBits.
        for &a in &all {
            let vals_a = concretize_4bit(a);
            if a.max_u() < 4 {
                let bfm = KnownBits {
                    width: 4,
                    zero: (!mask(a.max_u() as u32)) & 0xF,
                    one: mask(a.min_u() as u32) & 0xF,
                };
                for &y in &vals_a {
                    assert_sound_4bit(bfm, (1u64 << y) - 1, "bfm_mask");
                }
            }
            let true_min_u = *vals_a.iter().min().unwrap();
            let true_max_u = *vals_a.iter().max().unwrap();
            assert_eq!(a.min_u(), true_min_u);
            assert_eq!(a.max_u(), true_max_u);
            let true_min_s = vals_a.iter().map(|&x| signed(4, x)).min().unwrap();
            let true_max_s = vals_a.iter().map(|&x| signed(4, x)).max().unwrap();
            assert_eq!(a.min_s(), true_min_s);
            assert_eq!(a.max_s(), true_max_s);
            let za = a.zext(8);
            let sa = a.sext(8);
            let ta = a.trunc(2);
            for &x in &vals_a {
                assert_eq!(x & za.one, za.one);
                assert_eq!((!x & 0xFF) & za.zero, za.zero);
                let sx = (signed(4, x) as u64) & 0xFF;
                assert_eq!(sx & sa.one, sa.one);
                assert_eq!((!sx & 0xFF) & sa.zero, sa.zero);
                let tx = x & 3;
                assert_eq!(tx & ta.one, ta.one);
                assert_eq!((!tx & 3) & ta.zero, ta.zero);
            }
        }

        // Exhaustively verify every pair (81 x 81 = 6561) across all binary ops & comparisons.
        for &a in &all {
            let vals_a = concretize_4bit(a);
            for &b in &all {
                let vals_b = concretize_4bit(b);
                let kb_and = a.and(b);
                let kb_or = a.or(b);
                let kb_xor = a.xor(b);
                let kb_add = a.add_sub(b, false);
                let kb_sub = a.add_sub(b, true);
                let kb_mul = a.mul(b);
                let kb_shl = a.shift(Bin::Shl, b);
                let kb_lshr = a.shift(Bin::LShr, b);
                let kb_ashr = a.shift(Bin::AShr, b);
                let kb_udiv = a.udiv(b);
                let kb_urem = a.urem(b);
                let kb_sdiv = a.sdiv(b);
                let kb_join = a.join(b);

                for &x in &vals_a {
                    assert_sound_4bit(kb_join, x, "join_a");
                    for &y in &vals_b {
                        assert_sound_4bit(kb_join, y, "join_b");
                        assert_sound_4bit(kb_and, x & y, "and");
                        assert_sound_4bit(kb_or, x | y, "or");
                        assert_sound_4bit(kb_xor, x ^ y, "xor");
                        assert_sound_4bit(kb_add, x.wrapping_add(y), "add");
                        assert_sound_4bit(kb_sub, x.wrapping_sub(y), "sub");
                        assert_sound_4bit(kb_mul, x.wrapping_mul(y), "mul");
                        if y < 4 {
                            assert_sound_4bit(kb_shl, x << y, "shl");
                            assert_sound_4bit(kb_lshr, x >> y, "lshr");
                            assert_sound_4bit(kb_ashr, (signed(4, x) >> y) as u64, "ashr");
                        }
                        if let (Some(q), Some(r)) = (x.checked_div(y), x.checked_rem(y)) {
                            assert_sound_4bit(kb_udiv, q, "udiv");
                            assert_sound_4bit(kb_urem, r, "urem");
                            let (sx, sy) = (signed(4, x), signed(4, y));
                            if let Some(sq) = (sx != -8 || sy != -1)
                                .then(|| sx.checked_div(sy))
                                .flatten()
                            {
                                assert_sound_4bit(kb_sdiv, sq as u64, "sdiv");
                            }
                        }
                    }
                }

                // Exhaustive comparison soundness across all 10 integer predicates.
                for p in pred::EQ..=pred::SLE {
                    if let Some(claim) = a.cmp(p, b) {
                        for &x in &vals_a {
                            for &y in &vals_b {
                                let (sx, sy) = (signed(4, x), signed(4, y));
                                let actual = match p {
                                    pred::EQ => x == y,
                                    pred::NE => x != y,
                                    pred::IUGT => x > y,
                                    pred::IUGE => x >= y,
                                    pred::IULT => x < y,
                                    pred::IULE => x <= y,
                                    pred::SGT => sx > sy,
                                    pred::SGE => sx >= sy,
                                    pred::SLT => sx < sy,
                                    _ => sx <= sy,
                                };
                                assert_eq!(
                                    actual, claim,
                                    "cmp p={p} a={a:?} b={b:?} x={x} y={y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Targeted unit tests & soundness negative controls ----

    struct Kit {
        m: Module,
        b: usize,
        input: V,
        writes: V,
    }

    fn kit() -> Kit {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let input = m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        Kit {
            m,
            b,
            input,
            writes,
        }
    }

    fn stores(m: &Module, b: usize) -> Vec<Vec<V>> {
        m.blocks[b]
            .body
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(g, args) if m.funcs[*g as usize].name.contains("Store") => {
                    Some(args.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn bfe_dynamic_width_and_redundant_clamp_collapse() {
        let Kit {
            mut m,
            b,
            input,
            writes,
        } = kit();
        let i32t = m.int(32);
        let fty = m.func_ty(i32t, vec![i32t, i32t, i32t]);
        m.ptr(fty);
        let bin32 = m.function("dx.op.binary.i32", fty, Some(Attr::ReadNone), false);

        let (k0, k1, k31, k32, k2047, ones) = (
            m.i32(0),
            m.i32(1),
            m.i32(31),
            m.i32(32),
            m.i32(2047),
            m.i32(u32::MAX),
        );
        let a = m.call(b, input, vec![k0, k0]);
        let off = m.call(b, input, vec![k1, k0]);
        let raw_w = m.call(b, input, vec![k1, k1]);

        // Exact pattern emitted by `lower.rs` for `v_bfe_u32`:
        let w = m.bin(b, Bin::And, raw_w, k31);
        let off_m = m.bin(b, Bin::And, off, k31);
        let shifted = m.bin(b, Bin::LShr, a, off_m);
        let full = m.cmp(b, pred::IUGE, w, k32); // provably false!
        let w2 = m.bin(b, Bin::And, w, k31); // provably redundant -> w!
        let mask_lo = m.bin(b, Bin::Shl, k1, w2);
        let mask_lo = m.bin(b, Bin::Sub, mask_lo, k1);
        let mask = m.select(b, full, ones, mask_lo); // provably mask_lo!
        let extracted = m.bin(b, Bin::And, shifted, mask);

        // Redundant `Tag::Clamp` (`UMin(slot, 2047)`) on a value masked to 10 bits.
        let k1023 = m.i32(1023);
        let slot = m.bin(b, Bin::And, a, k1023);
        let op_umin = m.i32(40);
        let clamped = m.call(b, bin32, vec![op_umin, slot, k2047]);

        m.call(b, writes, vec![extracted, clamped]);
        m.terminate(b, Term::Ret);

        let before = m.blocks[b].body.len();
        m.optimise(DATAFLOW | DEAD);
        let after = m.blocks[b].body.len();
        assert!(
            after <= before - 4,
            "expected >=4 insts removed (full, w2, mask select, clamp): {before} -> {after}"
        );
        assert_eq!(stores(&m, b)[0][1], slot, "redundant UMin clamp eliminated");
        m.write();
    }

    #[test]
    fn exec_mask_nested_select_and_cross_block_bitfield_collapse() {
        let Kit {
            mut m,
            b,
            input,
            writes,
        } = kit();
        let i32t = m.int(32);
        let (k0, k1, k8, k16, kff, kffff) = (
            m.i32(0),
            m.i32(1),
            m.i32(8),
            m.i32(16),
            m.i32(0xFF),
            m.i32(0xFFFF),
        );
        let x = m.call(b, input, vec![k0, k0]);
        let y = m.call(b, input, vec![k1, k0]);
        let z = m.call(b, input, vec![k1, k1]);

        let g_outer = m.cmp(b, pred::IULT, x, k16);
        let y_nz = m.cmp(b, pred::NE, y, k0);
        let g_inner = m.bin(b, Bin::And, g_outer, y_nz);

        // Nested guarded write: `select(g_outer, new, select(g_outer & c2, mid, old))`
        // collapses to `select(g_outer, new, old)` because `!g_outer => !g_inner`!
        let mid_sel = m.select(b, g_inner, y, z);
        let outer_sel = m.select(b, g_outer, x, mid_sel);

        // Cross-block diamond where both arms produce values fitting in 16 bits,
        // joined by a PHI, then passed through `v_mul_u32_u24`'s `(p << 8) >>u 8`
        // and packed with `(z << 16)` whose low 16 bits are demanded!
        let (t_blk, f_blk, join) = (m.new_block(), m.new_block(), m.new_block());
        m.terminate(b, Term::CondBr(g_outer, t_blk, f_blk));
        // Inside t_blk, x <u 16 is known! So `x <u 32` is true and `x & 31 == x`!
        let k32 = m.i32(32);
        let redundant_cmp = m.cmp(t_blk, pred::IULT, x, k32);
        let t_val = m.select(t_blk, redundant_cmp, x, y);
        m.terminate(t_blk, Term::Br(join));
        let f_val = m.bin(f_blk, Bin::And, y, kff);
        m.terminate(f_blk, Term::Br(join));

        let joined = m.phi(join, i32t);
        m.set_phi(joined, vec![(t_val, t_blk), (f_val, f_blk)]);
        // `(joined << 8) >>u 8` is completely redundant because `joined <= 255`!
        let shl8 = m.bin(join, Bin::Shl, joined, k8);
        let u24 = m.bin(join, Bin::LShr, shl8, k8);
        // Pack `z << 16 | u24`, then demand only low 16 bits (`& 0xFFFF`):
        // DemandedBits drops `z << 16 | ...` completely!
        let hi = m.bin(join, Bin::Shl, z, k16);
        let packed = m.bin(join, Bin::Or, hi, u24);
        let low16 = m.bin(join, Bin::And, packed, kffff);

        m.call(join, writes, vec![outer_sel, low16]);
        m.terminate(join, Term::Ret);

        m.optimise(DATAFLOW | DEAD);
        // `mid_sel` must be dead, `outer_sel` must select directly between `x` and `z`!
        let st = stores(&m, join);
        let V::Inst(sel_i) = st[0][0] else {
            panic!("expected select");
        };
        assert!(
            matches!(m.insts[sel_i as usize].op, Op::Select(c, t, f) if c == g_outer && t == x && f == z),
            "nested exec guard select should collapse to select(g_outer, x, z), got {:?}",
            m.insts[sel_i as usize].op
        );
        // `low16` must collapse all the way to `joined`!
        assert_eq!(st[0][1], joined, "u24 shifts + packed high bits + &0xFFFF all collapse to joined PHI");
        m.write();
    }

    #[test]
    fn space3_lds_store_is_never_forwarded_to_load() {
        // Soundness rule #1 (#678): wave-ordered barrierless groupshared LDS
        // reads must observe concurrent lane stores and NEVER forward store->load.
        let Kit {
            mut m,
            b,
            input,
            writes,
        } = kit();
        let i32t = m.int(32);
        let arr = m.array(64, i32t);
        let lds = m.global("lds", arr, 3);
        let (k0, k1) = (m.i32(0), m.i32(1));
        let lane = m.call(b, input, vec![k0, k0]);
        let ptr = m.element(b, lds, k0);
        m.store(b, ptr, lane);
        let read_back = m.load(b, ptr);
        m.call(b, writes, vec![read_back, k1]);
        m.terminate(b, Term::Ret);

        m.optimise(ALL);
        let st = stores(&m, b);
        assert_eq!(
            st[0][0], read_back,
            "space-3 LDS load MUST remain live and never forward from store"
        );
        assert!(
            m.blocks[b].body.iter().any(|&i| matches!(m.insts[i as usize].op, Op::Load(_))),
            "LDS load instruction must still exist"
        );
        m.write();
    }

    #[test]
    fn get_dimensions_size_query_is_never_coalesced_or_reused() {
        // Soundness rule #2 (Belobug #324): `dx.op.getDimensions` must NEVER
        // be coalesced/reused across or within blocks.
        let Kit {
            mut m,
            b,
            input,
            writes,
        } = kit();
        let i32t = m.int(32);
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let dims = m.function("dx.op.getDimensions", fty, Some(Attr::ReadOnly), false);
        let (k0, k1) = (m.i32(0), m.i32(1));
        let x = m.call(b, input, vec![k0, k0]);
        let c = m.cmp(b, pred::NE, x, k0);
        let (then_b, else_b) = (m.new_block(), m.new_block());
        m.terminate(b, Term::CondBr(c, then_b, else_b));

        let d1 = m.call(b, dims, vec![x, k0]);
        let d2 = m.call(b, dims, vec![x, k0]);
        m.call(b, writes, vec![d1, d2]);

        let d3 = m.call(then_b, dims, vec![x, k0]);
        m.call(then_b, writes, vec![d3, k1]);
        m.terminate(then_b, Term::Ret);
        m.terminate(else_b, Term::Ret);

        m.optimise(ALL);
        assert_eq!(stores(&m, b), [[d1, d2]], "intra-block getDimensions must not coalesce");
        assert_eq!(stores(&m, then_b), [[d3, k1]], "cross-block getDimensions must not coalesce");
        m.write();
    }

    #[test]
    fn sccp_kills_unreachable_loop_backedge_phi_cycle() {
        let Kit {
            mut m,
            b,
            writes,
            ..
        } = kit();
        let i1t = m.int(1);
        let i32t = m.int(32);
        let (k0, k7, k99) = (m.i32(0), m.i32(7), m.i32(99));
        let (yes, no) = (m.bool(true), m.bool(false));
        let (head, dead_arm, exit) = (m.new_block(), m.new_block(), m.new_block());
        m.terminate(b, Term::Br(head));

        let flag = m.phi(head, i1t);
        let acc = m.phi(head, i32t);
        m.terminate(head, Term::CondBr(flag, dead_arm, exit));

        let next_acc = m.bin(dead_arm, Bin::Add, acc, k99);
        m.terminate(dead_arm, Term::Br(head));
        m.set_phi(flag, vec![(no, b), (yes, dead_arm)]);
        m.set_phi(acc, vec![(k7, b), (next_acc, dead_arm)]);

        m.call(exit, writes, vec![k0, acc]);
        m.terminate(exit, Term::Ret);

        m.optimise(DATAFLOW | DEAD);
        assert!(matches!(m.blocks[head].term, Some(Term::Br(t)) if t == exit));
        assert!(m.blocks[dead_arm].body.is_empty());
        assert_eq!(stores(&m, exit), [[k0, k7]]);
        m.write();
    }

    #[test]
    fn zext_trunc_64bit_pack_and_demanded_bits_collapse() {
        // Pattern from `s_lshl_b64` / 64-bit address packing where only one 32-bit half is used.
        let Kit {
            mut m,
            b,
            input,
            writes,
        } = kit();
        let (i32t, i64t) = (m.int(32), m.int(64));
        let (k0, k1) = (m.i32(0), m.i32(1));
        let lo = m.call(b, input, vec![k0, k0]);
        let hi = m.call(b, input, vec![k1, k0]);
        let zlo = m.cast(b, Cast::ZExt, lo, i64t);
        let zhi = m.cast(b, Cast::ZExt, hi, i64t);
        let sh32 = m.int_const(i64t, 32);
        let hi_sh = m.bin(b, Bin::Shl, zhi, sh32);
        let packed = m.bin(b, Bin::Or, zlo, hi_sh);
        let back_lo = m.cast(b, Cast::Trunc, packed, i32t);
        let back_hi_64 = m.bin(b, Bin::LShr, packed, sh32);
        let back_hi = m.cast(b, Cast::Trunc, back_hi_64, i32t);
        m.call(b, writes, vec![back_lo, back_hi]);
        m.terminate(b, Term::Ret);

        m.optimise(DEFAULT | DATAFLOW);
        assert_eq!(stores(&m, b), [[lo, hi]]);
        assert!(!m.usage().int64, "all i64 instructions should be eliminated");
        m.write();
    }

    #[test]
    fn sckp_2hop_sub_shl_loop_widening_does_not_miscompile() {
        // Regression test (2-hop SSA dependency in `run_sckp` for `sub(shl(base, amt), 1)`):
        // On iter 0, `%base = 1` gives `%msk = (1 << amt) - 1` (bit 31 known 0),
        // while `%sh = shl %base, %amt` is already `unknown(32)` because `%amt`
        // has 5 unknown bits. When the backedge widens `%base` to `{0, 1}`,
        // `%msk` must be re-enqueued so `%hi = lshr %msk, 31` is NOT folded to 0.
        let build = || {
            let Kit {
                mut m,
                b,
                input,
                writes,
            } = kit();
            let i32t = m.int(32);
            let (k0, k1, k31) = (m.i32(0), m.i32(1), m.i32(31));
            let x = m.call(b, input, vec![k0, k0]);
            let amt = m.bin(b, Bin::And, x, k31);

            let (head, exit) = (m.new_block(), m.new_block());
            m.terminate(b, Term::Br(head));

            let base = m.phi(head, i32t);
            let iter = m.phi(head, i32t);
            let sh = m.bin(head, Bin::Shl, base, amt);
            let msk = m.bin(head, Bin::Sub, sh, k1);
            let hi = m.bin(head, Bin::LShr, msk, k31);
            m.call(head, writes, vec![iter, hi]);

            let k2 = m.i32(2);
            let next_iter = m.bin(head, Bin::Add, iter, k1);
            let cont = m.cmp(head, pred::IULT, next_iter, k2);
            m.set_phi(base, vec![(k1, b), (k0, head)]);
            m.set_phi(iter, vec![(k0, b), (next_iter, head)]);
            m.terminate(head, Term::CondBr(cont, head, exit));
            m.terminate(exit, Term::Ret);
            m
        };

        let plain = build();
        for inp in [&[0u32][..], &[5], &[31], &[0xDEAD_BEEF]] {
            let want = run_ref(&plain, inp);
            assert_eq!(want, vec![vec![0, 0], vec![1, 1]]);
            for passes in [DATAFLOW, DATAFLOW | DEAD, DEFAULT | DATAFLOW, ALL] {
                let mut m = build();
                m.optimise(passes);
                assert_eq!(
                    run_ref(&m, inp),
                    want,
                    "2-hop sub(shl) loop miscompiled under passes={passes} inp={inp:?}"
                );
                m.write();
            }
        }
    }

    #[test]
    fn sckp_2hop_bitcast_roundtrip_loop_widening_does_not_miscompile() {
        // Regression test (2-hop SSA dependency in `run_sckp` for `bitcast(bitcast(inner))`):
        // `%f = bitcast i32 %p to float` stays `Lattice::Other` across loop widening,
        // so `%back = bitcast float %f to i32` must subscribe directly to `%p`.
        let build = || {
            let Kit {
                mut m,
                b,
                input,
                writes,
            } = kit();
            let (i32t, f32t) = (m.int(32), m.float());
            let (k0, k1, k5) = (m.i32(0), m.i32(1), m.i32(5));
            let x = m.call(b, input, vec![k0, k0]);

            let (head, exit) = (m.new_block(), m.new_block());
            m.terminate(b, Term::Br(head));

            let p = m.phi(head, i32t);
            let iter = m.phi(head, i32t);
            let f = m.cast(head, Cast::Bitcast, p, f32t);
            let back = m.cast(head, Cast::Bitcast, f, i32t);
            m.call(head, writes, vec![iter, back]);

            let k3 = m.i32(3);
            let next_p = m.bin(head, Bin::Add, p, x);
            let next_iter = m.bin(head, Bin::Add, iter, k1);
            let cont = m.cmp(head, pred::IULT, next_iter, k3);
            m.set_phi(p, vec![(k5, b), (next_p, head)]);
            m.set_phi(iter, vec![(k0, b), (next_iter, head)]);
            m.terminate(head, Term::CondBr(cont, head, exit));
            m.terminate(exit, Term::Ret);
            m
        };

        let plain = build();
        for inp in [&[1u32][..], &[7], &[0x1234_5678]] {
            let want = run_ref(&plain, inp);
            assert_eq!(want.len(), 3);
            assert_ne!(want[0][1], want[1][1]);
            for passes in [DATAFLOW, DATAFLOW | DEAD, DEFAULT | DATAFLOW, ALL] {
                let mut m = build();
                m.optimise(passes);
                assert_eq!(
                    run_ref(&m, inp),
                    want,
                    "2-hop bitcast loop miscompiled under passes={passes} inp={inp:?}"
                );
                m.write();
            }
        }
    }

    #[test]
    fn rdna4_tag_clamp_is_never_merged_or_folded_onto_another_clamp() {
        // Soundness rule (Belobug #324 / #258):
        // - Two per-handle `Tag::Clamp` `UMin(x, 2047)` calls must stay distinct
        //   (never CSE'd by REUSE or merged by DATAFLOW).
        // - Chained `Tag::Clamp` (`UMin(c1, 2047)` and `UMin(c1, 4095)`) must
        //   never fold onto `c1` (which would share one clamp across handles).
        // - Non-strict boundary `Tag::Clamp` (`UMin(x & 2047, 2047)`) stays intact.
        // - Strictly smaller non-clamp provenance `Tag::Clamp` (`UMin(x & 1023, 2047)`)
        //   still collapses cleanly to `x & 1023`.
        for passes in [REUSE, DATAFLOW | DEAD, DEFAULT | DATAFLOW, ALL] {
            let Kit {
                mut m,
                b,
                input,
                writes,
            } = kit();
            let i32t = m.int(32);
            let fty = m.func_ty(i32t, vec![i32t, i32t, i32t]);
            m.ptr(fty);
            let bin32 = m.function("dx.op.binary.i32", fty, Some(Attr::ReadNone), false);
            let (k0, k40, k1023, k2047, k4095) = (
                m.i32(0),
                m.i32(40),
                m.i32(1023),
                m.i32(2047),
                m.i32(4095),
            );
            let x = m.call(b, input, vec![k0, k0]);

            let c1 = m.call(b, bin32, vec![k40, x, k2047]);
            m.tag(c1, Tag::Clamp);
            let c2 = m.call(b, bin32, vec![k40, x, k2047]);
            m.tag(c2, Tag::Clamp);
            let c3 = m.call(b, bin32, vec![k40, c1, k2047]);
            m.tag(c3, Tag::Clamp);
            let c4 = m.call(b, bin32, vec![k40, c1, k4095]);
            m.tag(c4, Tag::Clamp);

            let eq_bound = m.bin(b, Bin::And, x, k2047);
            let c5 = m.call(b, bin32, vec![k40, eq_bound, k2047]);
            m.tag(c5, Tag::Clamp);

            let strict_small = m.bin(b, Bin::And, x, k1023);
            let c6 = m.call(b, bin32, vec![k40, strict_small, k2047]);
            m.tag(c6, Tag::Clamp);

            m.call(b, writes, vec![c1, c2]);
            m.call(b, writes, vec![c3, c4]);
            m.call(b, writes, vec![c5, c6]);
            m.terminate(b, Term::Ret);

            m.optimise(passes);
            let st = stores(&m, b);
            assert_eq!(st[0], [c1, c2], "passes={passes}: twin Tag::Clamps must stay distinct");
            assert_eq!(st[1], [c3, c4], "passes={passes}: chained Tag::Clamps must not fold onto c1");
            assert_eq!(st[2][0], c5, "passes={passes}: non-strict boundary Tag::Clamp stays intact");
            if passes & DATAFLOW != 0 {
                assert_eq!(
                    st[2][1], strict_small,
                    "passes={passes}: strictly smaller non-clamp provenance collapses"
                );
            }
            m.write();
        }
    }

    #[test]
    fn symmetric_and_or_mask_strip_preserves_semantics() {
        // Regression test: `(x & 0xFF) & (y & 0xFF)` must only strip ONE side
        // (becoming `x & (y & 0xFF)`), never both sides (`x & y`). Same for
        // `(x | 0xFF00) | (y | 0xFF00)`.
        let build = || {
            let Kit {
                mut m,
                b,
                input,
                writes,
            } = kit();
            let (k0, k1, kff, kff00, k0ff0) = (
                m.i32(0),
                m.i32(1),
                m.i32(0xFF),
                m.i32(0xFF00),
                m.i32(0x0FF0),
            );
            let x = m.call(b, input, vec![k0, k0]);
            let y = m.call(b, input, vec![k1, k0]);
            let x_and = m.bin(b, Bin::And, x, kff);
            let y_and = m.bin(b, Bin::And, y, kff);
            let both_and = m.bin(b, Bin::And, x_and, y_and);

            let x_or = m.bin(b, Bin::Or, x, kff00);
            let y_or = m.bin(b, Bin::Or, y, kff00);
            let both_or = m.bin(b, Bin::Or, x_or, y_or);

            let x_mid = m.bin(b, Bin::And, x, k0ff0);
            let partial_and = m.bin(b, Bin::And, x_mid, y_and);

            m.call(b, writes, vec![both_and, both_or]);
            m.call(b, writes, vec![partial_and, k0]);
            m.terminate(b, Term::Ret);
            (m, b)
        };

        let (base_m, base_b) = build();
        let vectors: &[[u32; 2]] = &[
            [0xFFFF_FFFF, 0xFFFF_FFFF],
            [0x1234_5678, 0x9ABC_DEF0],
            [0x0000_0000, 0x0000_0000],
            [0xDEAD_BEEF, 0xCAFE_BABE],
        ];
        for passes in [DATAFLOW, DATAFLOW | DEAD, DEFAULT | DATAFLOW, ALL] {
            let (mut opt_m, _) = build();
            opt_m.optimise(passes);
            for v in vectors {
                assert_eq!(
                    run_ref(&opt_m, v),
                    run_ref(&base_m, v),
                    "passes={passes} inputs={v:#x?}"
                );
            }
            opt_m.write();
        }
        // Verify one redundant mask WAS stripped from each symmetric pair.
        let (mut opt_m, opt_b) = build();
        opt_m.optimise(DATAFLOW | DEAD);
        assert!(
            opt_m.blocks[opt_b].body.len() < base_m.blocks[base_b].body.len(),
            "expected redundant symmetric mask to be stripped"
        );
    }

    // ---- Randomized whole-function CFG/bitfield/exec-guard equivalence oracle ----

    fn run_ref(m: &Module, inputs: &[u32]) -> Vec<Vec<u64>> {
        let mut vals: Vec<Option<u64>> = vec![None; m.insts.len()];
        let mut trace = Vec::new();
        let eval = |vals: &[Option<u64>], v: V| -> u64 {
            match v {
                V::Inst(i) => vals[i as usize].expect("SSA value defined before use"),
                V::Const(c) => match (&m.consts[c as usize].1, m.ty_int_bits(m.consts[c as usize].0)) {
                    (Const::Int(x), Some(bits)) => (*x as u64) & mask(bits),
                    (Const::Float(bits), _) => u64::from(*bits),
                    _ => 0,
                },
                _ => panic!("unexpected global/func operand"),
            }
        };
        let (mut b, mut from) = (0usize, usize::MAX);
        for _ in 0..4000 {
            let blk = &m.blocks[b];
            let joined: Vec<(u32, u64)> = blk
                .phis
                .iter()
                .map(|&p| {
                    let Op::Phi(incoming) = &m.insts[p as usize].op else {
                        panic!("expected phi");
                    };
                    let v = incoming
                        .iter()
                        .find(|x| x.1 == from)
                        .expect("incoming edge for predecessor");
                    (p, eval(&vals, v.0))
                })
                .collect();
            for (p, x) in joined {
                vals[p as usize] = Some(x);
            }
            for &i in &blk.body {
                let inst = &m.insts[i as usize];
                let bits = m.ty_int_bits(inst.ty);
                let r = match &inst.op {
                    Op::Bin(op, x, y) => {
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        let b_w = bits.expect("int binop");
                        let r = match op {
                            Bin::Add => x.wrapping_add(y),
                            Bin::Sub => x.wrapping_sub(y),
                            Bin::Mul => x.wrapping_mul(y),
                            Bin::And => x & y,
                            Bin::Or => x | y,
                            Bin::Xor => x ^ y,
                            Bin::Shl => x << y,
                            Bin::LShr => x >> y,
                            Bin::AShr => (signed(b_w, x) >> y) as u64,
                            Bin::UDiv => x / y,
                            Bin::URem => x % y,
                            Bin::SDiv => (signed(b_w, x) / signed(b_w, y)) as u64,
                        };
                        r & mask(b_w)
                    }
                    Op::Cmp(p, x, y) => {
                        let w = m.int_width_of(*x).expect("int cmp");
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        let (sx, sy) = (signed(w, x), signed(w, y));
                        let holds = match *p {
                            pred::EQ => x == y,
                            pred::NE => x != y,
                            pred::IUGT => x > y,
                            pred::IUGE => x >= y,
                            pred::IULT => x < y,
                            pred::IULE => x <= y,
                            pred::SGT => sx > sy,
                            pred::SGE => sx >= sy,
                            pred::SLT => sx < sy,
                            _ => sx <= sy,
                        };
                        u64::from(holds)
                    }
                    Op::Select(c, t, f) => {
                        if eval(&vals, *c) != 0 {
                            eval(&vals, *t)
                        } else {
                            eval(&vals, *f)
                        }
                    }
                    Op::Cast(op, x) => {
                        let fw = m.int_width_of(*x);
                        let x = eval(&vals, *x);
                        match (op, fw, bits) {
                            (Cast::Bitcast, _, _) => x,
                            (Cast::ZExt, _, _) => x,
                            (Cast::SExt, Some(f), Some(t)) => (signed(f, x) as u64) & mask(t),
                            (Cast::Trunc, _, Some(t)) => x & mask(t),
                            _ => panic!("unsupported cast in oracle"),
                        }
                    }
                    Op::Call(f, args) => {
                        let a: Vec<u64> = args.iter().map(|&v| eval(&vals, v)).collect();
                        match m.funcs[*f as usize].name.as_str() {
                            "dx.op.loadInput.i32" => {
                                u64::from(inputs[(a[0] ^ a[1]) as usize % inputs.len()])
                            }
                            "dx.op.binary.i32" => match a[0] {
                                37 => (signed(32, a[1]).max(signed(32, a[2])) as u64) & mask(32),
                                38 => (signed(32, a[1]).min(signed(32, a[2])) as u64) & mask(32),
                                39 => a[1].max(a[2]),
                                40 => a[1].min(a[2]),
                                _ => panic!("unexpected binary.i32 opcode"),
                            },
                            "dx.op.unary.i32" => u64::from((a[1] as u32).reverse_bits()),
                            "dx.op.unaryBits.i32" => match a[0] {
                                31 => u64::from((a[1] as u32).count_ones()),
                                32 => u64::from(if a[1] == 0 {
                                    u32::MAX
                                } else {
                                    (a[1] as u32).trailing_zeros()
                                }),
                                33 => u64::from(if a[1] == 0 {
                                    u32::MAX
                                } else {
                                    (a[1] as u32).leading_zeros()
                                }),
                                _ => panic!("unexpected unaryBits opcode"),
                            },
                            _ => {
                                trace.push(a);
                                0
                            }
                        }
                    }
                    _ => panic!("unexpected op in oracle"),
                };
                vals[i as usize] = Some(r);
            }
            from = b;
            b = match blk.term.expect("term") {
                Term::Ret => return trace,
                Term::Br(t) => t,
                Term::CondBr(c, t, f) => {
                    if eval(&vals, c) != 0 {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        panic!("oracle hit step limit");
    }

    struct RandGen {
        m: Module,
        rng: u64,
        cur: usize,
        regs: Vec<V>,
        conds: Vec<V>,
        writes: V,
        bin32: V,
        unary32: V,
        unary_bits: V,
    }

    impl RandGen {
        fn next(&mut self, n: u64) -> u64 {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            (self.rng >> 11) % n
        }

        fn reg(&mut self) -> V {
            let idx = self.next(self.regs.len() as u64) as usize;
            self.regs[idx]
        }

        fn val(&mut self) -> V {
            if self.next(4) == 0 {
                let ks = [
                    0u32, 1, 7, 8, 15, 16, 31, 32, 0xFF, 0xFFFF, 0x7FFF_FFFF, 0x8000_0000,
                    u32::MAX,
                ];
                let k = ks[self.next(ks.len() as u64) as usize];
                self.m.i32(k)
            } else {
                self.reg()
            }
        }

        fn cond(&mut self) -> V {
            let c = self.cur;
            let ncond = self.conds.len() as u64;
            let made = match self.next(6) {
                0 if !self.conds.is_empty() => {
                    let i = self.next(ncond) as usize;
                    return self.conds[i];
                }
                1 if self.conds.len() > 1 => {
                    let ia = self.next(ncond) as usize;
                    let ib = self.next(ncond) as usize;
                    let (a, b) = (self.conds[ia], self.conds[ib]);
                    let op = [Bin::And, Bin::Or, Bin::Xor][self.next(3) as usize];
                    self.m.bin(c, op, a, b)
                }
                2 if !self.conds.is_empty() => {
                    let ia = self.next(ncond) as usize;
                    let a = self.conds[ia];
                    let t = self.m.bool(true);
                    self.m.bin(c, Bin::Xor, a, t)
                }
                3 if self.conds.len() > 2 => {
                    let is = self.next(ncond) as usize;
                    let it = self.next(ncond) as usize;
                    let if_ = self.next(ncond) as usize;
                    let (s, t, f) = (self.conds[is], self.conds[it], self.conds[if_]);
                    self.m.select(c, s, t, f)
                }
                _ => {
                    let p = pred::EQ + self.next(10) as u8;
                    let (x, y) = (self.reg(), self.val());
                    self.m.cmp(c, p, x, y)
                }
            };
            self.conds.push(made);
            made
        }

        fn stmt(&mut self) {
            let c = self.cur;
            let dst = self.next(self.regs.len() as u64) as usize;
            let (i8t, i32t, i64t) = (self.m.int(8), self.m.int(32), self.m.int(64));
            let r = match self.next(13) {
                0..=2 => {
                    let op = [Bin::Add, Bin::Sub, Bin::Mul, Bin::And, Bin::Or, Bin::Xor]
                        [self.next(6) as usize];
                    let (x, y) = (self.val(), self.val());
                    self.m.bin(c, op, x, y)
                }
                3 => {
                    // RDNA2 shift with `& 31` masking.
                    let op = [Bin::Shl, Bin::LShr, Bin::AShr][self.next(3) as usize];
                    let x = self.val();
                    let sh = if self.next(2) == 0 {
                        let k = self.next(32) as u32;
                        self.m.i32(k)
                    } else {
                        let s = self.val();
                        let k31 = self.m.i32(31);
                        self.m.bin(c, Bin::And, s, k31)
                    };
                    self.m.bin(c, op, x, sh)
                }
                4 => {
                    // `v_bfe_u32` style dynamic extract.
                    let (a, off, w) = (self.val(), self.val(), self.val());
                    let (k31, k32, k1, ones) = (
                        self.m.i32(31),
                        self.m.i32(32),
                        self.m.i32(1),
                        self.m.i32(u32::MAX),
                    );
                    let w = self.m.bin(c, Bin::And, w, k31);
                    let off = self.m.bin(c, Bin::And, off, k31);
                    let sh = self.m.bin(c, Bin::LShr, a, off);
                    let full = self.m.cmp(c, pred::IUGE, w, k32);
                    let w2 = self.m.bin(c, Bin::And, w, k31);
                    let msk = self.m.bin(c, Bin::Shl, k1, w2);
                    let msk = self.m.bin(c, Bin::Sub, msk, k1);
                    let msk = self.m.select(c, full, ones, msk);
                    self.m.bin(c, Bin::And, sh, msk)
                }
                5 => {
                    // `v_mul_u32_u24` / SDWA shift-pair.
                    let x = self.val();
                    let sh_v = [8, 16, 24][self.next(3) as usize];
                    let sh = self.m.i32(sh_v);
                    let op2 = if self.next(2) == 0 { Bin::LShr } else { Bin::AShr };
                    let t = self.m.bin(c, Bin::Shl, x, sh);
                    self.m.bin(c, op2, t, sh)
                }
                6 => {
                    // `v_bfi_b32` bitfield insert.
                    let (m, a, b) = (self.val(), self.val(), self.val());
                    let ones = self.m.i32(u32::MAX);
                    let nm = self.m.bin(c, Bin::Xor, m, ones);
                    let x = self.m.bin(c, Bin::And, m, a);
                    let y = self.m.bin(c, Bin::And, nm, b);
                    self.m.bin(c, Bin::Or, x, y)
                }
                7 | 8 => {
                    // Guarded `exec` select (possibly chained with existing register).
                    let g = self.cond();
                    let x = self.val();
                    self.m.select(c, g, x, self.regs[dst])
                }
                9 => {
                    // Pure DXIL clamp / min / max (`dx.op.binary.i32`).
                    let op_k = 37 + self.next(4) as u32;
                    let op = self.m.i32(op_k);
                    let (x, y) = (self.val(), self.val());
                    self.m.call(c, self.bin32, vec![op, x, y])
                }
                10 => {
                    // Pure DXIL bit intrinsics (`Bfrev`, `Countbits`, `FirstbitLo`, `FirstbitHi`).
                    let x = self.val();
                    match self.next(4) {
                        0 => {
                            let op = self.m.i32(30);
                            self.m.call(c, self.unary32, vec![op, x])
                        }
                        k => {
                            let op = self.m.i32(30 + k as u32);
                            self.m.call(c, self.unary_bits, vec![op, x])
                        }
                    }
                }
                11 => {
                    // `i32 -> i8 -> i32` or `i32 -> i64 -> i32` cast roundtrip.
                    let x = self.val();
                    if self.next(2) == 0 {
                        let t = self.m.cast(c, Cast::Trunc, x, i8t);
                        let ext = if self.next(2) == 0 { Cast::ZExt } else { Cast::SExt };
                        self.m.cast(c, ext, t, i32t)
                    } else {
                        let y = self.val();
                        let zx = self.m.cast(c, Cast::ZExt, x, i64t);
                        let zy = self.m.cast(c, Cast::ZExt, y, i64t);
                        let k32 = self.m.int_const(i64t, 32);
                        let hi = self.m.bin(c, Bin::Shl, zy, k32);
                        let packed = self.m.bin(c, Bin::Or, zx, hi);
                        let pick = if self.next(2) == 0 {
                            packed
                        } else {
                            self.m.bin(c, Bin::LShr, packed, k32)
                        };
                        self.m.cast(c, Cast::Trunc, pick, i32t)
                    }
                }
                _ => {
                    // Boolean zext/sext or bounded urem/udiv.
                    if self.next(2) == 0 {
                        let g = self.cond();
                        let ext = if self.next(2) == 0 { Cast::ZExt } else { Cast::SExt };
                        self.m.cast(c, ext, g, i32t)
                    } else {
                        let x = self.val();
                        let d_v = [1, 2, 4, 7, 16, 32][self.next(6) as usize];
                        let d = self.m.i32(d_v);
                        let op = if self.next(2) == 0 { Bin::UDiv } else { Bin::URem };
                        self.m.bin(c, op, x, d)
                    }
                }
            };
            self.regs[dst] = r;
        }

        fn code(&mut self, depth: u32) {
            for _ in 0..1 + self.next(6) {
                match self.next(8) {
                    0 | 1 if depth > 0 => self.branch(depth - 1),
                    2 if depth > 0 => self.loop_n(depth - 1),
                    _ => self.stmt(),
                }
            }
        }

        fn branch(&mut self, depth: u32) {
            let c = self.cond();
            let i32t = self.m.int(32);
            let (then_b, else_b, join) = (
                self.m.new_block(),
                self.m.new_block(),
                self.m.new_block(),
            );
            self.m.terminate(self.cur, Term::CondBr(c, then_b, else_b));
            let (before, nconds) = (self.regs.clone(), self.conds.len());
            self.cur = then_b;
            self.code(depth);
            let (then_regs, then_end) = (std::mem::replace(&mut self.regs, before), self.cur);
            self.m.terminate(then_end, Term::Br(join));
            self.conds.truncate(nconds);
            self.cur = else_b;
            if self.next(3) != 0 {
                self.code(depth);
            }
            let else_end = self.cur;
            self.m.terminate(else_end, Term::Br(join));
            self.conds.truncate(nconds);
            for (r, &tv) in then_regs.iter().enumerate() {
                let p = self.m.phi(join, i32t);
                self.m.set_phi(p, vec![(tv, then_end), (self.regs[r], else_end)]);
                self.regs[r] = p;
            }
            self.cur = join;
        }

        fn loop_n(&mut self, depth: u32) {
            let i32t = self.m.int(32);
            let (head, exit) = (self.m.new_block(), self.m.new_block());
            let before = self.cur;
            self.m.terminate(before, Term::Br(head));
            let entering = self.regs.clone();
            let phis: Vec<V> = (0..self.regs.len()).map(|_| self.m.phi(head, i32t)).collect();
            let cnt = self.m.phi(head, i32t);
            self.regs = phis.clone();
            let nconds = self.conds.len();
            self.cur = head;
            self.code(depth);
            let one = self.m.i32(1);
            let next = self.m.bin(self.cur, Bin::Add, cnt, one);
            let limit_v = 1 + self.next(3) as u32;
            let limit = self.m.i32(limit_v);
            let mut more = self.m.cmp(self.cur, pred::IULT, next, limit);
            if self.next(2) == 0 {
                let c = self.cond();
                more = self.m.bin(self.cur, Bin::And, more, c);
            }
            let latch = self.cur;
            self.m.terminate(latch, Term::CondBr(more, head, exit));
            for (r, &p) in phis.iter().enumerate() {
                self.m.set_phi(p, vec![(entering[r], before), (self.regs[r], latch)]);
            }
            let zero = self.m.i32(0);
            self.m.set_phi(cnt, vec![(zero, before), (next, latch)]);
            self.conds.truncate(nconds);
            self.cur = exit;
        }
    }

    fn build_random_dataflow_prog(seed: u64) -> Module {
        let Kit {
            mut m,
            b,
            input,
            writes,
        } = kit();
        let i32t = m.int(32);
        let fty3 = m.func_ty(i32t, vec![i32t, i32t, i32t]);
        m.ptr(fty3);
        let fty2 = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty2);
        let bin32 = m.function("dx.op.binary.i32", fty3, Some(Attr::ReadNone), false);
        let unary32 = m.function("dx.op.unary.i32", fty2, Some(Attr::ReadNone), false);
        let unary_bits = m.function("dx.op.unaryBits.i32", fty2, Some(Attr::ReadNone), false);
        let mut g = RandGen {
            m,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            cur: b,
            regs: Vec::new(),
            conds: Vec::new(),
            writes,
            bin32,
            unary32,
            unary_bits,
        };
        for r in 0..5u32 {
            let (x, y) = (g.m.i32(r), g.m.i32(r * 5));
            let v = g.m.call(b, input, vec![x, y]);
            g.regs.push(v);
        }
        g.code(3);
        g.code(2);
        // Demand full 32 bits of some registers and masked low bits of others.
        for r in 0..g.regs.len() {
            let at = g.m.i32(r as u32);
            let v = if r == g.regs.len() - 1 {
                let m16 = g.m.i32(0xFFFF);
                g.m.bin(g.cur, Bin::And, g.regs[r], m16)
            } else {
                g.regs[r]
            };
            g.m.call(g.cur, g.writes, vec![at, v]);
        }
        g.m.terminate(g.cur, Term::Ret);
        g.m
    }

    #[test]
    fn randomized_equivalence_oracle_across_500_programs_and_pass_combos() {
        let mut total_baseline_left = 0usize;
        let mut total_dataflow_left = 0usize;
        let baseline_passes = DEAD | FOLD | REUSE | FACTS | LOADS;
        for seed in 1..=500u64 {
            let plain = build_random_dataflow_prog(seed);
            let inputs: Vec<Vec<u32>> = (0..6u32)
                .map(|n| {
                    (0..7u32)
                        .map(|i| match n {
                            0 => 0,
                            1 => i,
                            2 => u32::MAX - i,
                            3 => 0x8000_0000u32.wrapping_add(i),
                            _ => {
                                (seed as u32 ^ n)
                                    .wrapping_mul(0x9E37_79B9)
                                    .rotate_left(i * 5)
                                    ^ (i << 27)
                            }
                        })
                        .collect()
                })
                .collect();
            let expected: Vec<_> = inputs.iter().map(|i| run_ref(&plain, i)).collect();

            let mut base_m = build_random_dataflow_prog(seed);
            base_m.optimise(baseline_passes);
            total_baseline_left += base_m
                .blocks
                .iter()
                .map(|b| b.body.len() + b.phis.len())
                .sum::<usize>();

            for passes in [
                DATAFLOW,
                DATAFLOW | DEAD,
                FOLD | DATAFLOW | DEAD,
                DEFAULT | DATAFLOW,
                ALL,
            ] {
                let mut m = build_random_dataflow_prog(seed);
                m.optimise(passes);
                for (inp, want) in inputs.iter().zip(&expected) {
                    assert_eq!(
                        &run_ref(&m, inp),
                        want,
                        "mismatch on seed {seed} passes {passes:#x}"
                    );
                }
                if passes == (DEFAULT | DATAFLOW) {
                    total_dataflow_left += m
                        .blocks
                        .iter()
                        .map(|b| b.body.len() + b.phis.len())
                        .sum::<usize>();
                }
                // Bitcode serialization verifies dominance & value numbering invariants.
                let _ = m.write();
            }
        }
        assert!(
            total_dataflow_left < total_baseline_left,
            "DATAFLOW must strictly reduce instructions beyond baseline ({total_baseline_left} -> {total_dataflow_left})"
        );
    }
}
