use crate::data::Data;
use crate::expr::{Bi, Node, Ts, Un};

/// Rolling windows need at least 60% valid observations (and 2 for dispersion statistics).
pub fn minp(w: usize) -> usize {
    ((w as f64 * 0.6).ceil() as usize).max(2)
}

pub fn eval(node: &Node, d: &Data) -> Vec<f32> {
    let mut out = match node {
        Node::T(i) => d.terms[*i as usize].clone(),
        Node::U(op, a) => {
            let mut x = eval(a, d);
            unary(*op, &mut x, d);
            x
        }
        Node::B(op, a, b) => {
            let mut x = eval(a, d);
            let y = eval(b, d);
            binary(*op, &mut x, &y);
            x
        }
        Node::S(op, a, w) => {
            let x = eval(a, d);
            ts(*op, &x, *w as usize, d)
        }
        Node::C(a, b, w) => {
            let x = eval(a, d);
            let y = eval(b, d);
            ts_corr(&x, &y, *w as usize, d)
        }
    };
    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = f32::NAN;
        }
    }
    out
}

fn unary(op: Un, x: &mut [f32], d: &Data) {
    match op {
        Un::Abs => x.iter_mut().for_each(|v| *v = v.abs()),
        Un::Slog => x.iter_mut().for_each(|v| *v = v.signum() * v.abs().ln_1p()),
        Un::Sign => x.iter_mut().for_each(|v| {
            if v.is_finite() {
                *v = if *v > 0.0 { 1.0 } else if *v < 0.0 { -1.0 } else { 0.0 }
            }
        }),
        Un::CsRank => cross_section(x, d, true),
        Un::CsZ => cross_section(x, d, false),
    }
}

fn binary(op: Bi, x: &mut [f32], y: &[f32]) {
    match op {
        Bi::Add => x.iter_mut().zip(y).for_each(|(a, b)| *a += b),
        Bi::Sub => x.iter_mut().zip(y).for_each(|(a, b)| *a -= b),
        Bi::Mul => x.iter_mut().zip(y).for_each(|(a, b)| *a *= b),
        Bi::Div => x.iter_mut().zip(y).for_each(|(a, b)| *a = if b.abs() < 1e-9 { f32::NAN } else { *a / b }),
    }
}

/// Average ranks of `vals` (all finite), scaled to [0, 1].
pub fn avg_rank01(vals: &[f32], out: &mut Vec<f32>) {
    let n = vals.len();
    out.clear();
    out.resize(n, f32::NAN);
    if n < 2 {
        return;
    }
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_unstable_by(|&a, &b| vals[a as usize].partial_cmp(&vals[b as usize]).unwrap());
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && vals[idx[j] as usize] == vals[idx[i] as usize] {
            j += 1;
        }
        let r = (i + j - 1) as f32 / 2.0 / (n - 1) as f32;
        for k in i..j {
            out[idx[k] as usize] = r;
        }
        i = j;
    }
}

fn cross_section(x: &mut [f32], d: &Data, rank: bool) {
    let (tl, nl) = (d.t, d.n);
    let mut vals: Vec<f32> = Vec::with_capacity(nl);
    let mut who: Vec<usize> = Vec::with_capacity(nl);
    let mut r: Vec<f32> = Vec::with_capacity(nl);
    for t in 0..tl {
        vals.clear();
        who.clear();
        for n in 0..nl {
            let v = x[n * tl + t];
            if v.is_finite() && d.univ[n * tl + t] == 1 {
                vals.push(v);
                who.push(n);
            } else {
                x[n * tl + t] = f32::NAN;
            }
        }
        if vals.len() < 2 {
            for &n in &who {
                x[n * tl + t] = f32::NAN;
            }
            continue;
        }
        if rank {
            avg_rank01(&vals, &mut r);
            for (k, &n) in who.iter().enumerate() {
                x[n * tl + t] = r[k];
            }
        } else {
            let m = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
            let sd = var.sqrt();
            for (k, &n) in who.iter().enumerate() {
                x[n * tl + t] = if sd > 1e-12 { ((vals[k] as f64 - m) / sd) as f32 } else { f32::NAN };
            }
        }
    }
}

fn ts(op: Ts, x: &[f32], w: usize, d: &Data) -> Vec<f32> {
    let tl = d.t;
    let mut out = vec![f32::NAN; x.len()];
    for n in 0..d.n {
        let xs = &x[n * tl..(n + 1) * tl];
        let os = &mut out[n * tl..(n + 1) * tl];
        match op {
            Ts::Mean | Ts::Std | Ts::Z => roll_moments(xs, w, op, os),
            Ts::Delta => (w..tl).for_each(|t| os[t] = xs[t] - xs[t - w]),
            Ts::Delay => (w..tl).for_each(|t| os[t] = xs[t - w]),
            Ts::Max | Ts::Min | Ts::Rank | Ts::Decay => roll_naive(xs, w, op, os),
        }
    }
    out
}

fn roll_moments(x: &[f32], w: usize, op: Ts, out: &mut [f32]) {
    let mp = minp(w);
    let (mut s, mut s2, mut c) = (0f64, 0f64, 0usize);
    for t in 0..x.len() {
        let v = x[t];
        if v.is_finite() {
            s += v as f64;
            s2 += (v as f64) * (v as f64);
            c += 1;
        }
        if t >= w {
            let u = x[t - w];
            if u.is_finite() {
                s -= u as f64;
                s2 -= (u as f64) * (u as f64);
                c -= 1;
            }
        }
        if c < mp {
            continue;
        }
        let m = s / c as f64;
        let var = ((s2 - s * s / c as f64) / (c - 1) as f64).max(0.0);
        let sd = var.sqrt();
        out[t] = match op {
            Ts::Mean => m as f32,
            Ts::Std => sd as f32,
            _ => {
                if v.is_finite() && sd > 1e-9 * (1.0 + m.abs()) {
                    ((v as f64 - m) / sd) as f32
                } else {
                    f32::NAN
                }
            }
        };
    }
}

fn roll_naive(x: &[f32], w: usize, op: Ts, out: &mut [f32]) {
    let mp = minp(w);
    for t in 0..x.len() {
        let a = (t + 1).saturating_sub(w);
        let cur = x[t];
        if op == Ts::Rank && !cur.is_finite() {
            continue;
        }
        let mut c = 0usize;
        let (mut mx, mut mn) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut less, mut eq) = (0usize, 0usize);
        let (mut ws, mut wx) = (0f64, 0f64);
        for k in a..=t {
            let v = x[k];
            if !v.is_finite() {
                continue;
            }
            c += 1;
            match op {
                Ts::Max => mx = mx.max(v),
                Ts::Min => mn = mn.min(v),
                Ts::Rank => {
                    if v < cur {
                        less += 1
                    } else if v == cur {
                        eq += 1
                    }
                }
                _ => {
                    let wt = (w - (t - k)) as f64;
                    ws += wt;
                    wx += wt * v as f64;
                }
            }
        }
        if c < mp {
            continue;
        }
        out[t] = match op {
            Ts::Max => mx,
            Ts::Min => mn,
            Ts::Rank => (less as f32 + 0.5 * (eq as f32 - 1.0)) / (c - 1) as f32,
            _ => (wx / ws) as f32,
        };
    }
}

fn ts_corr(x: &[f32], y: &[f32], w: usize, d: &Data) -> Vec<f32> {
    let tl = d.t;
    let mp = minp(w).max(4);
    let mut out = vec![f32::NAN; x.len()];
    for n in 0..d.n {
        let (xs, ys) = (&x[n * tl..(n + 1) * tl], &y[n * tl..(n + 1) * tl]);
        let os = &mut out[n * tl..(n + 1) * tl];
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut c) = (0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
        for t in 0..tl {
            let (a, b) = (xs[t], ys[t]);
            if a.is_finite() && b.is_finite() {
                let (a, b) = (a as f64, b as f64);
                sx += a;
                sy += b;
                sxx += a * a;
                syy += b * b;
                sxy += a * b;
                c += 1;
            }
            if t >= w {
                let (a, b) = (xs[t - w], ys[t - w]);
                if a.is_finite() && b.is_finite() {
                    let (a, b) = (a as f64, b as f64);
                    sx -= a;
                    sy -= b;
                    sxx -= a * a;
                    syy -= b * b;
                    sxy -= a * b;
                    c -= 1;
                }
            }
            if c < mp {
                continue;
            }
            let cf = c as f64;
            let vx = sxx - sx * sx / cf;
            let vy = syy - sy * sy / cf;
            let cov = sxy - sx * sy / cf;
            let ex = 1e-10 * (1.0 + sxx);
            let ey = 1e-10 * (1.0 + syy);
            if vx > ex && vy > ey {
                os[t] = (cov / (vx * vy).sqrt()).clamp(-1.0, 1.0) as f32;
            }
        }
    }
    out
}
