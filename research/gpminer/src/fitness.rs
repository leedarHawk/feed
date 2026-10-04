use crate::data::Data;
use crate::eval::avg_rank01;

/// One cross-section of the mining universe: stocks in M with a forward return and a small score.
/// fr / sr are centred [0,1] ranks of the forward return and of the small score.
pub struct Cs {
    pub t: usize,
    pub idx: Vec<u32>,
    pub fr: Vec<f32>,
    pub sr: Vec<f32>,
    pub fr_ss: f64,
    pub sr_ss: f64,
}

pub fn build(d: &Data, a: usize, b: usize) -> Vec<Cs> {
    let tl = d.t;
    let mut out = Vec::new();
    let mut r = Vec::new();
    for t in a..=b.min(tl - 1) {
        let idx: Vec<u32> = (0..d.n)
            .filter(|&n| {
                let k = n * tl + t;
                d.mask[k] == 1 && d.fwd[k].is_finite() && d.small[k].is_finite()
            })
            .map(|n| n as u32)
            .collect();
        if idx.len() < 50 {
            continue;
        }
        let centred = |src: &[f32], r: &mut Vec<f32>| -> (Vec<f32>, f64) {
            let vals: Vec<f32> = idx.iter().map(|&n| src[n as usize * tl + t]).collect();
            avg_rank01(&vals, r);
            let m = r.iter().map(|&v| v as f64).sum::<f64>() / r.len() as f64;
            let c: Vec<f32> = r.iter().map(|&v| (v as f64 - m) as f32).collect();
            let ss = c.iter().map(|&v| (v as f64) * (v as f64)).sum();
            (c, ss)
        };
        let (fr, fr_ss) = centred(&d.fwd, &mut r);
        let (sr, sr_ss) = centred(&d.small, &mut r);
        out.push(Cs { t, idx, fr, sr, fr_ss, sr_ss });
    }
    out
}

/// Rank IC of factor f on one cross-section after removing the small score (OLS on ranks).
/// Missing factor values get the middle rank. Returns (ic, n_finite); optionally writes the
/// unit-norm residual into `sig` (used for factor-to-factor correlation).
pub fn ic_day(f: &[f32], tl: usize, cs: &Cs, vals: &mut Vec<f32>, r: &mut Vec<f32>, sig: Option<&mut [f32]>) -> (f32, usize) {
    vals.clear();
    let mut pos: Vec<usize> = Vec::with_capacity(cs.idx.len());
    for (k, &n) in cs.idx.iter().enumerate() {
        let v = f[n as usize * tl + cs.t];
        if v.is_finite() {
            vals.push(v);
            pos.push(k);
        }
    }
    let nf = vals.len();
    let m = cs.idx.len();
    if nf < 2 {
        if let Some(s) = sig {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        return (f32::NAN, nf);
    }
    avg_rank01(vals, r);
    let mut rf = vec![0.5f32; m];
    for (j, &k) in pos.iter().enumerate() {
        rf[k] = r[j];
    }
    let mean = rf.iter().map(|&v| v as f64).sum::<f64>() / m as f64;
    let mut cov_s = 0f64;
    for k in 0..m {
        cov_s += (rf[k] as f64 - mean) * cs.sr[k] as f64;
    }
    let beta = if cs.sr_ss > 0.0 { cov_s / cs.sr_ss } else { 0.0 };
    let (mut ss, mut sf) = (0f64, 0f64);
    for k in 0..m {
        let e = rf[k] as f64 - mean - beta * cs.sr[k] as f64;
        rf[k] = e as f32;
        ss += e * e;
        sf += e * cs.fr[k] as f64;
    }
    if ss < 1e-10 {
        if let Some(s) = sig {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        return (f32::NAN, nf);
    }
    if let Some(s) = sig {
        let inv = 1.0 / ss.sqrt();
        for k in 0..m {
            s[k] = (rf[k] as f64 * inv) as f32;
        }
    }
    ((sf / (ss * cs.fr_ss).sqrt()) as f32, nf)
}

#[derive(Clone, Debug, Default)]
pub struct Score {
    pub fitness: f64,
    pub mu: f64,
    pub sd: f64,
    pub icir: f64,
    pub stab: usize,
    pub nyears: usize,
    pub yearly: Vec<f64>,
    pub cover: f64,
    pub nodes: usize,
}

/// Training context: cross-sections, year of each, and which ones go into the signature.
pub struct Ctx {
    pub cs: Vec<Cs>,
    pub year_of: Vec<usize>,
    pub nyears: usize,
    pub sig_off: Vec<Option<usize>>,
    pub sig_len: usize,
    pub sig_days: usize,
}

impl Ctx {
    pub fn new(d: &Data, a: usize, b: usize, sig_every: usize) -> Ctx {
        let cs = build(d, a, b);
        let mut years: Vec<usize> = Vec::new();
        let year_of: Vec<usize> = cs
            .iter()
            .map(|c| {
                let y = d.years.iter().position(|(_, s, e)| c.t >= *s && c.t <= *e).unwrap();
                if let Some(p) = years.iter().position(|&x| x == y) {
                    p
                } else {
                    years.push(y);
                    years.len() - 1
                }
            })
            .collect();
        let mut sig_off = vec![None; cs.len()];
        let mut off = 0;
        let mut days = 0;
        for (k, c) in cs.iter().enumerate() {
            if k % sig_every == 0 {
                sig_off[k] = Some(off);
                off += c.idx.len();
                days += 1;
            }
        }
        Ctx { nyears: years.len(), cs, year_of, sig_off, sig_len: off, sig_days: days }
    }
}

pub fn score(f: &[f32], d: &Data, ctx: &Ctx, nodes: usize, want_sig: bool) -> (Score, Vec<f32>, Vec<f32>) {
    let mut vals = Vec::new();
    let mut r = Vec::new();
    let mut sig = if want_sig { vec![0f32; ctx.sig_len] } else { Vec::new() };
    let mut ics = Vec::with_capacity(ctx.cs.len());
    let (mut tot, mut fin) = (0usize, 0usize);
    for (k, c) in ctx.cs.iter().enumerate() {
        let s = match (want_sig, ctx.sig_off[k]) {
            (true, Some(o)) => Some(&mut sig[o..o + c.idx.len()]),
            _ => None,
        };
        let (ic, nf) = ic_day(f, d.t, c, &mut vals, &mut r, s);
        ics.push(ic);
        tot += c.idx.len();
        fin += nf;
    }
    let cover = fin as f64 / tot.max(1) as f64;
    let good: Vec<(usize, f64)> = ics.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(k, &v)| (k, v as f64)).collect();
    let mut sc = Score { cover, nodes, nyears: ctx.nyears, ..Default::default() };
    if good.len() < 100 || cover < 0.9 {
        return (sc, ics, sig);
    }
    let n = good.len() as f64;
    let mu = good.iter().map(|x| x.1).sum::<f64>() / n;
    let sd = (good.iter().map(|x| (x.1 - mu).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let mut ys = vec![(0f64, 0usize); ctx.nyears];
    for &(k, v) in &good {
        let y = ctx.year_of[k];
        ys[y].0 += v;
        ys[y].1 += 1;
    }
    let yearly: Vec<f64> = ys.iter().map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 }).collect();
    let stab = yearly.iter().filter(|&&y| y * mu > 0.0).count();
    let icir = if sd > 0.0 { mu / sd } else { 0.0 };
    let mult = if stab == ctx.nyears {
        1.0
    } else if stab + 1 == ctx.nyears {
        0.6
    } else {
        0.2
    };
    let fitness = icir.abs() * mult - 0.004 * (nodes.saturating_sub(7)) as f64;
    sc.fitness = fitness;
    sc.mu = mu;
    sc.sd = sd;
    sc.icir = icir;
    sc.stab = stab;
    sc.yearly = yearly;
    (sc, ics, sig)
}

/// Mean per-day correlation of two factors' size-neutral ranks (signatures are unit-norm per day).
pub fn sig_corr(a: &[f32], b: &[f32], days: usize) -> f64 {
    let mut s = 0f64;
    for (x, y) in a.iter().zip(b) {
        s += (*x as f64) * (*y as f64);
    }
    s / days.max(1) as f64
}
