//! GP factor miner for daily A-share data exported by research/gp_export.py.
//!
//!   gpminer mine  --data DIR --out FILE.json [--seed 1 --pop 400 --gens 30 ...]
//!   gpminer eval  --data DIR --formulas FILE --out FILE.json [--write-factors DIR]
//!   gpminer bench --data DIR
//!
//! Fitness uses the TRAIN split only. `mine` reports train and valid IC series for the hall of
//! fame; `eval` reports IC series for every split present in the data (test only exists in the
//! final-test export).
mod data;
mod eval;
mod expr;
mod fitness;
mod gp;
mod rng;

use data::Data;
use fitness::{score, sig_corr, Ctx};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

fn args() -> (String, HashMap<String, String>) {
    let a: Vec<String> = std::env::args().collect();
    let mut m = HashMap::new();
    let mut i = 2;
    while i < a.len() {
        let k = a[i].trim_start_matches("--").to_string();
        let v = a.get(i + 1).cloned().unwrap_or_default();
        m.insert(k, v);
        i += 2;
    }
    (a.get(1).cloned().unwrap_or_default(), m)
}

fn get<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, dflt: T) -> T {
    m.get(k).and_then(|v| v.parse().ok()).unwrap_or(dflt)
}

/// IC series for each split: {split: {"t": [...], "ic": [...]}}
fn split_ics(f: &[f32], d: &Data, nodes: usize) -> Value {
    let mut out = serde_json::Map::new();
    for (name, a, b) in &d.splits {
        let ctx = Ctx::new(d, *a, *b, usize::MAX);
        let (s, ics, _) = score(f, d, &ctx, nodes, false);
        let t: Vec<usize> = ctx.cs.iter().map(|c| c.t).collect();
        let ic: Vec<Value> = ics.iter().map(|v| if v.is_finite() { json!(*v as f64) } else { Value::Null }).collect();
        out.insert(name.clone(), json!({"t": t, "ic": ic, "cover": s.cover}));
    }
    Value::Object(out)
}

fn main() {
    let (mode, a) = args();
    let dir = a.get("data").expect("--data DIR").clone();
    let t0 = Instant::now();
    let d = Data::load(&dir);
    eprintln!("loaded T={} N={} terminals={:?} splits={:?} [{:.1}s]", d.t, d.n, d.term_names, d.splits, t0.elapsed().as_secs_f64());
    if let Some(th) = a.get("threads") {
        rayon::ThreadPoolBuilder::new().num_threads(th.parse().unwrap()).build_global().unwrap();
    }
    match mode.as_str() {
        "bench" => {
            let (ta, tb) = d.split("train").unwrap();
            let ctx = Ctx::new(&d, ta, tb, 8);
            eprintln!("train cross-sections {} signature days {} len {}", ctx.cs.len(), ctx.sig_days, ctx.sig_len);
            let mut rng = rng::Rng::new(7);
            let trees: Vec<expr::Node> = (0..32).map(|i| expr::Node::random(&mut rng, 3 + i % 2, i % 2 == 0, d.terms.len())).collect();
            let t1 = Instant::now();
            for n in &trees {
                let _ = gp::evaluate(n, &d, &ctx);
            }
            let serial = t1.elapsed().as_secs_f64() / trees.len() as f64;
            let t2 = Instant::now();
            use rayon::prelude::*;
            let _: Vec<_> = trees.par_iter().map(|n| gp::evaluate(n, &d, &ctx)).collect();
            let par = t2.elapsed().as_secs_f64() / trees.len() as f64;
            let nodes: f64 = trees.iter().map(|n| n.size() as f64).sum::<f64>() / trees.len() as f64;
            println!("mean nodes {nodes:.1}; serial {:.0} ms/tree; parallel {:.0} ms/tree", serial * 1e3, par * 1e3);
        }
        "mine" => {
            let (ta, tb) = d.split("train").expect("train split");
            let dt = d.truncate(tb + 1); // forward returns are precomputed; factors at t only use data <= t
            let ctx = Ctx::new(&dt, ta, tb, 8);
            let p = gp::Params {
                pop: get(&a, "pop", 400),
                gens: get(&a, "gens", 30),
                seed: get(&a, "seed", 1),
                hof_cap: get(&a, "hof", 200),
                hof_min: get(&a, "hof-min", 0.15),
                max_depth: get(&a, "max-depth", 6),
                max_nodes: get(&a, "max-nodes", 20),
                tour: get(&a, "tour", 5),
                elite: get(&a, "elite", 10),
                dup_corr: get(&a, "dup-corr", 0.7),
                crowd_corr: get(&a, "crowd-corr", 0.9),
            };
            let hof = gp::run(&dt, &ctx, &p);
            let members: Vec<Value> = hof
                .iter()
                .map(|m| {
                    let f = eval::eval(&m.node, &d);
                    json!({
                        "expr": m.expr, "nodes": m.score.nodes, "fitness": m.score.fitness, "mu": m.score.mu,
                        "icir": m.score.icir, "stab": m.score.stab, "yearly": m.score.yearly, "cover": m.score.cover,
                        "splits": split_ics(&f, &d, m.score.nodes),
                    })
                })
                .collect();
            let corr: Vec<Vec<f64>> =
                hof.iter().map(|x| hof.iter().map(|y| sig_corr(&x.sig, &y.sig)).collect()).collect();
            let out = json!({"seed": p.seed, "pop": p.pop, "gens": p.gens, "terminals": d.term_names,
                             "members": members, "corr": corr});
            std::fs::write(a.get("out").expect("--out"), serde_json::to_string(&out).unwrap()).unwrap();
            eprintln!("wrote {} members [{:.0}s]", hof.len(), t0.elapsed().as_secs_f64());
        }
        "eval" => {
            let text = std::fs::read_to_string(a.get("formulas").expect("--formulas")).unwrap();
            let formulas: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with('#')).collect();
            let wdir = a.get("write-factors").cloned();
            if let Some(w) = &wdir {
                std::fs::create_dir_all(w).unwrap();
            }
            let mut res = Vec::new();
            let sig_ctx = d.split("train").map(|(ta, tb)| Ctx::new(&d, ta, tb, 8));
            let mut sigs: Vec<Vec<f32>> = Vec::new();
            for (k, s) in formulas.iter().enumerate() {
                let node = expr::parse(s, &d.term_names).unwrap_or_else(|e| panic!("{s}: {e}"));
                let f = eval::eval(&node, &d);
                if let Some(w) = &wdir {
                    let bytes: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(format!("{w}/f{k}.f32"), bytes).unwrap();
                }
                if let Some(c) = &sig_ctx {
                    sigs.push(score(&f, &d, c, node.size(), true).2);
                }
                res.push(json!({"expr": node.fmt(&d.term_names), "nodes": node.size(), "splits": split_ics(&f, &d, node.size())}));
                eprintln!("  [{}/{}] {}", k + 1, formulas.len(), s);
            }
            let corr: Vec<Vec<f64>> = sigs.iter().map(|x| sigs.iter().map(|y| sig_corr(x, y)).collect()).collect();
            std::fs::write(a.get("out").expect("--out"), serde_json::to_string(&json!({"factors": res, "train_corr": corr})).unwrap()).unwrap();
        }
        _ => panic!("mode must be mine | eval | bench"),
    }
}
