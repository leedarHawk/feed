use crate::data::Data;
use crate::eval::eval;
use crate::expr::{Node, BIS, CORR_WINDOWS, TSS, UNS, WINDOWS};
use crate::fitness::{score, sig_corr, Ctx, Score};
use crate::rng::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

pub struct Params {
    pub pop: usize,
    pub gens: usize,
    pub seed: u64,
    pub hof_cap: usize,
    pub hof_min: f64,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub tour: usize,
    pub elite: usize,
    pub dup_corr: f64,
    pub crowd_corr: f64,
}

pub struct Member {
    pub node: Node,
    pub expr: String,
    pub score: Score,
    pub sig: Vec<f32>,
}

fn qualifies(s: &Score, p: &Params) -> bool {
    s.fitness >= p.hof_min && s.stab + 1 >= s.nyears
}

/// Insert into the hall of fame. A candidate correlated >= dup_corr with a member replaces it only if
/// it scores higher; otherwise it is appended (or replaces the weakest member when full).
fn hof_insert(hof: &mut Vec<Member>, m: Member, p: &Params, days: usize) -> bool {
    let corrs: Vec<f64> = hof.par_iter().map(|h| sig_corr(&h.sig, &m.sig, days).abs()).collect();
    if let Some((j, &c)) = corrs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
        if c >= p.dup_corr {
            if m.score.fitness > hof[j].score.fitness {
                hof[j] = m;
                return true;
            }
            return false;
        }
    }
    if hof.len() < p.hof_cap {
        hof.push(m);
        return true;
    }
    let (w, wf) = hof.iter().enumerate().map(|(i, h)| (i, h.score.fitness)).fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
    if m.score.fitness > wf {
        hof[w] = m;
        return true;
    }
    false
}

fn tournament<'a>(pop: &'a [Node], sel: &[f64], k: usize, rng: &mut Rng) -> &'a Node {
    let mut best = rng.below(pop.len());
    for _ in 1..k {
        let i = rng.below(pop.len());
        if sel[i] > sel[best] {
            best = i;
        }
    }
    &pop[best]
}

/// Pick a node index, preferring function nodes 90% of the time (Koza).
fn pick(n: &Node, rng: &mut Rng) -> usize {
    let s = n.size();
    for _ in 0..8 {
        let k = rng.below(s);
        let is_term = matches!(n.nth(k), Node::T(_));
        if !is_term || rng.f64() < 0.1 || s == 1 {
            return k;
        }
    }
    rng.below(s)
}

fn crossover(a: &Node, b: &Node, rng: &mut Rng) -> Node {
    let mut c = a.clone();
    let ka = pick(&c, rng);
    let kb = pick(b, rng);
    *c.nth_mut(ka) = b.nth(kb).clone();
    c
}

fn subtree_mut(a: &Node, rng: &mut Rng, nterm: usize) -> Node {
    let mut c = a.clone();
    let k = rng.below(c.size());
    let depth = 1 + rng.below(3);
    *c.nth_mut(k) = Node::random(rng, depth, false, nterm);
    c
}

fn point_mut(a: &Node, rng: &mut Rng, nterm: usize) -> Node {
    let mut c = a.clone();
    let k = rng.below(c.size());
    let flip = rng.f64() < 0.5;
    match c.nth_mut(k) {
        Node::T(i) => *i = rng.below(nterm) as u8,
        Node::U(op, _) => *op = UNS[rng.below(UNS.len())],
        Node::B(op, _, _) => *op = BIS[rng.below(BIS.len())],
        Node::S(op, _, w) => {
            if flip {
                *op = TSS[rng.below(TSS.len())]
            } else {
                *w = WINDOWS[rng.below(WINDOWS.len())]
            }
        }
        Node::C(a, b, w) => {
            if flip {
                std::mem::swap(a, b)
            } else {
                *w = CORR_WINDOWS[rng.below(CORR_WINDOWS.len())]
            }
        }
    }
    c
}

/// Replace a random function node by one of its children.
fn shrink(a: &Node, rng: &mut Rng) -> Node {
    let mut c = a.clone();
    let funcs: Vec<usize> = (0..c.size()).filter(|&k| !matches!(c.nth(k), Node::T(_))).collect();
    if funcs.is_empty() {
        return c;
    }
    let k = funcs[rng.below(funcs.len())];
    let child = match c.nth(k) {
        Node::U(_, x) | Node::S(_, x, _) => (**x).clone(),
        Node::B(_, x, y) | Node::C(x, y, _) => {
            if rng.f64() < 0.5 {
                (**x).clone()
            } else {
                (**y).clone()
            }
        }
        Node::T(_) => unreachable!(),
    };
    *c.nth_mut(k) = child;
    c
}

pub fn evaluate(node: &Node, d: &Data, ctx: &Ctx) -> (Score, Vec<f32>) {
    let f = eval(node, d);
    let (s, _, sig) = score(&f, d, ctx, node.size(), true);
    (s, sig)
}

pub fn run(d: &Data, ctx: &Ctx, p: &Params) -> Vec<Member> {
    let mut rng = Rng::new(p.seed);
    let nterm = d.terms.len();
    let names = &d.term_names;
    let t0 = Instant::now();
    let mut cache: HashMap<String, f64> = HashMap::new();
    let mut hof: Vec<Member> = Vec::new();

    let mut pop: Vec<Node> = Vec::with_capacity(p.pop);
    let mut seen = std::collections::HashSet::new();
    let mut i = 0;
    while pop.len() < p.pop {
        let n = Node::random(&mut rng, 2 + i % 3, i % 2 == 0, nterm);
        i += 1;
        if n.size() <= p.max_nodes && seen.insert(n.fmt(names)) {
            pop.push(n);
        }
    }
    let mut n_eval = 0usize;
    for gen in 0..=p.gens {
        // evaluate unseen expressions in parallel
        let exprs: Vec<String> = pop.iter().map(|n| n.fmt(names)).collect();
        let mut todo: Vec<usize> = Vec::new();
        let mut in_todo = std::collections::HashSet::new();
        for (k, e) in exprs.iter().enumerate() {
            if !cache.contains_key(e) && in_todo.insert(e.clone()) {
                todo.push(k);
            }
        }
        let res: Vec<(usize, Score, Vec<f32>)> = todo
            .par_iter()
            .map(|&k| {
                let (s, sig) = evaluate(&pop[k], d, ctx);
                (k, s, sig)
            })
            .collect();
        n_eval += res.len();
        let mut res = res;
        res.sort_by(|a, b| b.1.fitness.partial_cmp(&a.1.fitness).unwrap());
        for (k, s, sig) in res {
            let mut sel = s.fitness;
            if qualifies(&s, p) {
                // crowding: discount candidates that duplicate a stronger member
                let crowded = hof
                    .par_iter()
                    .any(|h| h.score.fitness >= s.fitness && sig_corr(&h.sig, &sig, ctx.sig_days).abs() >= p.crowd_corr);
                if crowded {
                    sel *= 0.5;
                }
                let m = Member { node: pop[k].clone(), expr: exprs[k].clone(), score: s, sig };
                hof_insert(&mut hof, m, p, ctx.sig_days);
            }
            cache.insert(exprs[k].clone(), sel);
        }
        let sel: Vec<f64> = exprs.iter().map(|e| cache[e]).collect();
        let best = hof.iter().map(|h| h.score.fitness).fold(f64::NEG_INFINITY, f64::max);
        let mean_sel = sel.iter().sum::<f64>() / sel.len() as f64;
        eprintln!(
            "seed {} gen {:2} evaluated {:5} (total {:6}) hof {:3} best {:.3} mean sel {:.3} [{:.0}s]",
            p.seed,
            gen,
            todo.len(),
            n_eval,
            hof.len(),
            best,
            mean_sel,
            t0.elapsed().as_secs_f64()
        );
        if gen == p.gens {
            break;
        }
        // breed the next generation
        let mut order: Vec<usize> = (0..pop.len()).collect();
        order.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap());
        let mut next: Vec<Node> = Vec::with_capacity(p.pop);
        let mut next_seen = std::collections::HashSet::new();
        for &k in order.iter() {
            if next.len() >= p.elite {
                break;
            }
            if next_seen.insert(exprs[k].clone()) {
                next.push(pop[k].clone());
            }
        }
        let mut tries = 0;
        while next.len() < p.pop {
            tries += 1;
            let r = rng.f64();
            let child = if r < 0.6 {
                let a = tournament(&pop, &sel, p.tour, &mut rng);
                let b = tournament(&pop, &sel, p.tour, &mut rng);
                crossover(a, b, &mut rng)
            } else if r < 0.8 {
                subtree_mut(tournament(&pop, &sel, p.tour, &mut rng), &mut rng, nterm)
            } else if r < 0.95 {
                point_mut(tournament(&pop, &sel, p.tour, &mut rng), &mut rng, nterm)
            } else {
                shrink(tournament(&pop, &sel, p.tour, &mut rng), &mut rng)
            };
            if child.depth() > p.max_depth || child.size() > p.max_nodes {
                continue;
            }
            let e = child.fmt(names);
            // prefer novel expressions; repeats are cheap (cached) but waste diversity
            if (cache.contains_key(&e) || next_seen.contains(&e)) && tries < p.pop * 20 && rng.f64() < 0.9 {
                continue;
            }
            next_seen.insert(e);
            next.push(child);
        }
        pop = next;
    }
    hof.sort_by(|a, b| b.score.fitness.partial_cmp(&a.score.fitness).unwrap());
    hof
}
