use serde_json::Value;
use std::fs;
use std::path::Path;

/// Matrices are stock-major: value of stock n on date t is at n * t_len + t.
pub struct Data {
    pub t: usize,
    pub n: usize,
    pub terms: Vec<Vec<f32>>,
    pub term_names: Vec<String>,
    pub mask: Vec<u8>,
    pub univ: Vec<u8>,
    pub fwd: Vec<f32>,
    pub small: Vec<f32>,
    pub dates: Vec<String>,
    pub splits: Vec<(String, usize, usize)>,
    pub years: Vec<(String, usize, usize)>,
}

fn read_f32(p: &Path, len: usize) -> Vec<f32> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("{}: {}", p.display(), e));
    assert_eq!(b.len(), len * 4, "{} has wrong size", p.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn ranges(v: &Value) -> Vec<(String, usize, usize)> {
    let mut out: Vec<(String, usize, usize)> = v
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, r)| (k.clone(), r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize))
        .collect();
    out.sort_by_key(|x| x.1);
    out
}

impl Data {
    pub fn load(dir: &str) -> Data {
        let d = Path::new(dir);
        let meta: Value = serde_json::from_str(&fs::read_to_string(d.join("meta.json")).unwrap()).unwrap();
        let t = meta["T"].as_u64().unwrap() as usize;
        let n = meta["N"].as_u64().unwrap() as usize;
        let term_names: Vec<String> =
            meta["terminals"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let terms = term_names.iter().map(|k| read_f32(&d.join(format!("{k}.f32")), t * n)).collect();
        let mask = fs::read(d.join("mask.u8")).unwrap();
        assert_eq!(mask.len(), t * n);
        let univ = fs::read(d.join("univ.u8")).unwrap();
        assert_eq!(univ.len(), t * n);
        Data {
            t,
            n,
            terms,
            term_names,
            mask,
            univ,
            fwd: read_f32(&d.join("fwd.f32"), t * n),
            small: read_f32(&d.join("small.f32"), t * n),
            dates: meta["dates"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            splits: ranges(&meta["splits"]),
            years: ranges(&meta["years"]),
        }
    }
    /// Copy holding only the first `tn` dates (mining never needs data past the train window).
    pub fn truncate(&self, tn: usize) -> Data {
        let tn = tn.min(self.t);
        let cut_f = |v: &Vec<f32>| -> Vec<f32> { (0..self.n).flat_map(|n| v[n * self.t..n * self.t + tn].iter().copied()).collect() };
        let cut_u = |v: &Vec<u8>| -> Vec<u8> { (0..self.n).flat_map(|n| v[n * self.t..n * self.t + tn].iter().copied()).collect() };
        Data {
            t: tn,
            n: self.n,
            terms: self.terms.iter().map(cut_f).collect(),
            term_names: self.term_names.clone(),
            mask: cut_u(&self.mask),
            univ: cut_u(&self.univ),
            fwd: cut_f(&self.fwd),
            small: cut_f(&self.small),
            dates: self.dates[..tn].to_vec(),
            splits: self.splits.iter().filter(|s| s.2 < tn).cloned().collect(),
            years: self.years.clone(),
        }
    }
    pub fn split(&self, name: &str) -> Option<(usize, usize)> {
        self.splits.iter().find(|s| s.0 == name).map(|s| (s.1, s.2))
    }
}
