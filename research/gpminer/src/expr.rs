use crate::rng::Rng;

pub const WINDOWS: [u16; 6] = [3, 5, 10, 20, 40, 60];
pub const CORR_WINDOWS: [u16; 5] = [5, 10, 20, 40, 60];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Un {
    Abs,
    Slog,
    Sign,
    CsRank,
    CsZ,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bi {
    Add,
    Sub,
    Mul,
    Div,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ts {
    Mean,
    Std,
    Delta,
    Delay,
    Max,
    Min,
    Rank,
    Z,
    Decay,
}

pub const UNS: [Un; 5] = [Un::Abs, Un::Slog, Un::Sign, Un::CsRank, Un::CsZ];
pub const BIS: [Bi; 4] = [Bi::Add, Bi::Sub, Bi::Mul, Bi::Div];
pub const TSS: [Ts; 9] = [Ts::Mean, Ts::Std, Ts::Delta, Ts::Delay, Ts::Max, Ts::Min, Ts::Rank, Ts::Z, Ts::Decay];

impl Un {
    pub fn name(self) -> &'static str {
        match self {
            Un::Abs => "abs",
            Un::Slog => "slog",
            Un::Sign => "sign",
            Un::CsRank => "cs_rank",
            Un::CsZ => "cs_z",
        }
    }
}
impl Bi {
    pub fn name(self) -> &'static str {
        match self {
            Bi::Add => "add",
            Bi::Sub => "sub",
            Bi::Mul => "mul",
            Bi::Div => "div",
        }
    }
}
impl Ts {
    pub fn name(self) -> &'static str {
        match self {
            Ts::Mean => "ts_mean",
            Ts::Std => "ts_std",
            Ts::Delta => "ts_delta",
            Ts::Delay => "ts_delay",
            Ts::Max => "ts_max",
            Ts::Min => "ts_min",
            Ts::Rank => "ts_rank",
            Ts::Z => "ts_z",
            Ts::Decay => "ts_decay",
        }
    }
}

/// T = terminal, U = unary, B = binary, S = time-series(x, d), C = ts_corr(x, y, d)
#[derive(Clone, Debug)]
pub enum Node {
    T(u8),
    U(Un, Box<Node>),
    B(Bi, Box<Node>, Box<Node>),
    S(Ts, Box<Node>, u16),
    C(Box<Node>, Box<Node>, u16),
}

impl Node {
    pub fn size(&self) -> usize {
        match self {
            Node::T(_) => 1,
            Node::U(_, a) | Node::S(_, a, _) => 1 + a.size(),
            Node::B(_, a, b) | Node::C(a, b, _) => 1 + a.size() + b.size(),
        }
    }
    pub fn depth(&self) -> usize {
        match self {
            Node::T(_) => 1,
            Node::U(_, a) | Node::S(_, a, _) => 1 + a.depth(),
            Node::B(_, a, b) | Node::C(a, b, _) => 1 + a.depth().max(b.depth()),
        }
    }
    pub fn fmt(&self, names: &[String]) -> String {
        match self {
            Node::T(i) => names[*i as usize].clone(),
            Node::U(op, a) => format!("{}({})", op.name(), a.fmt(names)),
            Node::B(op, a, b) => format!("{}({},{})", op.name(), a.fmt(names), b.fmt(names)),
            Node::S(op, a, w) => format!("{}({},{})", op.name(), a.fmt(names), w),
            Node::C(a, b, w) => format!("ts_corr({},{},{})", a.fmt(names), b.fmt(names), w),
        }
    }
    /// Preorder access to the k-th node.
    pub fn nth(&self, k: usize) -> &Node {
        if k == 0 {
            return self;
        }
        match self {
            Node::T(_) => unreachable!(),
            Node::U(_, a) | Node::S(_, a, _) => a.nth(k - 1),
            Node::B(_, a, b) | Node::C(a, b, _) => {
                let sa = a.size();
                if k - 1 < sa {
                    a.nth(k - 1)
                } else {
                    b.nth(k - 1 - sa)
                }
            }
        }
    }
    pub fn nth_mut(&mut self, k: usize) -> &mut Node {
        if k == 0 {
            return self;
        }
        match self {
            Node::T(_) => unreachable!(),
            Node::U(_, a) | Node::S(_, a, _) => a.nth_mut(k - 1),
            Node::B(_, a, b) | Node::C(a, b, _) => {
                let sa = a.size();
                if k - 1 < sa {
                    a.nth_mut(k - 1)
                } else {
                    b.nth_mut(k - 1 - sa)
                }
            }
        }
    }

    pub fn random(rng: &mut Rng, depth: usize, full: bool, nterm: usize) -> Node {
        if depth <= 1 || (!full && rng.f64() < 0.3) {
            return Node::T(rng.below(nterm) as u8);
        }
        let r = rng.f64();
        let sub = |rng: &mut Rng| Box::new(Node::random(rng, depth - 1, full, nterm));
        if r < 0.15 {
            Node::U(UNS[rng.below(UNS.len())], sub(rng))
        } else if r < 0.40 {
            let a = sub(rng);
            Node::B(BIS[rng.below(BIS.len())], a, sub(rng))
        } else if r < 0.90 {
            let op = TSS[rng.below(TSS.len())];
            Node::S(op, sub(rng), WINDOWS[rng.below(WINDOWS.len())])
        } else {
            let a = sub(rng);
            Node::C(a, sub(rng), CORR_WINDOWS[rng.below(CORR_WINDOWS.len())])
        }
    }
}

// ------------------------------------------------------------------ parser for `eval` mode
pub fn parse(s: &str, names: &[String]) -> Result<Node, String> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let (node, rest) = parse_at(&s, names)?;
    if !rest.is_empty() {
        return Err(format!("trailing input: {rest}"));
    }
    Ok(node)
}

fn parse_at<'a>(s: &'a str, names: &[String]) -> Result<(Node, &'a str), String> {
    let end = s.find(|c: char| c == '(' || c == ',' || c == ')').unwrap_or(s.len());
    let head = &s[..end];
    let rest = &s[end..];
    if !rest.starts_with('(') {
        let i = names.iter().position(|n| n == head).ok_or(format!("unknown terminal {head}"))?;
        return Ok((Node::T(i as u8), rest));
    }
    let mut args: Vec<Node> = Vec::new();
    let mut nums: Vec<u16> = Vec::new();
    let mut r = &rest[1..];
    loop {
        if let Ok(v) = r[..r.find(|c: char| c == ',' || c == ')').unwrap_or(r.len())].parse::<u16>() {
            nums.push(v);
            r = &r[r.find(|c: char| c == ',' || c == ')').unwrap()..];
        } else {
            let (a, rr) = parse_at(r, names)?;
            args.push(a);
            r = rr;
        }
        if let Some(rr) = r.strip_prefix(',') {
            r = rr;
        } else if let Some(rr) = r.strip_prefix(')') {
            r = rr;
            break;
        } else {
            return Err(format!("expected , or ) at {r}"));
        }
    }
    let bx = |n: Node| Box::new(n);
    let node = if head == "ts_corr" {
        let b = args.pop().ok_or("ts_corr args")?;
        let a = args.pop().ok_or("ts_corr args")?;
        Node::C(bx(a), bx(b), nums[0])
    } else if let Some(op) = UNS.iter().find(|o| o.name() == head) {
        Node::U(*op, bx(args.pop().ok_or("unary arg")?))
    } else if let Some(op) = BIS.iter().find(|o| o.name() == head) {
        let b = args.pop().ok_or("binary args")?;
        let a = args.pop().ok_or("binary args")?;
        Node::B(*op, bx(a), bx(b))
    } else if let Some(op) = TSS.iter().find(|o| o.name() == head) {
        Node::S(*op, bx(args.pop().ok_or("ts arg")?), *nums.first().ok_or("ts window")?)
    } else {
        return Err(format!("unknown op {head}"));
    };
    Ok((node, r))
}
