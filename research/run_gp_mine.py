"""Run the pre-registered GP mining (gp_spec.py): 4 seeds on the <= 2024 export.
    python3 gp_export.py /tmp/gp_data/mine && python3 run_gp_mine.py /tmp/gp_data/mine
Halls of fame go to /tmp/gp_runs (large JSON with IC series); a summary CSV is written to results/."""
import json, os, subprocess, sys
import gp_spec as gs
data = sys.argv[1]
meta = json.load(open(os.path.join(data, "meta.json")))
assert meta["end"] == "2024-12-31", "mining must use the <= 2024 export"
os.makedirs("/tmp/gp_runs", exist_ok=True)
exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpminer", "target", "release", "gpminer")
for s in gs.SEEDS:
    out = f"/tmp/gp_runs/hof_seed{s}.json"
    args = [exe, "mine", "--data", data, "--out", out, "--seed", str(s)]
    for k, v in gs.GP_ARGS.items():
        args += [f"--{k}", str(v)]
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)
