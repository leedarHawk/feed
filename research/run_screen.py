import time, numpy as np, polars as pl
import lib
t=time.time()
B = lib.build_base(); F = lib.build_factors(B)
print(f"built base+factors {time.time()-t:.0f}s", flush=True)
ics = lib.screen(B, F)
print(f"screened {len(ics)} (factor,h) pairs {time.time()-t:.0f}s", flush=True)
np.savez_compressed("/tmp/feed_cache/ics_discovery.npz", **{f"{k[0]}|{k[1]}": v for k, v in ics.items()}, dates=B.dates)
S = lib.summarize(B, ics)
S.write_csv("results/ic_screen_discovery.csv")
print("done", flush=True)
