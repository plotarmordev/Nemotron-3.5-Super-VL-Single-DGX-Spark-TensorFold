# Agent instructions

- Read [README.md](README.md) and [scripts/config.sh](scripts/config.sh) before changing the recipe.
- A pull request that changes `patches/` keeps replies token for token the same as `main`'s: compare
  `tools/bench.py decode` hashes as [Patches](CONTRIBUTING.md#patches) says.
- A pull request that claims a speed improvement needs evidence measured against the recipe baseline: follow
  [Performance claims](CONTRIBUTING.md#performance-claims) before you make the claim. Without a speed claim,
  nothing extra is needed, and the template's Performance claim stays `None.`
- For a speed claim the Grill measures (one request at a time, or 2, 4 or 8 at once), keep the claim only if the
  newest Grill release prints `MEASURED FASTER`; otherwise drop it. Never edit evidence, and attach every attempt.
