# Contributing

Read the [README](README.md) and [`scripts/config.sh`](scripts/config.sh) first, then fill in the pull request
template.

## Patches

The README promises replies token for token the same as stock TensorFold's. A pull request that changes `patches/`
keeps that promise: boot `main` and the pull request as in [Baseline](#baseline), run `tools/bench.py decode` on
each, and paste both outputs into Testing. Every `token_sha256` must be the same on both.

## Performance claims

Most pull requests need nothing extra. A pull request that **claims a speed improvement** brings raw numbers
measured against the recipe's baseline, so maintainers can judge the claim without re-testing it.

**What counts as a claim:** saying the change makes the recipe faster (in the title, description, a patch message
or your own comments), changing a number in the README, or changing a default, an option or a patch because it is
faster. No claim, no measurement. If the numbers don't support the claim, drop the claim; the pull request can
still be opened.

### Baseline

Every run uses the recipe's defaults on one DGX Spark, so numbers from different contributors compare:

- No `.env`, or one that only says where things are on your Spark: `MODEL_DIR`, `TF_HOME`, `ZIG`, `NVCC`, and
  `TF_ARCHIVE` or `TF_SRC` on a Spark without GitHub.
- A shell with none of the recipe's variables exported (`scripts/config.sh` lists them), and no `TF_*` or
  `TENSORFOLD_*`: exports override `.env` and the defaults, and reach the engine.
- `./start.sh` with no arguments, from that checkout: it builds the checkout's `patches/`.
- Nothing else running on the GPU.

A claim about other settings (for example `VISION=0`) uses them on every run instead. Say so in the declarations.

### Speed claims the Grill measures

The Grill measures generation speed on a short counting prompt (temperature 0, thinking off, 400 output tokens per
request), either one request at a time (`portable-v1`; the README's Stock vs this recipe table is its result) or
with 2, 4 or 8 requests at once (`portable-c2-v1`, `portable-c4-v1`,
`portable-c8-v1`, verdict on their combined tokens per second). A claim about either ("faster decode", "+N% tok/s",
"+N% at 4 requests at once") is measured with the newest release of
[The Grill](https://github.com/plotarmordev/thegrill/releases) (the highest version listed, pre-releases included;
0.8.0 or later). Install it as its
[install guide](https://github.com/plotarmordev/thegrill/blob/main/docs/performance/INSTALL.md#obtain-verify-unpack)
shows, checksum included. State the concurrency level in the claim before you measure, and use that one workload
for all three runs.

Claims the Grill doesn't measure (cold prefill, prose, images) use [`tools/bench.py`](tools/bench.py) instead, on
`main` and on the pull request booted the same way (`ttft` with the server settings its usage names). Paste both
raw outputs into the Performance claim section in a fenced code block.

Take three runs. Before each one: `./stop.sh`, check out the commit, `./start.sh`.

| Run | Checkout |
|---|---|
| A | `main` at your branch point (`git merge-base origin/main HEAD`) |
| B | your pull request |
| A2 | the same `main` commit again |

For a new opt-in option, all three runs use your pull request: A and A2 without the option, B with it.

Write one declaration for each side, `a.json` and `b.json`. `runtime` is the checkout's `git rev-parse HEAD`.
`settings` is `baseline` (the checkout's `scripts/config.sh` defaults), plus the option on B for an option claim,
plus the settings on both for a claim at other settings. The two files differ only in `runtime` (a code, patch or
default change) or only in `settings` (an option); the Grill refuses B otherwise.

```json
{"model_revision": "as in scripts/config.sh",
 "runtime": "Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold@<git rev-parse HEAD>",
 "hardware": "DGX Spark GB10",
 "settings": "baseline"}
```

On the Spark, with `ENDPOINT=http://127.0.0.1:8888/v1/chat/completions`, `MODEL=nemotron-3.5-super-vl` and
`WORKLOAD` set to the claim's workload (`portable-v1`, or `portable-c2-v1`/`-c4-v1`/`-c8-v1`):

```sh
# A
grill-perf baseline --workload "$WORKLOAD" --client-placement same-host \
  --endpoint "$ENDPOINT" --local-http --model "$MODEL" --deployment a.json --out A
# B: --change runtime for a code, patch or default change, --change settings for an option
grill-perf check A --change runtime --deployment b.json --out B
# A2 (its own exit code doesn't matter)
grill-perf check A --change none --deployment A/deployment.json --out A2

grill-perf compare A B --reference A2 > qualification.txt
grill-perf compare A B --reference A2 --json > qualification.json
```

B and A2 reuse A's endpoint, model and placement, and must be measured by the same Grill binary as A.

Each run must start at least 60 seconds after the previous one finished; a boot usually covers that. If a run
needs more than five minutes, pass the same `--seconds N` to all three. The concurrent workloads ask for up to
102,400 tokens per run, so on one Spark use `--seconds 1800` for them; it is a ceiling, not a target.

**The claim stands only if `compare` prints `MEASURED FASTER`:** B is faster than both A and A2. Quote the measured
change, not a bigger number. Anything else, drop the claim. If you push code after measuring, measure again.

### Attach the evidence

```sh
tar czf grill-evidence.tgz A B A2      # add every earlier attempt
jq -r --slurpfile a2 A2/report.json '
  def pct: . * 100 | round / 100 | if . >= 0 then "+\(.)" else "\(.)" end;
  def rounds: map(. * 10 | round / 10) | join(" ");
  def leg: "\(.observed_change_percent | pct)% [\(.model_based_interval_percent | map(pct) | join(", "))]";
  .deployment as $d |
  "grill:    \($d.baseline.collector.version), binary sha256 A \($d.baseline.collector_sha256), B \($d.candidate.collector_sha256)",
  "A:        \($d.baseline.declaration | .runtime + "; " + .settings)",
  "B:        \($d.candidate.declaration | .runtime + "; " + .settings)",
  "result:   \(.result)",
  "B vs A:   \(leg)    B vs A2: \($d.candidate_against_reference | leg)    A2 vs A: \($d.reference_against_baseline | leg)",
  "complete: A \(.baseline_accounting.complete_requests)/\(.baseline_accounting.request_ceiling), B \(.candidate_accounting.complete_requests)/\(.candidate_accounting.request_ceiling), A2 \($a2[0].candidate_accounting.complete_requests)/\($a2[0].candidate_accounting.request_ceiling)",
  "tok/s A:  \(.baseline_acquisition_medians | rounds)",
  "tok/s B:  \(.candidate_acquisition_medians | rounds)",
  "tok/s A2: \($a2[0].candidate_acquisition_medians | rounds)"' qualification.json
```

Attach `grill-evidence.tgz` to the pull request, and paste the summary and `qualification.txt` into its
Performance claim section. Exact values are in `qualification.json`. The evidence holds only the counting prompt,
the replies, a loopback address and your declarations.

A check on every pull request reads that section: it passes when the section says `None.`, or contains
`MEASURED FASTER`, or a fenced code block. It checks the form; maintainers check the numbers.

### Maintainer check

No GPU and no re-run. `compare` re-verifies every byte (one edited byte makes it `INVALID`), and the `jq` checks
the rest. Set `BASE` and `HEAD` to the pull request's merge base and head commits:

```sh
tar xzf grill-evidence.tgz
V=v$(jq -r .collector.version A/capture.json)
START=$(jq '.acquisitions[0].started_unix_ms / 1000 | floor' A/capture.json)
NEWEST=$(gh release list --repo plotarmordev/thegrill --json tagName,publishedAt,isDraft \
  --jq "[.[] | select((.isDraft | not) and (.publishedAt | fromdate) < $START)][0].tagName")
SHA=$(gh release download "$V" --repo plotarmordev/thegrill \
  -p "grill-perf-${V#v}-$(jq -r .collector.target A/capture.json).receipt.json" \
  -O - | jq -r .binary.sha256)
grill-perf compare A B --reference A2 --json | jq --arg sha "$SHA" \
  --arg base "${BASE:?set BASE to the merge base sha}" \
  --arg head "${HEAD:?set HEAD to the pull request head sha}" \
  --arg v "$V" --arg newest "$NEWEST" -e '
  .result == "IMPROVED"
  and $v == $newest
  and .deployment.baseline.collector_sha256 == $sha
  and (.declared_change | IN("runtime", "settings"))
  and (.deployment.candidate.declaration.runtime | endswith($head))
  and (.deployment.baseline.declaration.runtime | endswith($base) or endswith($head))'
```

`true` means: B is faster than both baseline runs; the measuring binary is the official one of the Grill release
that was newest when measuring started; B is the pull request head and A its merge base (or the head, for an
option); and B is a `runtime` or `settings` change, for which the Grill itself guarantees that only that
declaration field differs and that one binary measured all three runs. The evidence is tamper-evident, not
signed. Hardware, settings and the boots are the contributor's word, so maintainers may still re-run a claim
occasionally.
