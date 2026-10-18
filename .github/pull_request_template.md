## What this changes

<!-- What changes and why. For a setting, default or engine release: what it does to memory, speed or replies. -->

## Testing

<!--
How you verified it, e.g. `shellcheck` on changed scripts.
-->

## Performance claim

<!--
Leave "None." unless this pull request claims a speed improvement (CONTRIBUTING.md, Performance claims). If it
does, replace it with the evidence: the Grill summary and qualification.txt, which says MEASURED FASTER, with
grill-evidence.tgz attached; or tools/bench.py's raw output for main and this pull request in a fenced code block.
-->

None.

## Checklist

- [ ] Script, setting or release changes: `./start.sh` at the defaults and `tools/smoke.py` pass on a DGX Spark.
- [ ] The README and `scripts/config.sh` match any setting or default I changed.
