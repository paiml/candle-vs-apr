# CLAUDE.md — candle-vs-apr

## Project Overview

Head-to-head benchmark: **Candle** (HuggingFace Rust ML) vs **realizr** (Sovereign AI Stack)
on RTX 4090 with locked clocks. Uses `probador llm load` for v2 methodology.

Not a Rust project — this is a benchmark orchestration repo with shell scripts,
forjar configs, and result analysis.

## Key Commands

```bash
make bench-all          # Full benchmark suite
make bench-candle       # Candle-only decode benchmark
make bench-realizr-c1   # realizr single-request benchmark
make compare            # Generate comparison tables
```

## Code Search

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

```bash
pmat query "benchmark" --limit 10          # Find by intent
pmat query --regex "tok/s" --limit 10      # Regex pattern
pmat query --literal "probador" --limit 5  # Literal match
pmat query "kernel" --faults --limit 10    # With fault patterns
```

See global CLAUDE.md for full `pmat query` decision tree.

## Quality

- `pmat comply` must pass before push
- Spec at `docs/specifications/candle-vs-apr-spec.md` — Popperian falsification
- All benchmark results must be reproducible with locked GPU clocks
