# AGENTS.md

## Project Goal

This repository is the server-side code snapshot for the BRM-Net thesis/paper experiments.

Primary objective:

> Organize, verify, and extend experiments for budget-aware lightweight multimodal remote sensing classification and modality-reliability-aware fusion.

## Safety Rules

- Do not delete historical logs, checkpoints, figures, or `.out` files unless explicitly instructed.
- Do not move large experiment directories without a migration note and explicit confirmation.
- Do not invent experimental results.
- When producing paper tables, every number must point to a source log, CSV, checkpoint, or explicit user instruction.
- Keep local edits small and reversible.

## Main Code Lines

- Pruning/lightweight: `prune/`, `MCL/`, `models/`, `configuration/prune_config.py`.
- Missing/degraded modality: `missing*`, `Fmc/`, `Drfuse/`, `claude/`.
- Shared infrastructure: `datasets/`, `src/`, `lib/`, `loss/`, `models/`.

## Preferred Workflow

1. Inspect `docs/CODE_MAP.md` and `docs/RUNBOOK.md`.
2. Use `scripts/summarize_results.py` to locate usable results before rerunning experiments.
3. Put new experiment commands and expected outputs in `docs/EXPERIMENT_INDEX.md`.
4. Keep final paper-facing results synchronized with `D:/Academic/paper_submission/brmnet_pricai2026/notes/experiments.md`.

## Coding Rules

- Preserve existing training behavior unless the user asks for implementation changes.
- Add new modules as standalone files first; wire them into training only after the entry point and expected experiment are clear.
- Avoid changing global config defaults unless a runbook update accompanies the change.

