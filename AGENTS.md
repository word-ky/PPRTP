# PPRTP — Codex Research Execution Protocol

## Roles

This repository is run as a research loop between ChatGPT (research lead) and Codex (engineering executor).

- **ChatGPT** owns: scientific hypothesis, literature positioning, experiment priority, ablation design, interpretation, go/no-go decisions, and next-hour task definition.
- **Codex** owns: implementation, debugging, reproducible experiment execution, logging, result collection, plots/tables, and concise engineering diagnosis.
- Do not silently change the scientific question. If an implementation issue forces a change, record it in `CHATGPT_CODEX_BRIDGE.md` first.

## Primary scientific question

Can global class prototypes be used directly as an all-class classifier so that classes missing on a local client still participate in the client's optimization, and can this provide a cleaner alternative to FedProto-style same-class prototype alignment and FedGH-style learned global heads?

The first mechanism to test is:

`FedProto-L2 -> Global Prototype Classifier (GPC)`

For normalized local feature `z` and normalized global class prototype `p_c`:

`logit_c = s * cosine(z, p_c)`

and

`L_gpc = CE([logit_1, ..., logit_C], y)`.

All global classes must appear in the softmax denominator, including classes absent from the current client's local dataset.

## Non-negotiable research discipline

1. **Establish the baseline before inventing complexity.** First reproduce FedProto and add the smallest possible GPC variant.
2. **One scientific change at a time.** Do not bundle multiple tricks into the first experiment.
3. **Extreme class-missing split first.** The core hypothesis must be tested under severe label-set heterogeneity, not only ordinary Dirichlet non-IID.
4. Report separately:
   - seen-class accuracy;
   - locally-missing-class accuracy;
   - all-class accuracy;
   - per-client macro accuracy;
   - convergence / instability;
   - communication payload if it changes.
5. Keep global prototypes detached from local gradient unless an experiment explicitly tests learnable prototypes.
6. L2-normalize features and prototypes for the primary GPC experiment; expose temperature / scale as a config.
7. Never claim that a missing class appearing in the softmax denominator alone guarantees positive recognition of that class. The experiment must distinguish decision-space transfer from genuine missing-class recognition.
8. Every result must include seed, split definition, dataset, model, optimizer, LR, local epochs, rounds, batch size, prototype update rule, temperature/scale, and commit SHA.
9. Preserve failed runs. Negative evidence matters.
10. Prefer small fast experiments that falsify the mechanism before expensive sweeps.

## Communication protocol

`CHATGPT_CODEX_BRIDGE.md` is the single source of truth for coordination.

At the start of work:
- Read this file and `CHATGPT_CODEX_BRIDGE.md` fully.
- Work only on the current ACTIVE task unless a blocker requires a minimal prerequisite.

During work:
- Commit coherent changes frequently.
- Append progress/result entries to `CHATGPT_CODEX_BRIDGE.md` instead of replacing prior history.
- Do not wait for ChatGPT after every small step. Continue through the current one-hour task autonomously.

At the end of each assigned block, append a report containing:
- `STATUS`: DONE / PARTIAL / BLOCKED
- exact commands run
- files changed
- tests performed
- experiment table if available
- failures / suspicious observations
- interpretation limited to what data support
- recommended next action
- commit SHA(s)

## Code quality

- Keep baseline and proposed method selectable through configuration, not separate drifting codebases.
- Add assertions for class-index/prototype-index alignment.
- Handle classes absent globally in a round without NaNs or accidental zero-vector logits.
- Make aggregation sample-count-aware where appropriate and document the rule.
- Fix random seeds and log client class sets.
- Avoid large refactors until the first mechanism experiment works.

## First-stage comparison

Minimum comparison:

1. Local only (sanity lower bound)
2. FedProto original prototype-alignment objective
3. **FedProto-GPC**: same prototype construction/aggregation, but global prototypes are used as an all-class cosine classifier
4. FedGH if a faithful implementation can be added without delaying the first mechanism test

The decisive early question is whether (3) substantially improves missing-class / all-class behavior over (2) under the same prototype information budget.
