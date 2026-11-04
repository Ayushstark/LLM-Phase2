# Neuro-Symbolic FinDKG experiments

This directory is an isolated, additive experiment area. The original DKG implementation, dataset files, and existing outputs remain untouched.

## Dataset roles established from this checkout

- `../FinDKG_dataset/FinDKG` is the controlled study benchmark used by the current training script (`graph_mode = "FinDKG"`). It has 13,645 entities, 15 relations, and 144,062 quadruples across chronological train/validation/test periods (time IDs 0–99, 100–112, and 113–125).
- `../FinDKG_dataset/FinDKG-full` is the larger future-research variant: 242,149 quadruples, time IDs 0–260, and a `time2id.txt` date mapping. It is not the initial controlled benchmark.

These are graph datasets, not raw news articles. The notebook `../notebooks/02_ickg_extraction_demo.py` is an API demonstration over five hard-coded examples; it is not the original ICKG model or the source corpus pipeline.

## Phased implementation status

1. Repository and dataset audit: complete. Findings are in `reports/phase_1_audit.md`.
2. Article acquisition and extraction setup: unavailable for this study. No article text is supplied. Do not infer article text from graph triples or present graph-only filtering as extraction validation.
3. Graph-level symbolic constraints: implemented in `graph_symbolic.py`. It checks ID ranges, type-map coverage, split time bounds, cross-split overlap, and chronological ordering. Repeated quadruples/facts are recorded as recurrence statistics and preserved. A relation/head-type/tail-type signature ontology is inferred using only the earliest 80% of training time points and applied to the final 20% of train. Validation and test are copied unchanged. All rejected training rows are written to an audit file. It cannot validate source evidence or objective truth, and may reject valid novel signatures.
4. Neural-symbolic controlled experiments: prepared on the study benchmark. The neural component is the unchanged KGTransformer; the symbolic component applies the train-only relation/type constraint before graph loading. Baseline checkpoints/logs are absent from this checkout; the checked-in training config differs from the stated baseline patience.

The isolated experiment runner is `run_experiment.py`. It copies both conditions under a unique run folder, applies symbolic rules, hashes source inputs, records environment/config, launches each condition with separate logs/checkpoints, and parses test metrics into the manifest. It patches the legacy logger within each copied run script so outputs stay inside the run folder. Example:

```powershell
python neurosymbolic/run_experiment.py --dataset FinDKG --run-id my_run --train --max-epochs 150 --patience 20
```

The current local workspace lacks DGL and `torch_scatter`. The Kaggle run bundle contains the current runner and rules. The symbolic type-signature rule is fit from train only; tune no thresholds against validation/test outcomes. Treat filtered edge coverage and downstream prediction performance as separate outcomes.

On the current `FinDKG` split, the early 80% training window is time IDs 0–79 and the application window is 80–99. With minimum signature support 1, the preflight learns 1,571 signatures and rejects 28 late-training rows; validation/test remain untouched. This is a small perturbation and any measured effect may be negligible. The exact audit and coverage are regenerated in each run.

See `KAGGLE_RUN_GUIDE.md` for notebook setup, the known-compatible environment pins, the GPU checks, the paired run command, and output archiving steps.

## Running the API-independent checks

From `FinDKG` run:

```powershell
python -m unittest discover -s neurosymbolic -p "test_*.py"
```

The checks exercise claim-schema behavior and graph-level symbolic constraints on small software fixtures. Run a real dataset audit with:

```powershell
python neurosymbolic/graph_symbolic.py FinDKG_dataset/FinDKG neurosymbolic/results/FinDKG_symbolic_audit
```

Passing software checks or graph consistency audits does not establish extraction accuracy, article evidence, or financial truth.

## Pasted Kaggle run artifact

The user supplied a Kaggle notebook transcript that includes an R-GCN+RNN run on `FinDKG` with seed 41, a 15 GB Kaggle T4 noted in the script context, and patience 20. Logged test results: loss 137.3764, MRR 0.116748, Rec@1 0.073839, Rec@3 0.128778, Rec@10 0.198561, Rec@100 0.369959. This is a supplied run transcript, not a checkpoint independently reproduced in this workspace. It is an R-GCN variant, not the KGTransformer baseline and not an extraction/verification experiment.

## Data layout

- `data/articles/`: source articles in JSONL (one record per article; retain text, timestamp, source, URL/document ID, author and metadata).
- `data/claims/`: raw candidate and adjudicated claim JSONL.
- `data/verified_graph/`: exported four-column FinDKG-compatible candidates, with provenance kept separately.
- `rules/`: versioned symbolic constraints.
- `configs/`: experiment configuration snapshots.
- `results/`: run summaries and audit logs.
- `reports/`: phase reports and final research report.

No synthetic article or claim is supplied as research data. The example in tests is clearly synthetic and used only to check software behavior.
