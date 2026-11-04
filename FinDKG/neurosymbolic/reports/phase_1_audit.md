# Phase 1: repository and dataset audit

## Scope and outcome

Read-only inspection completed before adding this experimental scaffold. The original source, checkpoints, data and outputs were not altered. This report corrects the initial assumption that the named dataset directories were not available: both graph dataset variants are present and have distinct roles.

## Dataset inventory

| Directory | Train | Validation | Test | Total | Time IDs | Purpose |
|---|---:|---:|---:|---:|---|---|
| `FinDKG_dataset/FinDKG` | 119,549 | 11,444 | 13,069 | 144,062 | 0–125 | Study/evaluation benchmark |
| `FinDKG_dataset/FinDKG-full` | 222,732 | 9,404 | 10,013 | 242,149 | 0–260 | Larger future-research dataset |

Both report 13,645 entities and 15 relations. The study benchmark split time ranges are train 0–99, validation 100–112, test 113–125. Full dataset ranges are train 0–233, validation 234–246, test 247–260. Quadruple intersections across splits were zero in the scan. The full dataset provides `time2id.txt`; the study dataset does not. Study entity types are named (12 types); the full entity map stores type IDs numerically.

## Architecture and execution

`train_DKG_run.py` chooses `graph_mode = "FinDKG"`, loads the graph via `DKG.data.load_temporal_knowledge_graph`, builds chronological batches using `collate_fn`, initializes `DynamicGraphModel` with `EmbeddingUpdater`, `Combiner`, `EdgeModel`, and `InterEventTimeModel`, trains edge/time objectives with truncated backpropagation, validates/early-stops, then evaluates a selected checkpoint. `KGTransformer` graph-attention layers are in `DKG/model/gnn/kgt.py`; temporal/recurrent and embedding/model components are in `DKG/model`.

The loader in `DKG/data.py` expects data under DKG's `data/<graph_name>` runtime path, while the repository dataset is stored under `FinDKG_dataset`; an experiment must copy/link data into a separate run data root or pass a compatible root. The script currently uses seed 41, max 150 epochs, truncation 50, patience 20, 200-dimensional embedding settings, 2 graph-convolution layers, 8 attention heads, and learning rate 0.0005. This does not establish the exact configuration for the supplied baseline (reported effective patience 30).

## News-to-graph stage

No article corpus, article identifiers/provenance, original ICKG weights, corpus extraction pipeline, annotation set, or extraction output is present. `notebooks/02_ickg_extraction_demo.py` calls Gemini using `GEMINI_API_KEY` and processes five hard-coded sample strings. It is an illustrative API demo and is disconnected from graph construction/training. The quadruple datasets cannot serve as evidence for extraction reliability because they contain no article text or claim-to-source alignment.

## Baseline artifact status

No `.pt`/`.pth` checkpoints or training logs are included. Existing figures and CSV tables are present under `outputs`, but they do not contain the run artifacts required to reproduce or independently validate the baseline scores. The README contains reported outcomes; those remain unverified from this checkout.

## Minimum requirements for later phases

1. Article corpus with reuse permission and article text, publication time, source and stable document ID; preferably URLs/authors and original extraction alignments.
2. Executable extraction model and weights, or credentials/approval for an external extraction API. The demo specifically expects a Gemini API key.
3. For independent baseline verification, the original checkpoints/logs or an authorized fresh rerun with environment/hardware details and agreed config.
4. For claim-level reliability metrics, annotated reference claims on a time-safe sample; graph prediction scores alone cannot measure extraction correctness.

The graph-only extension now has a symbolic consistency auditor for identifier ranges, type-map coverage, split timestamp ranges, duplicates, and temporal order. These constraints can find malformed or structurally inconsistent rows, but cannot establish source support or objective truth. No extraction or model-improvement claims are made.
