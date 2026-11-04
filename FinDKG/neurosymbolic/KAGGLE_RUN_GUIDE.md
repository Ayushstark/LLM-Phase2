# Kaggle run guide

## What this run compares

- **Baseline:** original KGTransformer and unmodified FinDKG train/validation/test splits.
- **Neuro-symbolic:** same KGTransformer, same seed and hyperparameters; symbolic relation/head-type/tail-type signatures are learned from train time IDs 0–79 and applied only to training rows at time IDs 80–99. Validation and test are copied unchanged. All rejections are audited.

This is a graph-only symbolic consistency hypothesis. It does not validate claims against news articles. A symbolic rule may reject valid new relation/type patterns, so report rejected-row coverage next to link prediction metrics.

## 1. Prepare the Kaggle notebook

1. Create a new Kaggle Notebook and set **Accelerator = GPU T4** (or another available GPU). Turn **Internet = On** for the setup cell.
2. Upload `neurosymbolic_kaggle_bundle.zip` as a private Kaggle Dataset, add it to the notebook as an input, and note its input path.
3. Use a fresh notebook session. Do not edit or train inside the cloned source files; the runner copies the training script into isolated run folders.

## 2. Clone the repository and add the experiment files

```python
%cd /kaggle/working
!git clone https://github.com/DARK-1926/LLM-Project.git
!unzip -o /kaggle/input/REPLACE_WITH_YOUR_DATASET_SLUG/neurosymbolic_kaggle_bundle.zip -d /kaggle/working/LLM-Project/FinDKG
%cd /kaggle/working/LLM-Project/FinDKG
```

The repository clone supplies the two FinDKG graph datasets and original model. The uploaded zip supplies the uncommitted neuro-symbolic extension. Do not use the Kaggle `FinDKG-full` data for this first controlled run.

## 3. Enable and verify the GPU

```python
!nvidia-smi
```

Confirm an NVIDIA GPU is listed. If this fails, stop and enable the GPU accelerator in notebook settings before installing packages or training.

## 4. Install the known-compatible environment

These pins match the supplied Kaggle run transcript (Python 3.10, PyTorch 2.1.0, CUDA 12.1, DGL and `torch_scatter`). Keep the cell output as environment provenance.

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
!/usr/local/bin/uv venv --clear /kaggle/working/my_py310_env --python 3.10
!/usr/local/bin/uv pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --python /kaggle/working/my_py310_env/bin/python
!/usr/local/bin/uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html --python /kaggle/working/my_py310_env/bin/python
!/usr/local/bin/uv pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --python /kaggle/working/my_py310_env/bin/python
!/usr/local/bin/uv pip install pandas numpy==1.26.4 networkx scikit-learn tqdm matplotlib --python /kaggle/working/my_py310_env/bin/python
```

If a wheel-install step fails, stop and retain the error output. Do not silently switch package versions, since DGL and `torch_scatter` must match the installed PyTorch/CUDA build.

## 5. Check imports, GPU, and data before training

```python
%cd /kaggle/working/LLM-Project/FinDKG
!/kaggle/working/my_py310_env/bin/python -c "import torch,dgl,torch_scatter,pandas,numpy; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda); print('dgl',dgl.__version__); print('scatter',torch_scatter.__version__); assert torch.cuda.is_available()"
!wc -l FinDKG_dataset/FinDKG/train.txt FinDKG_dataset/FinDKG/valid.txt FinDKG_dataset/FinDKG/test.txt
```

Expected dataset row counts are 119,549 / 11,444 / 13,069. Stop if counts differ or CUDA reports unavailable.

## 6. Run the paired experiment

```python
!mkdir -p /kaggle/working/experiment_output
!/kaggle/working/my_py310_env/bin/python neurosymbolic/run_experiment.py --dataset FinDKG --run-id kaggle_findkg_seed41 --train --max-epochs 150 --patience 20
```

This runs baseline and symbolic conditions sequentially with seed 41 and the checked-in script settings. It streams logs to the notebook and saves each condition's console log, checkpoints, and result files under:

`/kaggle/working/LLM-Project/FinDKG/neurosymbolic/results/kaggle_findkg_seed41/`

The symbolic preflight should report the rule fit interval, signature count and number of rejected training rows. It must show validation and test as unchanged. The current implementation filters only the final 20% of the training times; do not tune rules based on test results.

## 7. Inspect metrics and preserve outputs

```python
import json
from pathlib import Path
p = Path('/kaggle/working/LLM-Project/FinDKG/neurosymbolic/results/kaggle_findkg_seed41/run_manifest.json')
m = json.loads(p.read_text())
print('status:', m['status'])
print('audits:', json.dumps(m['audits'], indent=2))
print('metrics:', json.dumps(m.get('metrics'), indent=2))
print('training:', json.dumps(m.get('training'), indent=2))
assert m['status'] == 'training_complete'
assert m['metrics']['baseline'] is not None and m['metrics']['symbolic'] is not None
```

Archive the complete run directory, including checkpoints and logs, before the Kaggle session ends:

```python
%cd /kaggle/working/LLM-Project/FinDKG
!zip -qr /kaggle/working/findkg_neurosymbolic_results.zip neurosymbolic/results/kaggle_findkg_seed41
```

Download `findkg_neurosymbolic_results.zip` from the notebook output/file browser or commit the notebook version so `/kaggle/working` artifacts are retained. Keep the run manifest, console logs, audit files and checkpoints together.

## Reporting

Compare the baseline and symbolic test metrics as a paired run under identical settings. Also report the early-train fit window, learned signature count, rows filtered from late-train, retained-train percentage, and unchanged validation/test counts. The pasted R-GCN run and user-supplied baseline metrics are reference context only; they are not same-run KGTransformer controls. One seed does not support statistical significance claims.
