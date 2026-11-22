# RIDDLE

**RIDDLE: Revealing Irregularities through Distributional Discrepancies in Latent-space via Stein-Witness Estimation**

## Setup

> [!IMPORTANT]
> For NRP, follow [README_NRP.md](README_NRP.md) for setup, data preparation, batch submission, and plotting.<br>
> The instructions below apply to local machines and other computing environments.<br><br>
> NRP uses the pre-built `ghcr.io/asimsek/riddle-runtime:v1` image, pinned by digest, for Jupyter and GPU jobs.<br>
> No environment creation or package installation is needed on NRP.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

**Pull LaCathode framework from original GitHub repo:**

```bash
git clone --no-checkout https://github.com/HEPML-AnomalyDetection/CATHODE.git external/lacathode
git -C external/lacathode checkout --detach 8ead8cd6671b93fc385d8f440c06b8fa870b0be5
python run.py setup
```

**Pull R-ANODE framework from original GitHub repo:**

```bash
git clone --no-checkout https://github.com/rd804/R-ANODE.git external/ranode
git -C external/ranode checkout --detach d6deed7cb949eb4483c6f484b96e8e4ff2133e25
python run.py setup --methods ranode
```

## Prepare LHCO data

```bash
python run.py prepare --dataset lhco --catalog config/datasets.yaml --output data/lhco --io-workers 16 --verbose 1
```

### Optional control datasets

```bash
python run.py prepare --dataset lhco --catalog config/datasets.yaml --variant deltaR --output data/lhco_deltaR --io-workers 16 --verbose 1
python run.py prepare --dataset lhco --catalog config/datasets.yaml --variant shifted --output data/lhco_shifted --io-workers 16 --verbose 1
```

- **Shifted:** replace the first two features by `m1 + 0.1*mjj` and `delta_m + 0.1*mjj`, after conversion to TeV.
- **DeltaR:** append the jet angular distance as a fifth feature.

## Run

**RIDDLE:**

```bash
python run.py run --methods riddle --data data/lhco --output results \
  --scenarios signal_injection background_only --seed 40 41 42 43 44 --config config/settings.yaml \
  --fits 20 --epochs 200 --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

**LaCathode:**

```bash
python run.py run --methods lacathode --data data/lhco --output results \
  --scenarios signal_injection background_only --seed 40 41 42 43 44 --config config/settings.yaml \
  --epochs 100 --lacathode-background independent \
  --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

**R-ANODE:**

```bash
python run.py run --methods ranode --data data/lhco --output results \
  --scenarios signal_injection background_only --seed 40 41 42 43 44 --config config/settings.yaml \
  --fits 20 --epochs 300 --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

**Idealized AD:**

```bash
python run.py run --methods iad --data data/lhco --output results \
  --scenarios signal_injection --seed 40 41 42 43 44 --config config/settings.yaml \
  --fits 20 --epochs 200 --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

**Supervised AD:**

```bash
python run.py run --methods supervised --data data/lhco --output results \
  --scenarios signal_injection --seed 40 41 42 43 44 --config config/settings.yaml \
  --fits 20 --epochs 200 --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

Use `--device cpu` for CPU execution. On Linux with an NVIDIA GPU, `--mps auto` uses MPS when available.
`--workers` controls concurrent fits or LaCathode cases, `--io-workers` controls I/O concurrency, and `--torch-threads` sets CPU compute threads per worker.

`--fits` controls the ensemble fits inside RIDDLE/Idealized/Supervised/R-ANODE.<br>
`--seed 40 41 42 43 44` runs each listed seed once for every requested method; default = 42.<br>
`--epochs` controls the native RIDDLE/Idealized/Supervised/R-ANODE fit epochs and LaCathode classifier epochs.<br>
Each LaCathode seed trains its own background flow and classifier; the default background mode is `independent`.

For all methods, to continue compatible checkpoints after an implementation update, add `--resume-across-code-change`.<br>
To continue unfinished training on a different CUDA GPU, add `--resume-across-device-change`.<br>
Both require `--resume` and can be combined.

For control runs, use the same method commands with `--data data/lhco_deltaR --output results_deltaR` or `--data data/lhco_shifted --output results_shifted`.

## Plot

```bash
python paper_plot.py --data data/lhco --results results --focused \
  --config config/settings.yaml --output paper_plots_all \
  --methods riddle iad supervised lacathode --variants default deltaR shifted \
  --plot-formats pdf --file-formats csv --io-workers 16 --plot-workers 16 --overwrite --verbose 1
```

Omit `--focused` for the full set of normal plots.

## Optional signal-strength scan

```bash
python run.py prepare-scan --config config/settings.yaml --output data/injection_scan --io-workers 16 --resume
```

For a control scan, append `_deltaR` or `_shifted` to the `--data`, `--output`, and `--reuse-results` paths in the scan commands.

```bash
python run.py prepare-scan --config config/settings.yaml --variant deltaR --output data/injection_scan_deltaR --io-workers 16 --resume
python run.py prepare-scan --config config/settings.yaml --variant shifted --output data/injection_scan_shifted --io-workers 16 --resume
```

```bash
python run.py scan --methods riddle iad supervised --seed 40 41 42 43 44 --config config/settings.yaml \
  --data data/injection_scan --output results_injection_scan \
  --reuse-results results --resume-across-code-change --resume-across-device-change \
  --fits 20 --epochs 200 --scan-bg-workers 7 \
  --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

```bash
python run.py scan --methods lacathode --seed 40 41 42 43 44 --config config/settings.yaml \
  --data data/injection_scan --output results_injection_scan \
  --reuse-results results --resume-across-code-change --resume-across-device-change \
  --epochs 100 --lacathode-background independent \
  --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

```bash
python run.py scan --methods ranode --seed 40 41 42 43 44 --config config/settings.yaml \
  --data data/injection_scan --output results_injection_scan \
  --reuse-results results --resume-across-code-change --resume-across-device-change \
  --fits 20 --epochs 300 \
  --device cuda:0 --workers 5 --io-workers 2 --torch-threads 2 --mps auto --resume
```

```bash
python paper_plot.py --scan-data data/injection_scan --scan-results results_injection_scan \
  --config config/settings.yaml --output paper_plots \
  --methods riddle iad supervised lacathode --variants default \
  --checkpoint-policy both --last-epoch 200 \
  --plot-formats pdf --file-formats csv --io-workers 16 --plot-workers 16 --overwrite --verbose 1
```
