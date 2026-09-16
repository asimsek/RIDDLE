# RIDDLE

**RIDDLE: Residual Identification of Distributional Deviations in Latent-space through density Estimation**

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
python run.py prepare --dataset lhco --catalog config/datasets.yaml --output data/lhco --io-workers 8 --verbose 1
```

### Optional control datasets

```bash
python run.py prepare --variant shifted --output data/lhco_shifted --io-workers 8
python run.py prepare --variant deltaR --output data/lhco_deltaR --io-workers 8
```

- **Shifted:** replace the first two features by `m1 + 0.1*mjj` and `delta_m + 0.1*mjj`, after conversion to TeV.
- **DeltaR:** append the jet angular distance as a fifth feature.

## Run


**RIDDLE:**

```bash
python run.py run --methods riddle --data data/lhco --output results \
  --scenarios signal_injection --seeds 42 --device cuda:0 --runs 10 --epochs 100 \
  --workers 2 --io-workers 8 --mps auto --resume
```

**LaCathode:**

```bash
python run.py run --methods lacathode --data data/lhco --output results \
  --scenarios signal_injection --seeds 42 --device cuda:0 --runs 10 --epochs 100 \
  --lacathode-background fixed --workers 1 --io-workers 8 --mps auto --resume
```

**R-ANODE:**

```bash
python run.py run --methods ranode --data data/lhco --output results \
  --scenarios signal_injection background_only --seeds 42 --device cuda:0 \
  --runs 10 --epochs 100 \
  --io-workers 8 --resume
```

Run all three methods and both scenarios (`signal_injection`, `background_only`):

```bash
python run.py run --methods lacathode riddle ranode --data data/lhco --output results \
  --scenarios signal_injection background_only --seeds 42 \
  --runs 10 --epochs 100 \
  --device cuda:0 --workers 2 --io-workers 8 --mps auto --resume
```

`--device cpu` for CPU execution.<br>
`--workers` controls concurrent RIDDLE/R-ANODE signal fits or complete independent LaCathode runs; fixed-background LaCathode stays sequential.<br>
`--io-workers` controls CPU threads per process.<br>
MPS is optional on Linux NVIDIA GPUs; `auto` falls back to ordinary concurrency, while `on` requires MPS.

`--runs` controls RIDDLE/R-ANODE signal fits and independent LaCathode background-flow-plus-classifier runs.<br>
`--epochs` controls signal/classifier epochs.<br>
`--lacathode-background fixed` shares same LaCathode background flow across `--runs` classifiers; default = `independent`.

For all three methods, to continue compatible checkpoints after an implementation update, add `--resume-across-code-change`.<br>
To move between CUDA GPUs, add `--resume-across-device-change`.<br>
Both require `--resume` and can be combined.

To run a control, change both the data and results locations, for example:

```bash
python run.py run --methods lacathode riddle ranode --data data/lhco_deltaR \
  --output results_deltaR --scenarios signal_injection --seeds 42 \
  --device cuda:0 --workers 2 --io-workers 8 --mps auto --resume
python plot.py --results results_deltaR --output plots_deltaR --methods lacathode riddle ranode
```

Use the corresponding `shifted` locations for the shifted control.

## Plot

```bash
python plot.py --results results --output plots --methods lacathode riddle ranode --verbose 1 --overwrite
```

Plotting uses CPU by default; add `--device cuda:0 --io-workers 8` to accelerate RIDDLE checkpoint inference.<br>

Request either method alone with `--methods lacathode` or `--methods riddle`.<br>
Add `--overwrite` to regenerate matching plots and tables in an existing output directory without removing other files.


## Optional signal-strength scan

```bash
python run.py prepare-scan --config config/settings.yaml --output data/injection_scan --io-workers 8 --resume
```

```bash
python run.py scan --methods lacathode riddle ranode --config config/settings.yaml \
  --data data/injection_scan --output results_injection_scan \
  --runs 10 --epochs 100 \
  --device cuda:0 --workers 2 --io-workers 8 --mps auto --resume
```

```bash
python plot.py --results results_injection_scan --output plots_injection_scan --methods lacathode riddle ranode --verbose 1 --overwrite
```
