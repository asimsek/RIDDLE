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

## Prepare LHCO data

```bash
python run.py prepare --dataset lhco --catalog config/datasets.yaml \
  --output data/lhco --io-workers 4 --verbose 1
```

### Optional control datasets

```bash
python run.py prepare --variant shifted --output data/lhco_shifted --io-workers 4
python run.py prepare --variant deltaR --output data/lhco_deltaR --io-workers 4
```

- **Shifted:** replace the first two features by `m1 + 0.1*mjj` and `delta_m + 0.1*mjj`, after conversion to TeV.
- **DeltaR:** append the jet angular distance as a fifth feature.

## Run

```bash
python run.py run --methods lacathode --data data/lhco --output results \
  --scenarios signal_injection --seeds 42 --device cuda:0 --io-workers 2 --resume

python run.py run --methods riddle --data data/lhco --output results \
  --scenarios signal_injection --seeds 42 --device cuda:0 \
  --workers 2 --io-workers 2 --mps auto --resume
```

Run both methods and both scenarios with independent seeds:

```bash
python run.py run --methods lacathode riddle --data data/lhco --output results \
  --scenarios signal_injection background_only --seeds 0-9 \
  --device cuda:0 --workers 2 --io-workers 2 --mps auto --resume
```

`--device cpu` for CPU execution.<br>
`--workers` controls concurrent RIDDLE fits on the selected device.<br>
`--io-workers` controls CPU threads per process.<br>
MPS is optional on Linux NVIDIA GPUs; `auto` falls back to ordinary concurrency, while `on` requires MPS.

The runner reads `config/settings.yaml` automatically. Use `--config path/to/settings.yaml` for a separate, self-contained study configuration.<br>
Explicit `--runs`, `--epochs` and `--fractions learned 0.003 0.01` arguments override RIDDLE's YAML values.

To continue compatible checkpoints after an implementation update, add `--resume-across-code-change`.<br>
To move between CUDA GPUs, add `--resume-across-device-change`.<br>
Both require `--resume` and can be combined.

To run a control, change both the data and results locations, for example:

```bash
python run.py run --methods lacathode riddle --data data/lhco_deltaR \
  --output results_deltaR --scenarios signal_injection --seeds 42 \
  --device cuda:0 --workers 2 --io-workers 4 --mps auto --resume
python plot.py --results results_deltaR --output plots_deltaR --methods lacathode riddle
```

Use the corresponding `shifted` locations for the shifted control.

## Plot

```bash
python plot.py --results results --output plots --methods lacathode riddle --verbose 1
```

Request either method alone with `--methods lacathode` or `--methods riddle`.<br>
Add `--overwrite` to regenerate matching plots and tables in an existing output directory without removing other files.
