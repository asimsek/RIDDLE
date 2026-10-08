# RIDDLE on NRP

**RIDDLE: Revealing Irregularities through Distributional Discrepancies in Latent-space via Stein-Witness Estimation**

## NRP Access and Storage (on your local terminal)

```bash
brew install kubectl kubelogin
kubectl version --client
kubectl oidc-login --version
mkdir -p ~/.kube
curl -fSL https://nrp.ai/config -o ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG="$HOME/.kube/config"
kubectl config get-contexts
kubectl auth can-i create pods -n cua-asimsek
kubectl auth can-i create persistentvolumeclaims -n cua-asimsek
```

Authenticate in the browser when prompted. Namespace membership must already be approved. Do not overwrite an existing kubeconfig with that name.<br>
See [NRP access setup](https://nrp.ai/documentation/userdocs/start/getting-started/) if login fails.

```bash
kubectl create -n cua-asimsek -f https://raw.githubusercontent.com/asimsek/RIDDLE/main/config/nrp/shared-storage.yaml
kubectl get pvc riddle-shared -n cua-asimsek
kubectl describe pvc riddle-shared -n cua-asimsek
```

Confirm creation succeeded and the claim is `Bound`, uses `ReadWriteMany`, and has storage class `rook-cephfs`.<br>
If `riddle-shared` already exists, stop and inspect it before proceeding; these instructions do not delete or reuse previous storage automatically.

### JupterHub Setup (on your local terminal)

Download the pod definition:

```bash
curl -fSL https://raw.githubusercontent.com/asimsek/RIDDLE/main/config/nrp/jupyter.yaml -o riddle-jupyter.yaml
```

Then start Jupyter:

```bash
kubectl apply -n cua-asimsek -f riddle-jupyter.yaml
kubectl get pod riddle-jupyter -n cua-asimsek -o wide
kubectl logs -n cua-asimsek riddle-jupyter -c jupyter
```

```bash
nohup kubectl port-forward -n cua-asimsek pod/riddle-jupyter 8888:8888 > "$HOME/riddle-port-forward.log" 2>&1 & echo $! > "$HOME/riddle-port-forward.pid"
kubectl exec -n cua-asimsek riddle-jupyter -- jupyter server list | grep -o 'token=[^ ]*' | tail -1
```

Open `http://localhost:8888` with the token from the log.<br> 
Jupyter is CPU-only, leaving both A100 allocations available for the batch jobs.<br>
It may stay running while they use the same storage.


**!!! CAUTION!!! Delete Pod (if needed):**

```bash
kubectl delete pod riddle-jupyter -n cua-asimsek --grace-period=0 --force --wait=false
```

**!!! CAUTION!!! To stop your port-forward:**

```bash
kill "$(cat "$HOME/riddle-port-forward.pid")"
```


## Jupyter terminal: pull the RIDDLE framework

Open a terminal inside Jupyter and run this once in the fresh project directory:

```bash
cd /shared/work/RIDDLE
git init -b main
git config --global --add safe.directory /shared/work/RIDDLE
git remote add origin https://github.com/asimsek/RIDDLE.git
git pull --ff-only origin main
```

For later updates, run only the following in the Jupyter terminal after all campaign jobs have stopped:

```bash
cd /shared/work/RIDDLE
git pull --ff-only origin main
```

## Jupyter terminal: verify the runtime, source and data

**LaCathode:**

```bash
cd /shared/work/RIDDLE
python scripts/nrp_runtime.py

git clone --no-checkout https://github.com/HEPML-AnomalyDetection/CATHODE.git external/lacathode
git -C external/lacathode checkout --detach 8ead8cd6671b93fc385d8f440c06b8fa870b0be5
python run.py setup
```

**R-ANODE**:

```bash
cd /shared/work/RIDDLE

git clone --no-checkout https://github.com/rd804/R-ANODE.git external/ranode
git -C external/ranode checkout --detach d6deed7cb949eb4483c6f484b96e8e4ff2133e25
python run.py setup --methods ranode
```

**Prepare LHCO data:**

```bash
python run.py prepare --dataset lhco --catalog config/datasets.yaml --output data/lhco --io-workers 16 --verbose 1

python run.py prepare --dataset lhco --catalog config/datasets.yaml --variant deltaR --output data/lhco_deltaR --io-workers 16 --verbose 1

python run.py prepare --dataset lhco --catalog config/datasets.yaml --variant shifted --output data/lhco_shifted --io-workers 16 --verbose 1
```


## Local terminal: batch job submission

Each block submits one job running seeds 40–44 sequentially, requesting **exactly one GPU, 16 CPUs and 64 GiB RAM**.<br>
A100 remains the default accelerator. Separate jobs use separate one-GPU allocations and may run concurrently.

**These examples use `signal_injection` inputs; replace the scenario with `background_only`, or list both scenarios for sequential runs within each job.**

**RIDDLE:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name riddle-seeds40-44 --methods riddle --scenarios signal_injection \
  --seed 40 41 42 43 44 --config config/settings.yaml --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --fits 20 --epochs 100 --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name riddle-bg-seeds40-44 --methods riddle --scenarios background_only \
  --seed 40 41 42 43 44 --config config/settings.yaml --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --fits 20 --epochs 100 --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

**LaCathode:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name lacathode-seeds40-44 --methods lacathode --scenarios signal_injection \
  --seed 40 41 42 43 44 --epochs 100 --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --lacathode-background independent --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name lacathode-bg-seeds40-44 --methods lacathode --scenarios background_only \
  --seed 40 41 42 43 44 --epochs 100 --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --lacathode-background independent --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

Each LaCathode seed trains its own background flow and classifier.

**R-ANODE:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name ranode-seeds40-44 --methods ranode --scenarios signal_injection \
  --fits 20 --epochs 300 --seed 40 41 42 43 44 --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name ranode-bg-seeds40-44 --methods ranode --scenarios background_only \
  --fits 20 --epochs 300 --seed 40 41 42 43 44 --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

**Idealized AD:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name iad-seeds40-44 --methods iad --scenarios signal_injection \
  --seed 40 41 42 43 44 --config config/settings.yaml --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --fits 20 --epochs 100 --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

**Supervised AD:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name supervised-seeds40-44 --methods supervised --scenarios signal_injection \
  --seed 40 41 42 43 44 --config config/settings.yaml --workers 5 --io-workers 2 --torch-threads 2 --mps on \
  --fits 20 --epochs 100 --data data/lhco --results results \
  | kubectl apply -n cua-asimsek -f -
```

For RIDDLE/Idealized/Supervised/R-ANODE, `--fits 20` trains one ensemble with twenty fits.<br>
`--seed 40 41 42 43 44` runs every requested method once per seed; uncertainty bands use the completed seed runs.<br>

For one combined RIDDLE benchmark job, request `--methods riddle iad supervised`; the two oracle methods use the same RIDDLE Stein flow with their pure-reference data roles.<br>
Do not submit that alongside the corresponding standalone jobs for the same result identities.

Optional controls use the preparation commands in `README.md`.<br>
To submit one later, add `--data data/lhco_shifted --results results_shifted` or `--data data/lhco_deltaR --results results_deltaR` to the generator command and choose a new job name.<br>
Each control still requests only one A100 per job.

To continue compatible checkpoints after an implementation update, add `--resume-across-code-change`.<br>
To continue unfinished training on a different CUDA GPU, add `--resume-across-device-change`.<br>
Both require `--resume` (already enabled by `nrp.py`) and can be combined.

Add `--gpu l40` or `--gpu l40s` to any `nrp.py` submission, including injection scans and any method. Omitting `--gpu` keeps the existing A100 request.<br>
Supported values (case-insensitive): `a100`, `l40`, `l40s`, `l4`, `a40`, `rtxa6000`, `rtx8000`, `rtx3090`, `rtx4090`, `h100`, `h200`.

Add `--exclude-node node-2-2.sdsc.optiputer.net` to exclude any server that fails your jobs.<br>
Space-separated multiple servers can be excluded at the same time.


## Monitoring and resuming

### RIDDLE:

```bash
kubectl get jobs,pods -n cua-asimsek -o wide
kubectl get pods -n cua-asimsek -l job-name=riddle-seeds40-44 -o wide
kubectl get pods -n cua-asimsek -l job-name=riddle-bg-seeds40-44 -o wide
```

**Check logs:**

```bash
kubectl logs -n cua-asimsek -f job/riddle-seeds40-44 -c campaign
kubectl logs -n cua-asimsek -f job/riddle-bg-seeds40-44 -c campaign
```

**!!! CAUTION !!! Delete jobs:**

```bash
kubectl delete job riddle-seeds40-44 -n cua-asimsek --ignore-not-found --wait=true
kubectl delete job riddle-bg-seeds40-44 -n cua-asimsek --ignore-not-found --wait=true
```

### LaCathode:

```bash
kubectl get jobs,pods -n cua-asimsek -o wide
kubectl get pods -n cua-asimsek -l job-name=lacathode-seeds40-44 -o wide
kubectl get pods -n cua-asimsek -l job-name=lacathode-bg-seeds40-44 -o wide
```

**Check logs:**

```bash
kubectl logs -n cua-asimsek -f job/lacathode-seeds40-44 -c campaign
kubectl logs -n cua-asimsek -f job/lacathode-bg-seeds40-44 -c campaign
```

**!!! CAUTION !!! Delete jobs:**

```bash
kubectl delete job lacathode-seeds40-44 -n cua-asimsek --ignore-not-found --wait=true
kubectl delete job lacathode-bg-seeds40-44 -n cua-asimsek --ignore-not-found --wait=true
```

### R-ANODE:


```bash
kubectl get jobs,pods -n cua-asimsek -o wide
kubectl get pods -n cua-asimsek -l job-name=ranode-seeds40-44 -o wide
kubectl get pods -n cua-asimsek -l job-name=ranode-bg-seeds40-44 -o wide
```

**Check logs:**

```bash
kubectl logs -n cua-asimsek -f job/ranode-seeds40-44 -c campaign
kubectl logs -n cua-asimsek -f job/ranode-bg-seeds40-44 -c campaign
```

**!!! CAUTION !!! Delete jobs:**

```bash
kubectl delete job ranode-seeds40-44 -n cua-asimsek --ignore-not-found --wait=true
kubectl delete job ranode-bg-seeds40-44 -n cua-asimsek --ignore-not-found --wait=true
```

## Monitor GPU usage of a batch job:

```bash
kubectl exec -n cua-asimsek \
  $(kubectl get pod -n cua-asimsek -l job-name=riddle-seeds40-44 -o jsonpath='{.items[0].metadata.name}') \
  -c campaign -- nvidia-smi
```


## Jupyter terminal: plots

```bash
cd /shared/work/RIDDLE
python scripts/nrp_runtime.py
python plot.py --results results --output plots --verbose 1 --io-workers 16 --overwrite

python paper_plot.py --data data/lhco --results results \
  --config config/settings.yaml --output paper_plots \
  --methods riddle lacathode ranode iad supervised --variants default deltaR shifted \
  --plot-formats png --file-formats csv --io-workers 16 --overwrite --verbose 1
```

All completed methods are discovered automatically.<br>
Request a method directly with `--methods lacathode`, `--methods riddle`, `--methods ranode`, `--methods iad`, or `--methods supervised`.<br>
Add `--overwrite` to regenerate matching plots and tables.


## Optional injection scan

Prepare each strength once with the same population settings as the normal runs; all methods use seeds 40–44.

```bash
cd /shared/work/RIDDLE

python run.py prepare-scan --config config/settings.yaml --output data/injection_scan --io-workers 16 --resume

python run.py prepare-scan --config config/settings.yaml --variant deltaR --output data/injection_scan_deltaR --io-workers 16 --resume

python run.py prepare-scan --config config/settings.yaml --variant shifted --output data/injection_scan_shifted --io-workers 16 --resume
```

After preparation is complete, submit the jobs below from your local terminal.

**RIDDLE:**

```bash
for SEED in 40 41 42 43 44; do
  kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
    python /shared/work/RIDDLE/nrp.py \
    --workflow scan --name "riddle-injection-scan-seed${SEED}" --methods riddle --seed "$SEED" \
    --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
    --reuse-results results --resume-across-code-change \
    --fits 20 --epochs 100 --workers 5 --scan-bg-workers 7 --io-workers 2 --torch-threads 2 --mps on \
    | kubectl apply -n cua-asimsek -f - || break
done
```

**Idealized AD:**

```bash
for SEED in 40 41 42 43 44; do
  kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
    python /shared/work/RIDDLE/nrp.py \
    --workflow scan --name "iad-injection-scan-seed${SEED}" --methods iad --seed "$SEED" \
    --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
    --reuse-results results --resume-across-code-change \
    --fits 20 --epochs 100 --workers 5 --scan-bg-workers 7 --io-workers 2 --torch-threads 2 --mps on \
    | kubectl apply -n cua-asimsek -f - || break
done
```

**Supervised AD:**

```bash
for SEED in 40 41 42 43 44; do
  kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
    python /shared/work/RIDDLE/nrp.py \
    --workflow scan --name "supervised-injection-scan-seed${SEED}" --methods supervised --seed "$SEED" \
    --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
    --reuse-results results --resume-across-code-change \
    --fits 20 --epochs 100 --workers 5 --scan-bg-workers 7 --io-workers 2 --torch-threads 2 --mps on \
    | kubectl apply -n cua-asimsek -f - || break
done
```

**LaCathode:**

```bash
for SEED in 40 41 42 43 44; do
  kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
    python /shared/work/RIDDLE/nrp.py \
    --workflow scan --name "lacathode-injection-scan-seed${SEED}" --methods lacathode --seed "$SEED" \
    --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
    --reuse-results results --resume-across-code-change \
    --epochs 100 --workers 5 --io-workers 2 --torch-threads 2 --mps on \
    --lacathode-background independent \
    | kubectl apply -n cua-asimsek -f - || break
done
```

**R-ANODE:**

```bash
for SEED in 40 41 42 43 44; do
  kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
    python /shared/work/RIDDLE/nrp.py \
    --workflow scan --name "ranode-injection-scan-seed${SEED}" --methods ranode --seed "$SEED" \
    --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
    --reuse-results results --resume-across-code-change \
    --fits 20 --epochs 300 --workers 5 --io-workers 2 --torch-threads 2 --mps on \
    | kubectl apply -n cua-asimsek -f - || break
done
```


In Jupyter, plot completed scan results:

```bash
python plot.py --results results_injection_scan --output plots_injection_scan --verbose 1  --io-workers 16 --overwrite

python paper_plot.py --scan-data data/injection_scan --scan-results results_injection_scan \
  --config config/settings.yaml --output paper_plots \
  --methods riddle lacathode ranode iad supervised --variants default deltaR shifted \
  --plot-formats png --file-formats csv --io-workers 16 --overwrite --verbose 1
```


## Troubleshooting

- Pending job: inspect `kubectl describe pod <name> -n cua-asimsek` for GPU availability, quota or storage events.
