# RIDDLE on NRP

RIDDLE: Residual Identification of Distributional Deviations in Latent-space through density Estimation

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
python run.py prepare --dataset lhco --catalog config/datasets.yaml --output data/lhco --io-workers 8 --verbose 1
```


## Local terminal: RIDDLE, LaCathode, and R-ANODE batch job submission

Each block submits one seed-42 job requesting **one A100, 16 CPUs and 64 GiB RAM**.<br>
Together they use two separate GPU allocations, not a two-GPU request. They may run concurrently and write to separate method directories.

**These examples use `signal_injection` inputs; replace the scenario with `background_only`, or list both scenarios for sequential runs within each job.**

**RIDDLE:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name riddle-seed42 --methods riddle --scenarios signal_injection \
  --seeds 42 --config config/settings.yaml --workers 4 --io-workers 16 \
  --runs 10 --fits 10 --epochs 100 | kubectl apply -n cua-asimsek -f -
```

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name riddle-bg-seed42 --methods riddle --scenarios background_only \
  --seeds 42 --config config/settings.yaml --workers 4 --io-workers 16 \
  --runs 10 --fits 10 --epochs 100 | kubectl apply -n cua-asimsek -f -
```

**LaCathode:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name lacathode-seed42 --methods lacathode --scenarios signal_injection \
  --seeds 42 --runs 10 --epochs 100 --workers 2 --io-workers 16 \
  --lacathode-background independent \
  | kubectl apply -n cua-asimsek -f -
```

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name lacathode-bg-seed42 --methods lacathode --scenarios background_only \
  --seeds 42 --runs 10 --epochs 100 --workers 2 --io-workers 16 \
  --lacathode-background independent \
  | kubectl apply -n cua-asimsek -f -
```

Add `--lacathode-background fixed` to a LaCathode submission to share one background flow across `--runs` classifiers; the default is `independent`.

**R-ANODE:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name ranode-seed42 --methods ranode --scenarios signal_injection \
  --runs 10 --fits 10 --epochs 100 --seeds 42 --workers 4 --io-workers 16 \
  | kubectl apply -n cua-asimsek -f -
```

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --name ranode-bg-seed42 --methods ranode --scenarios background_only \
  --runs 10 --fits 10 --epochs 100 --seeds 42 --workers 4 --io-workers 16 \
  | kubectl apply -n cua-asimsek -f -
```

For RIDDLE/R-ANODE, `--fits 10` trains one ensemble with ten signal fits.<br>
Add `--runs 10` to retrain the complete method three times, including separate background models; method uncertainty bands use these independent runs.<br>

For one combined job instead, request `--methods lacathode riddle ranode` with a different job name.<br>
Do not submit that alongside these three jobs for the same result identities.

Optional controls use the preparation commands in `README.md`.<br>
To submit one later, add `--data data/lhco_shifted --results results_shifted` or `--data data/lhco_deltaR --results results_deltaR` to the generator command and choose a new job name.<br>
Each control still requests only one A100 per job.

To continue compatible checkpoints after an implementation update, add `--resume-across-code-change`.<br>
To continue unfinished training on a different CUDA GPU, add `--resume-across-device-change`.<br>
Both require `--resume` (already enabled by `nrp.py`) and can be combined.

Add `--gpu l40` or `--gpu l40s` to any `nrp.py` submission, including injection scans and either method. Omitting `--gpu` keeps the existing A100 request.<br>
Supported values (case-insensitive): `a100`, `l40`, `l40s`, `l4`, `a40`, `rtxa6000`, `rtx8000`, `rtx3090`, `rtx4090`, `h100`, `h200`.


## Monitoring and resuming

### RIDDLE:

```bash
kubectl get jobs,pods -n cua-asimsek -o wide
kubectl get pods -n cua-asimsek -l job-name=riddle-seed42 -o wide
kubectl get pods -n cua-asimsek -l job-name=riddle-bg-seed42 -o wide
```

**Check logs:**

```bash
kubectl logs -n cua-asimsek -f job/riddle-seed42 -c campaign
kubectl logs -n cua-asimsek -f job/riddle-bg-seed42 -c campaign
```

**!!! CAUTION !!! Delete jobs:**

```bash
kubectl delete job riddle-seed42 -n cua-asimsek --ignore-not-found --wait=true
kubectl delete job riddle-bg-seed42 -n cua-asimsek --ignore-not-found --wait=true
```

### LaCathode:

```bash
kubectl get jobs,pods -n cua-asimsek -o wide
kubectl get pods -n cua-asimsek -l job-name=lacathode-seed42 -o wide
kubectl get pods -n cua-asimsek -l job-name=lacathode-bg-seed42 -o wide
```

**Check logs:**

```bash
kubectl logs -n cua-asimsek -f job/lacathode-seed42 -c campaign
kubectl logs -n cua-asimsek -f job/lacathode-bg-seed42 -c campaign
```

**!!! CAUTION !!! Delete jobs:**

```bash
kubectl delete job lacathode-seed42 -n cua-asimsek --ignore-not-found --wait=true
kubectl delete job lacathode-bg-seed42 -n cua-asimsek --ignore-not-found --wait=true
```

### R-ANODE:


```bash
kubectl get jobs,pods -n cua-asimsek -o wide
kubectl get pods -n cua-asimsek -l job-name=ranode-seed42 -o wide
kubectl get pods -n cua-asimsek -l job-name=ranode-bg-seed42 -o wide
```

**Check logs:**

```bash
kubectl logs -n cua-asimsek -f job/ranode-seed42 -c campaign
kubectl logs -n cua-asimsek -f job/ranode-bg-seed42 -c campaign
```

**!!! CAUTION !!! Delete jobs:**

```bash
kubectl delete job ranode-seed42 -n cua-asimsek --ignore-not-found --wait=true
kubectl delete job ranode-bg-seed42 -n cua-asimsek --ignore-not-found --wait=true
```


## Jupyter terminal: plots

```bash
cd /shared/work/RIDDLE
python scripts/nrp_runtime.py
python plot.py --results results --output plots --verbose 1 --io-workers 16 --overwrite
```

All completed methods are discovered automatically.<br>
Request either method alone with `--methods lacathode`, `--methods riddle`, or `--methods ranode`.<br>
Add `--overwrite` to regenerate matching plots and tables.


## Optional injection scan

In the Jupyter terminal, prepare the configured strengths and replicas once.

```bash
cd /shared/work/RIDDLE
python run.py prepare-scan --config config/settings.yaml --output data/injection_scan --io-workers 4 --resume
```

After preparation is complete, submit the jobs below from your local terminal.

**RIDDLE:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --workflow scan --name riddle-injection-scan --methods riddle \
  --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
  --runs 10 --fits 10 --epochs 100 --workers 2 --io-workers 4 | \
  kubectl apply -n cua-asimsek -f -
```

**LaCathode:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --workflow scan --name lacathode-injection-scan --methods lacathode \
  --config config/settings.yaml --data data/injection_scan --results results_injection_scan \
  --runs 10 --epochs 100 --workers 1 --io-workers 4 | \
  kubectl apply -n cua-asimsek -f -
```

**R-ANODE:**

```bash
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  python /shared/work/RIDDLE/nrp.py \
  --workflow scan --name ranode-injection-scan --methods ranode \
  --data data/injection_scan --results results_injection_scan \
  --runs 10 --fits 10 --epochs 100 --workers 1 --io-workers 8 | \
  kubectl apply -n cua-asimsek -f -
```

In Jupyter, plot completed scan results:

```bash
python plot.py --results results_injection_scan --output plots_injection_scan --verbose 1  --io-workers 16 --overwrite
```


## Troubleshooting

- Pending job: inspect `kubectl describe pod <name> -n cua-asimsek` for GPU availability, quota or storage events.
