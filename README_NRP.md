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

Then start the setup pod:

```bash
kubectl apply -n cua-asimsek -f https://raw.githubusercontent.com/asimsek/RIDDLE/main/config/nrp/jupyter.yaml
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

**!!! CAUTION!!! To kill the process/port-forward:**

```bash
kill "$(cat "$HOME/qlacathode-port-forward.pid")"
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

## Jupyter terminal: shared environment, source and data

```bash
cd /shared/work/RIDDLE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is not None'
python scripts/nrp_runtime.py --freeze

git clone --no-checkout https://github.com/HEPML-AnomalyDetection/CATHODE.git external/lacathode
git -C external/lacathode checkout --detach 8ead8cd6671b93fc385d8f440c06b8fa870b0be5
python run.py setup

python run.py prepare --dataset lhco --catalog config/datasets.yaml --output data/lhco --io-workers 4 --verbose 1
```

## Local terminal: two independent A100 jobs

Each block submits one seed-42 job requesting **one A100, 16 CPUs and 64 GiB RAM**.<br>
Together they use two separate GPU allocations, not a two-GPU request. They may run concurrently and write to separate method directories. 

These examples use `signal_injection` inputs; replace the scenario with `background_only`, or list both scenarios for sequential runs within each job.

**RIDDLE, using the YAML settings:**

```bash
riddle_image=$(kubectl get pod riddle-jupyter -n cua-asimsek \
  -o jsonpath='{.status.containerStatuses[?(@.name=="jupyter")].imageID}')
riddle_image="${riddle_image#docker-pullable://}"
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  /shared/work/RIDDLE/.venv/bin/python /shared/work/RIDDLE/nrp.py \
  --name riddle-seed42 --methods riddle --scenarios signal_injection \
  --seeds 42 --config config/settings.yaml --workers 2 --io-workers 4 --image "$riddle_image" | \
  kubectl apply -n cua-asimsek -f -
```

**LaCathode:**

```bash
riddle_image=$(kubectl get pod riddle-jupyter -n cua-asimsek -o jsonpath='{.status.containerStatuses[?(@.name=="jupyter")].imageID}')
riddle_image="${riddle_image#docker-pullable://}"
kubectl exec -n cua-asimsek riddle-jupyter -c jupyter -- \
  /shared/work/RIDDLE/.venv/bin/python /shared/work/RIDDLE/nrp.py \
  --name lacathode-seed42 --methods lacathode --scenarios signal_injection \
  --seeds 42 --workers 1 --io-workers 4 --image "$riddle_image" | \
  kubectl apply -n cua-asimsek -f -
```

For one combined job instead, request `--methods lacathode riddle` with a different job name.<br>
Do not submit that alongside these two jobs for the same result identities.

Optional controls use the preparation commands in `README.md`.<br>
To submit one later, add `--data data/lhco_shifted --results results_shifted` or `--data data/lhco_deltaR --results results_deltaR` to the generator command and choose a new job name.<br>
Each control still requests only one A100 per job.

## Monitoring and resuming

```bash
kubectl get jobs,pods -n cua-asimsek
kubectl get pods -n cua-asimsek -l job-name=lacathode-seed42 -o wide
kubectl get pods -n cua-asimsek -l job-name=riddle-seed42 -o wide
kubectl logs -n cua-asimsek -f job/lacathode-seed42 -c campaign
```

```bash
kubectl logs -n cua-asimsek -f job/riddle-seed42 -c campaign
```

## Jupyter terminal: plots

```bash
cd /shared/work/RIDDLE
source .venv/bin/activate
python scripts/nrp_runtime.py
python plot.py --results results --output plots --methods lacathode riddle --verbose 1
```


## Troubleshooting

- Pending job: inspect `kubectl describe pod <name> -n cua-asimsek` for GPU availability, quota or storage events.
