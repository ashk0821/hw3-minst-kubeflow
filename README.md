# Homework #3 — Simple DL Workflow on Kubeflow (GKE)

**Author:** Aashir Khan
**Date:** May 2026
**Task:** MNIST digit classification — PyTorch CNN trained as a Kubeflow `PyTorchJob`, served as a Kubernetes `Deployment` behind a `LoadBalancer` Service.

---

## 1. Overview

A PyTorch CNN is trained on MNIST inside a Kubeflow-managed pod, weights are persisted to a `PersistentVolumeClaim`, and a separate Flask service mounts the same PVC read-only to serve predictions over an external HTTP URL.

```
Training Data --> Training Code --> Model --> Inference Code --> Web-serving
   (MNIST)         (train.py)      (PVC)       (infer.py)      (LoadBalancer)
                       |                            |
                  Dockerfile.train            Dockerfile.infer
                  pytorchjob.yaml             inference.yaml
                  (Job, Volume)               (Deployment, Volume, Service)
```

Project: `kubeflow-hw3`, region `us-central1`, zone `us-central1-a`. All work done in Cloud Shell.

---

## 2. Repository layout

```
hw3-mnist-kubeflow/
├── README.md
├── code/
│   ├── train.py
│   ├── infer.py
│   ├── Dockerfile.train
│   └── Dockerfile.infer
├── manifests/
│   ├── pvc.yaml
│   ├── pytorchjob.yaml          (GPU, attempted)
│   ├── pytorchjob-cpu.yaml      (CPU, executed)
│   ├── inference.yaml           (Deployment + Service)
│   └── debug-pod.yaml
└── screenshots/
```

---

## 3. Prerequisites

| Item                  | Setting                                             |
| --------------------- | --------------------------------------------------- |
| GCP project           | `kubeflow-hw3` (billing enabled)                    |
| APIs                  | GKE, Artifact Registry, Cloud Build, Compute Engine |
| Default zone          | `us-central1-a`                                     |
| Tools                 | `gcloud`, `kubectl`, `docker` (Cloud Shell)         |
| Org policy constraint | `compute.vmExternalIpAccess` (private cluster used) |

```bash
gcloud config set project kubeflow-hw3
gcloud config set compute/zone us-central1-a
gcloud services enable container.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

---

## 4. Phase 1 — Cluster setup

### 4.1 Private GKE cluster

The org policy `compute.vmExternalIpAccess` blocked default node provisioning. Fix: create the cluster as **private** so nodes have no external IPs:

```bash
gcloud container clusters create kubeflow-cluster \
  --zone=us-central1-a --num-nodes=2 --machine-type=e2-standard-4 \
  --enable-ip-alias --enable-private-nodes \
  --master-ipv4-cidr=172.16.0.0/28 \
  --enable-master-authorized-networks --master-authorized-networks=0.0.0.0/0
```

GPU node pool, autoscaling `min=0, max=1`:

```bash
gcloud container node-pools create gpu-pool \
  --cluster=kubeflow-cluster --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=1 --enable-autoscaling --min-nodes=0 --max-nodes=1 \
  --enable-private-nodes
```

### 4.2 NVIDIA driver DaemonSet

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### 4.3 Kubeflow Training Operator

Pinned to v1.8.1 (the unpinned `master` reference in the slide command is broken in current kustomize):

```bash
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```

Operator pod runs in 30s; six Kubeflow CRDs registered (the relevant one being `pytorchjobs.kubeflow.org`).

### 4.4 Namespace and PVC

```bash
kubectl create namespace mnist
kubectl apply -f manifests/pvc.yaml
```

5 Gi `ReadWriteOnce` PVC on `standard-rwo`. Initially `Pending` because that storage class uses **`WaitForFirstConsumer`** binding — the disk is provisioned only when first mounted.

---

## 5. Phase 2 — Training

### 5.1 `train.py`

Two-conv + two-FC PyTorch CNN, trained 3 epochs with Adam (lr=1e-3). Auto-detects CUDA, falls back to CPU. Saves weights to `/mnt/model/mnist_cnn.pt` on the PVC.

### 5.2 `Dockerfile.train`

Base: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`. Because the cluster is private, MNIST is **pre-downloaded at build time** so the pod needs no internet access:

```dockerfile
RUN python -c "from torchvision import datasets; datasets.MNIST('/tmp/data', train=True, download=True)"
```

### 5.3 Build via Cloud Build

Direct `docker push` from Cloud Shell repeatedly hit `connection refused`. Switched to Cloud Build, which runs entirely on Google infrastructure:

```bash
gcloud builds submit --tag $IMG_TRAIN --timeout=30m .
```

Result: `digest: sha256:b79306378c20...23a2d99e3`, `STATUS: SUCCESS` after 10m27s.

### 5.4 GPU attempt

Applied `pytorchjob.yaml` with the GPU resource limit, T4 node selector, and the `nvidia.com/gpu=present:NoSchedule` toleration. Pod stayed `Pending`. `kubectl describe` showed:

```
Warning  FailedScaleUp  cluster-autoscaler  Node scale up in zones us-central1-a
                                            associated with this pod failed:
                                            GCE out of resources.
```

T4 capacity in `us-central1-a` was exhausted — a real GCE zonal-capacity issue, not a config problem.

### 5.5 CPU fallback

Deleted the GPU job and applied `pytorchjob-cpu.yaml` (same spec, no GPU bits). `train.py` already handles device fallback, so no code changes:

```bash
kubectl delete pytorchjob -n mnist mnist-train
kubectl apply -f manifests/pytorchjob-cpu.yaml
```

Job ran 4m55s, status `Succeeded`. Loss collapsed from 2.31 to 0.02 across 3 epochs:

```
Using device: cpu
Epoch 0 step 0   loss 2.3104
...
Epoch 2 step 800 loss 0.0177
Saved model to /mnt/model/mnist_cnn.pt
```

### 5.6 PVC handoff verified

PVC flipped to `Bound`, and a short-lived debug pod (`debug-pod.yaml`) confirmed the file landed:

```
-rw-r--r-- 1 root root 4.6M May 9 03:16 mnist_cnn.pt
```

---

## 6. Phase 3 — Inference

### 6.1 `infer.py`

Flask app, three routes:

| Method | Path       | Purpose                          |
| ------ | ---------- | -------------------------------- |
| GET    | `/`        | HTML upload form                 |
| GET    | `/healthz` | Probe endpoint, returns `ok`     |
| POST   | `/predict` | Multipart image, returns JSON    |

`Net` class duplicated from `train.py` because `state_dict` is weights-only. Model loaded once at startup. Same preprocessing pipeline as training (Grayscale → 28×28 → tensor → MNIST normalize); `argmax` for prediction, `softmax` for confidence.

### 6.2 `Dockerfile.infer`

Base: `python:3.10-slim` with **CPU-only PyTorch** from the official CPU wheel index. Image size dropped from ~3.8 GB (training) to ~289 MB (inference) — much faster pod startup and rolling updates.

```dockerfile
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "numpy<2" flask pillow
```

`numpy<2` is pinned to fix a v1 runtime bug (Section 10.3).

### 6.3 Built and pushed via Cloud Build:

```
infer  digest: sha256:f855e74a5e945c84fd52dde5b1cb8b646a5994c7390ef641396132c195daba3b   289 MB
train  digest: sha256:b79306378c208ca9d9718fc91cb977d2914a5f522714aa3c56c548023a2d99e3  3804 MB
```

### 6.4 Deployment + Service

`inference.yaml` defines two resources:

- **Deployment `mnist-infer`** — `replicas: 1`, mounts `model-pvc` at `/mnt/model` **read-only**, `readinessProbe` and `livenessProbe` against `/healthz`.
- **Service `mnist-infer-svc`** — `type: LoadBalancer`, port 80 → 8080. The org policy targets VM external IPs, not Service LoadBalancers, so this works on a private cluster.

```bash
kubectl apply -f manifests/inference.yaml
```

After ~60s:

```
NAME              TYPE           CLUSTER-IP       EXTERNAL-IP    PORT(S)
mnist-infer-svc   LoadBalancer   34.118.239.125   136.119.9.55   80:31312/TCP
```

### 6.5 Health check

```
$ curl -v http://136.119.9.55/healthz
< HTTP/1.1 200 OK
< Server: Werkzeug/3.1.8 Python/3.10.20
ok
```

### 6.6 Predictions

5 MNIST test digits submitted via `curl`:

```
=== digit_0_label_7.png === {"prediction":7,"confidence":0.9998}
=== digit_1_label_2.png === {"prediction":2,"confidence":0.9985}
=== digit_2_label_1.png === {"prediction":1,"confidence":0.9994}
=== digit_3_label_0.png === {"prediction":0,"confidence":0.9999}
=== digit_4_label_4.png === {"prediction":4,"confidence":0.9991}
```

5/5 correct — the full pipeline (training pod → PVC → inference pod → LoadBalancer) works end-to-end.

---

## 7. Reflections — Kubernetes controllers used

**`PyTorchJob`** (Kubeflow Training Operator) for training. Job-like lifecycle that's also distributed-PyTorch-aware: with multiple replicas it would auto-inject `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`. Reports aggregated `Succeeded/Failed` status at the job level.

**`Deployment`** for inference. Inference is a long-running service, not a finite job — Deployment provides replica management, self-healing, and rolling updates. I demonstrated rolling updates while fixing the v1 numpy bug: `kubectl set image` triggered zero-downtime cutover from v1 to v2.

**`Service`** of type `LoadBalancer` for external access. Provisions a Google-managed Network Load Balancer with a stable external IP; the LB forwards to whichever pods match the label selector, so clients see one URL even as pods come and go.

**`PersistentVolumeClaim`** for model handoff. Training writes `mnist_cnn.pt`, inference reads it. `WaitForFirstConsumer` binding avoided allocating disk before it was needed. `ReadWriteOnce` is fine for one inference replica but would constrain horizontal scaling.

**`DaemonSet`** for the NVIDIA driver installer — exactly one pod per matching node, perfect for kernel-module installation. Was correctly waiting for a GPU node that never materialized due to the capacity issue.

---

## 8. Cleanup

```bash
gcloud container clusters delete kubeflow-cluster --zone us-central1-a --quiet
gcloud artifacts repositories delete mnist-images --location us-central1 --quiet
```

---

## 9. Screenshots

In `screenshots/`:

- `nodes-ready.png` — `kubectl get nodes` after cluster creation
- `training-operator-running.png` — operator install, pod 1/1 Running, CRDs
- `artifact-registry-repo.png` — `mnist-images` repo created
- `cloudbuild-train-success.png` — train image pushed, digest, `STATUS: SUCCESS`
- `pytorchjob-succeeded.png` — `kubectl get pytorchjob` showing `Succeeded`
- `training-logs.png` — full training log, loss 2.31 → 0.02
- `model-on-pvc.png` — debug pod confirming `mnist_cnn.pt` (4.6M) on PVC
- `pvc-bound.png` — PVC status `Bound` after training mounted it
- `cloudbuild-infer-success.png` — infer image pushed, digest, `STATUS: SUCCESS`
- `artifact-registry.png` — both images present in registry
- `healthz-200.png` — `curl -v /healthz` returning HTTP 200 `ok`
- `browser-form.png` — browser at `http://136.119.9.55/` showing upload form

---

## 10. Challenges encountered

**10.1 Org policy blocking VM external IPs.** Default GKE creation failed with `compute.vmExternalIpAccess violated`. Fixed by creating the cluster as private (`--enable-private-nodes`); side effect was no internet egress from pods, worked around by pre-baking MNIST into the training image.

**10.2 GPU zonal capacity exhaustion.** Autoscaler reported `GCE out of resources` for T4s in `us-central1-a`. Pivoted to CPU PyTorchJob (5 minutes of training, no functional impact for MNIST). Both YAMLs are kept in the submission.

**10.3 NumPy missing from slim inference image.** `infer:v1` returned HTTP 500 with `RuntimeError: Numpy is not available` from inside `transforms.ToTensor()`. The slim Python base lacks numpy, and the CPU torchvision wheel doesn't pull it as a hard dep. Fixed with `pip install "numpy<2"` (numpy 2.x has ABI breaks with torch 2.1), rebuilt as `infer:v2`, and rolled out via `kubectl set image`.

**10.4 Cloud Shell ↔ Artifact Registry network flakiness.** Direct `docker push` repeatedly failed with `connection refused`. Switching to `gcloud builds submit` ran the build and push entirely on Google infrastructure and worked first try.
