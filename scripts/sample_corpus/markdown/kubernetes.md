# Kubernetes Reference Guide

## Overview
Kubernetes (K8s) is a container orchestration platform that automates deployment, scaling, networking, and management of containerized applications across a cluster of machines. It provides self-healing, declarative configuration, and horizontal scaling for production workloads.

## Core Objects

### Pod
The smallest deployable unit — one or more containers that share networking and storage, scheduled together on the same node.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: rag-api-pod
spec:
  containers:
    - name: rag-api
      image: myrepo/rag-api:latest
      ports:
        - containerPort: 8000
```

### Deployment
Manages a set of replica Pods, handles rolling updates, and self-heals by recreating failed Pods.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
        - name: rag-api
          image: myrepo/rag-api:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
```

### Service
Provides a stable network identity for a set of Pods (see Service Types below).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### ConfigMap & Secret
Externalize configuration and sensitive values from container images.

```bash
kubectl create configmap app-config --from-literal=LOG_LEVEL=info
kubectl create secret generic app-secret --from-literal=API_KEY=xxxx
```

### StatefulSet
Like a Deployment, but for stateful workloads needing stable network identities and persistent storage per replica (e.g. clustered vector databases).

### Namespace
Logical partition of a cluster for isolating environments, teams, or applications.

## Service Types

| Type | Behavior | Typical Use |
|---|---|---|
| `ClusterIP` | Internal-only virtual IP | Internal services (DB, retriever, inference) |
| `NodePort` | Exposes on a static port on every node | Quick dev/testing access |
| `LoadBalancer` | Provisions external cloud load balancer | Public-facing production API |
| `ExternalName` | Maps to an external DNS name | Referencing external managed services |
| Headless (`clusterIP: None`) | DNS returns Pod IPs directly | StatefulSets, per-replica addressing |

## Horizontal Pod Autoscaler (HPA)
Automatically scales replica count based on observed metrics like CPU utilization.

```bash
kubectl autoscale deployment rag-api --cpu-percent=70 --min=2 --max=10
```

Requires `metrics-server` installed in the cluster to function.

## Rolling Updates & Rollbacks

```bash
kubectl set image deployment/rag-api rag-api=myrepo/rag-api:v2
kubectl rollout status deployment/rag-api
kubectl rollout history deployment/rag-api
kubectl rollout undo deployment/rag-api
```

## Common kubectl Commands

```bash
kubectl get pods -A                     # List all pods, all namespaces
kubectl describe pod <name>             # Detailed pod info and events
kubectl logs -f <pod>                   # Stream logs
kubectl exec -it <pod> -- bash          # Shell into a pod
kubectl apply -f manifest.yaml          # Create/update resources from file
kubectl delete -f manifest.yaml         # Delete resources from file
kubectl get events --sort-by='.lastTimestamp'  # Recent cluster events
kubectl top pods                        # Resource usage per pod (needs metrics-server)
```

## Local Development with k3d
`k3d` runs lightweight k3s clusters inside Docker, ideal for local testing before deploying to a managed cloud cluster (EKS/GKE/AKS).

```bash
k3d cluster create mycluster --agents 2
k3d image import myrepo/rag-api:latest -c mycluster
kubectl get nodes
```

## Health Checks
- **Liveness probe**: restarts a container if it becomes unresponsive.
- **Readiness probe**: removes a Pod from Service load-balancing until it's ready to accept traffic.

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Common Troubleshooting

- **Pod stuck in `Pending`**: insufficient cluster resources or unschedulable node constraints — check `kubectl describe pod` events.
- **`CrashLoopBackOff`**: application is crashing on startup — check `kubectl logs <pod> --previous`.
- **`ImagePullBackOff`**: incorrect image name/tag, or (for local clusters) image not imported into the cluster.
- **Service not reachable**: check `kubectl get endpoints <service>` — if empty, the Service's label selector doesn't match any Pod labels.
- **HPA not scaling**: confirm `metrics-server` is installed and reporting resource usage.

## Monitoring Integration
Kubernetes workloads are commonly monitored with Prometheus (scraping metrics endpoints exposed by ClusterIP Services) and visualized in Grafana — including custom application metrics like retrieval latency or model-serving throughput in ML-serving deployments.
