---
layout: docs
title: "Kubernetes: Workloads & Storage"
permalink: /docs/technology/kubernetes/workloads.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #326ce5 0%, #54a3ff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Kubernetes: Workloads & Storage</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Configure persistent storage, manage application resources, implement security best practices, and monitor your Kubernetes workloads.</p>
</div>

## Storage: Persistent Data in an Ephemeral World

Containers are ephemeral - when a pod dies, its local filesystem is lost. This creates a challenge: how do you run databases, store uploaded files, or maintain application state?

Kubernetes solves this with a storage abstraction layer that separates "what storage I need" from "how that storage is provided." This separation lets you write portable applications that work across cloud providers and on-premises infrastructure.

**Consider the following storage concepts**:

| Concept | What It Does | Analogy |
|---------|--------------|---------|
| **PersistentVolume (PV)** | A piece of storage in the cluster | A hard drive |
| **PersistentVolumeClaim (PVC)** | A request for storage | "I need 10GB of fast storage" |
| **StorageClass** | Defines types of storage available | "Premium SSD" vs "Standard HDD" |
| **CSI Driver** | Connects Kubernetes to storage backends | A device driver for storage |

The typical workflow: you create a PVC requesting storage, Kubernetes finds or creates a matching PV, and your pod mounts the volume.

### Container Storage Interface (CSI)

CSI is a standard interface that allows any storage system to work with Kubernetes. You typically do not implement CSI drivers yourself - you use drivers provided by storage vendors (AWS EBS, Google PD, Azure Disk, etc.).

**How CSI works** (simplified):
1. You create a PVC requesting storage
2. The CSI provisioner creates a volume in the storage backend
3. When a pod needs the volume, the CSI node plugin mounts it
4. When the pod is deleted, the volume is unmounted (but data persists)

**Common CSI drivers**:
| Cloud Provider | CSI Driver |
|----------------|------------|
| AWS | aws-ebs-csi-driver |
| Google Cloud | gce-pd-csi-driver |
| Azure | azuredisk-csi-driver |
| On-premises | Rook-Ceph, Longhorn, OpenEBS |

### Volume Lifecycle

Understanding how volumes move through states helps with troubleshooting:

| State | Meaning | What to Check |
|-------|---------|---------------|
| **Pending** | PVC waiting for a matching PV | Is there a matching StorageClass? |
| **Bound** | PVC connected to a PV | Normal state when in use |
| **Released** | PVC deleted but PV retains data | Check reclaim policy |
| **Failed** | Provisioning error | Check CSI driver logs |

**Common issues**:
- PVC stuck in Pending: Usually means no StorageClass or insufficient capacity
- PVC stuck in Terminating: Often a finalizer issue; check for pods still using it

### Storage Patterns for Production

Different applications need different storage strategies. Here are common patterns:

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Single-writer** | One pod writes, many read | ReadWriteOnce access mode |
| **Shared storage** | Multiple pods read/write | ReadWriteMany (requires NFS or similar) |
| **Per-pod volumes** | Each replica has own storage | StatefulSet with volumeClaimTemplates |
| **Ephemeral storage** | Scratch space, caches | emptyDir volume |

**Choosing access modes**:
- **ReadWriteOnce (RWO)**: Single node can mount read-write. Most common for databases.
- **ReadOnlyMany (ROX)**: Many nodes can mount read-only. Good for shared config.
- **ReadWriteMany (RWX)**: Many nodes can mount read-write. Requires special storage (NFS, CephFS).


### Storage Classes: Defining Your Storage Tiers

Storage Classes let you define different tiers of storage (fast SSD, cheap HDD, replicated, etc.) so developers can request storage by characteristics without knowing infrastructure details.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

**Key StorageClass settings**:

| Setting | Purpose | Common Values |
|---------|---------|---------------|
| `reclaimPolicy` | What happens when PVC deleted | Delete (default), Retain |
| `allowVolumeExpansion` | Can volumes grow? | true, false |
| `volumeBindingMode` | When to provision | Immediate, WaitForFirstConsumer |

**Tip**: Use `WaitForFirstConsumer` to ensure volumes are created in the same zone as the pod that will use them.

### Volume Snapshots: Point-in-Time Backups

Snapshots create point-in-time copies of volumes for backup and recovery. They require a CSI driver that supports snapshots.

**Creating a snapshot**:
```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: db-snapshot
spec:
  source:
    persistentVolumeClaimName: database-pvc
```

**Restoring from a snapshot**:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-db
spec:
  dataSource:
    name: db-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

**When to use snapshots**: Before database migrations, major updates, or as part of a backup strategy. Note that snapshots are not backups - they often depend on the original storage and may not survive storage system failures.

## Configuration Management: Fine-Tuning Your Applications

Getting resource allocation right is crucial: too little and your app crashes or gets throttled; too much and you waste money and prevent other workloads from scheduling.

### Resource Requests and Limits

Every container should specify resource requests and limits:

```yaml
resources:
  requests:      # Guaranteed minimum
    memory: "256Mi"
    cpu: "500m"
  limits:        # Maximum allowed
    memory: "512Mi"
    cpu: "1000m"
```

**Understanding the difference**:

| Type | Purpose | What Happens if Exceeded |
|------|---------|--------------------------|
| **Request** | Scheduling: "I need at least this much" | N/A - it is a minimum |
| **Limit** | Protection: "Never use more than this" | CPU: throttled; Memory: OOMKilled |

**Best practices**:
- Always set memory limits (OOM is harder to debug than CPU throttling)
- Set requests based on actual usage (check with `kubectl top pods`)
- Limits should be 1.5-2x requests for most workloads
- Use Vertical Pod Autoscaler to get recommendations

### Autoscaling: HPA vs VPA

Kubernetes offers two autoscaling approaches:

| Autoscaler | What It Scales | Best For |
|------------|----------------|----------|
| **HPA** (Horizontal) | Number of pods | Stateless apps that scale horizontally |
| **VPA** (Vertical) | CPU/memory per pod | Stateful apps, right-sizing resources |

**HPA Example** - scale pods based on CPU:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**VPA Example** - automatically adjust resource requests:
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app-deployment
  updatePolicy:
    updateMode: "Auto"
```

**Important**: Do not use HPA and VPA together on the same metric (e.g., both scaling on CPU). They will fight each other.

## Security: Protecting Your Cluster and Applications

Kubernetes security operates at multiple layers. Understanding each layer helps you build defense in depth.

| Layer | Protection | Kubernetes Feature |
|-------|------------|-------------------|
| **Cluster access** | Who can talk to the API | RBAC, authentication |
| **Pod-to-pod** | Which pods can communicate | Network Policies |
| **Container** | What containers can do | Security Context, PSS |
| **Secrets** | Protecting sensitive data | Secrets, external vaults |

### Role-Based Access Control (RBAC)

RBAC controls who can do what in your cluster. It uses Roles (what actions are allowed) and RoleBindings (who gets those permissions).

```yaml
# Role: defines what actions are allowed
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
---
# RoleBinding: grants the role to a user
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
subjects:
- kind: User
  name: jane
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

**Role vs ClusterRole**: Roles are namespace-scoped; ClusterRoles are cluster-wide. Use ClusterRoles for cluster-level resources (nodes, namespaces) or when you need the same permissions across all namespaces.

### Pod Security Standards

Pod Security Standards (PSS) replace the deprecated PodSecurityPolicy. They define three levels:

| Level | Description | Use Case |
|-------|-------------|----------|
| **Privileged** | No restrictions | System components, trusted workloads |
| **Baseline** | Prevents known privilege escalations | Most workloads |
| **Restricted** | Heavily restricted, best practices | Security-sensitive workloads |

Apply PSS at the namespace level:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
```

**Essential security context settings**:
```yaml
securityContext:
  runAsNonRoot: true
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```

### Service Accounts

Service accounts provide identity for pods. Each pod automatically gets a service account, but you should create dedicated accounts with minimal permissions for each application.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
---
# Use in deployment
spec:
  template:
    spec:
      serviceAccountName: app-service-account
```

**Security tip**: Disable automatic token mounting for pods that do not need to talk to the Kubernetes API:
```yaml
automountServiceAccountToken: false
```

## Observability: Understanding Application Health

Observability in Kubernetes has three pillars: health probes, metrics, and logs.

### Health Probes

Probes tell Kubernetes whether your application is healthy:

| Probe Type | Purpose | What Happens on Failure |
|------------|---------|-------------------------|
| **Liveness** | Is the app alive? | Container restarted |
| **Readiness** | Can the app accept traffic? | Removed from service endpoints |
| **Startup** | Has the app started? | Liveness/readiness delayed |

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

**Common mistake**: Making liveness probes too aggressive. If your app occasionally takes 10 seconds to respond, a 5-second timeout will cause unnecessary restarts.

### Monitoring with Prometheus

Prometheus is the standard for Kubernetes metrics. Applications expose metrics at `/metrics`, and Prometheus scrapes them.

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: app-metrics
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: metrics
    interval: 30s
```

**What to monitor**: Request rate, error rate, latency (the "RED" method), plus resource usage (CPU, memory, network).

