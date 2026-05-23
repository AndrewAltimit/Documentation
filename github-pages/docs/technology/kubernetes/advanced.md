---
layout: docs
title: "Kubernetes: Advanced Topics"
permalink: /docs/technology/kubernetes/advanced.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #326ce5 0%, #54a3ff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Kubernetes: Advanced Topics</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Deep dives into CRDs, Operators, Service Mesh, GitOps, and production-grade deployment patterns for enterprise Kubernetes.</p>
</div>

## Advanced Topics: Taking Kubernetes to Production

### Custom Resource Definitions (CRDs) and Operators

Kubernetes is extensible by design. Custom Resource Definitions allow you to create new resource types, while Operators encode operational knowledge into software. Together, they enable Kubernetes to manage complex stateful applications as naturally as it manages stateless ones.

#### Creating a Custom Resource

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              engine:
                type: string
                enum: ["postgres", "mysql", "mongodb"]
              version:
                type: string
              replicas:
                type: integer
                minimum: 1
              backup:
                type: object
                properties:
                  enabled:
                    type: boolean
                  schedule:
                    type: string
  scope: Namespaced
  names:
    plural: databases
    singular: database
    kind: Database
```

#### Building an Operator

A CRD on its own only teaches the API server a new *noun* — it stores `Database` objects but does nothing with them. The verb comes from an **operator**: a controller that watches for those objects and drives the cluster toward the state they describe. This is the same reconcile loop that powers built-in controllers, applied to your custom type. The simplified `Reconcile` below shows the core idea — fetch the desired object, then create or update the real resources (a StatefulSet, a backup CronJob) to match it.

```go
// Simplified operator logic
func (r *DatabaseReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    // Fetch the Database instance
    database := &examplev1.Database{}
    err := r.Get(ctx, req.NamespacedName, database)
    if err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }
    
    // Ensure StatefulSet exists
    statefulSet := r.statefulSetForDatabase(database)
    err = r.Create(ctx, statefulSet)
    if err != nil && !errors.IsAlreadyExists(err) {
        return ctrl.Result{}, err
    }
    
    // Ensure backup CronJob exists if enabled
    if database.Spec.Backup.Enabled {
        cronJob := r.backupCronJobForDatabase(database)
        err = r.Create(ctx, cronJob)
        if err != nil && !errors.IsAlreadyExists(err) {
            return ctrl.Result{}, err
        }
    }
    
    return ctrl.Result{}, nil
}
```

### Service Mesh: Advanced Networking with Istio

As microservices architectures grow, managing service-to-service communication becomes complex. Service meshes like Istio provide a dedicated infrastructure layer for handling service communications, offering features like traffic management, security, and observability without changing application code.

#### Traffic Management

A service mesh injects a sidecar proxy next to every pod and routes all traffic through it. Because the proxy sees every request, you can shift, split, and inspect traffic with configuration rather than code. The `VirtualService` below does a **canary release**: requests from the user `jason` go to `v2`, while everyone else is split 75/25 between `v1` and `v2`. Shifting the weights gradually rolls the canary out (or rolls it back) with no redeploy.

```yaml
# Virtual Service for canary deployment
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
  - reviews
  http:
  - match:
    - headers:
        end-user:
          exact: jason
    route:
    - destination:
        host: reviews
        subset: v2
  - route:
    - destination:
        host: reviews
        subset: v1
      weight: 75
    - destination:
        host: reviews
        subset: v2
      weight: 25
```

#### Resilience: Circuit Breaking and Outlier Detection

The other half of mesh traffic policy is keeping a struggling service from taking down its callers. A `DestinationRule` caps how many connections and pending requests pile up (the connection pool), and **outlier detection** ejects an instance from the load-balancing pool after it returns too many consecutive errors — the mesh's version of a circuit breaker. The settings below eject a backend for 30 seconds after 5 consecutive errors, while never ejecting more than half the pool.

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      minHealthPercent: 50
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

### GitOps: Declarative Continuous Deployment

GitOps uses Git as the single source of truth for declarative infrastructure and applications. Tools like ArgoCD or Flux monitor Git repositories and automatically sync the cluster state with the desired state defined in Git.

#### ArgoCD Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/myapp
    targetRevision: HEAD
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

### Multi-Tenancy: Sharing Clusters Safely

Running multiple teams or applications on the same cluster requires careful isolation. Kubernetes provides several mechanisms for multi-tenancy, from simple namespace isolation to virtual clusters.

#### Hierarchical Namespaces

```yaml
apiVersion: hnc.x-k8s.io/v1alpha2
kind: HierarchicalConfiguration
metadata:
  name: hierarchy
  namespace: team-platform
spec:
  parent: organization-root
---
apiVersion: hnc.x-k8s.io/v1alpha2
kind: SubnamespaceAnchor
metadata:
  name: team-frontend
  namespace: team-platform
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-quota
  namespace: team-platform
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    persistentvolumeclaims: "10"
```

### Advanced Scheduling: Placing Pods Intelligently

Kubernetes' scheduler is highly configurable, allowing you to influence pod placement based on various criteria. This becomes crucial for performance optimization, compliance requirements, and cost management.

#### Custom Scheduler Configuration

```yaml
apiVersion: kubescheduler.config.k8s.io/v1beta2
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: custom-scheduler
  plugins:
    score:
      enabled:
      - name: NodeResourcesFit
        weight: 1
      - name: NodeAffinity
        weight: 2
    filter:
      enabled:
      - name: NodeResourcesFit
      - name: NodeAffinity
      - name: NodePorts
      - name: NodeVolumeLimits
  pluginConfig:
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: MostAllocated
        resources:
        - name: cpu
          weight: 1
        - name: memory
          weight: 1
```

#### Pod Topology Spread Constraints

Most teams never write a custom scheduler — instead they nudge the default one with constraints. **Topology spread constraints** are the most useful: they tell the scheduler to spread replicas evenly across a failure domain (nodes, then zones) so a single node or zone outage cannot take down a disproportionate share of your pods. The `maxSkew: 1` below means no domain may hold more than one extra pod versus the least-loaded one, and `DoNotSchedule` makes that a hard requirement.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 6
  template:
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: web-app
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: web-app
```

### Cluster API: Kubernetes for Managing Kubernetes

Cluster API brings declarative, Kubernetes-style APIs to cluster creation, configuration, and management. It's like using Kubernetes to manage Kubernetes clusters themselves.

```yaml
apiVersion: cluster.x-k8s.io/v1beta1
kind: Cluster
metadata:
  name: my-cluster
spec:
  clusterNetwork:
    pods:
      cidrBlocks:
      - 10.96.0.0/12
    services:
      cidrBlocks:
      - 10.244.0.0/16
  controlPlaneRef:
    apiVersion: controlplane.cluster.x-k8s.io/v1beta1
    kind: KubeadmControlPlane
    name: my-cluster-control-plane
  infrastructureRef:
    apiVersion: infrastructure.cluster.x-k8s.io/v1beta1
    kind: AWSCluster
    name: my-cluster
```

## Production Best Practices: Real-World Lessons

### High Availability Patterns

1. **Control Plane HA**:
   - Run 3 or 5 master nodes (odd numbers for quorum)
   - Use external etcd cluster for large deployments
   - Place masters in different availability zones

2. **Application HA**:
   - Use PodDisruptionBudgets to prevent accidental outages
   - Implement proper health checks
   - Use anti-affinity rules to spread pods across nodes

### Cost Optimization Strategies

1. **Resource Right-Sizing**:
   ```yaml
   # Use VPA recommendations
   apiVersion: autoscaling.k8s.io/v1
   kind: VerticalPodAutoscaler
   metadata:
     name: app-vpa
   spec:
     targetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: app
     updatePolicy:
       updateMode: "Off"  # Just recommendations
   ```

2. **Spot Instance Integration**:
   ```yaml
   # Node selector for spot instances
   nodeSelector:
     node.kubernetes.io/lifecycle: spot
   tolerations:
   - key: spot
     operator: Equal
     value: "true"
     effect: NoSchedule
   ```

### Security Hardening Checklist

- [ ] Enable RBAC and remove default service account permissions
- [ ] Implement Pod Security Standards/Policies
- [ ] Use network policies for micro-segmentation
- [ ] Enable audit logging
- [ ] Scan images for vulnerabilities
- [ ] Rotate certificates regularly
- [ ] Implement admission webhooks for policy enforcement
- [ ] Use service mesh for mTLS between services
- [ ] Implement secrets management (Sealed Secrets, Vault)
- [ ] Regular security updates and patches

### Disaster Recovery Planning

1. **Backup Strategy**:
   ```bash
   # Velero backup example
   velero backup create production-backup \
     --include-namespaces production \
     --snapshot-volumes \
     --ttl 720h
   ```

2. **Multi-Region Failover**: The older KubeFed (Federation v2) project is effectively retired; the modern pattern is **independent clusters per region**, each reconciled from the same Git repo, with a global load balancer steering traffic. Fleet/multi-cluster managers such as Karmada, Open Cluster Management, or a hub-and-spoke Argo CD coordinate placement and failover across them.

   ```yaml
   # Argo CD ApplicationSet fans the same app out to every regional cluster
   apiVersion: argoproj.io/v1alpha1
   kind: ApplicationSet
   metadata:
     name: app-all-regions
     namespace: argocd
   spec:
     generators:
     - clusters: {}          # one Application per registered cluster
     template:
       metadata:
         name: 'app-{% raw %}{{name}}{% endraw %}'
       spec:
         project: default
         source:
           repoURL: https://github.com/myorg/app
           targetRevision: HEAD
           path: k8s
         destination:
           server: '{% raw %}{{server}}{% endraw %}'
           namespace: production
   ```

## Common Pitfalls and How to Avoid Them

### 1. Resource Limits Not Set
**Problem**: Pods consume all available resources
**Solution**: Always set resource requests and limits
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### 2. Using Latest Tag
**Problem**: Unpredictable deployments
**Solution**: Use specific image tags
```yaml
# Bad
image: myapp:latest

# Good
image: myapp:v1.2.3
```

### 3. Not Using Health Checks
**Problem**: Unhealthy pods receive traffic
**Solution**: Implement proper probes
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

### 4. Ignoring Pod Disruption Budgets
**Problem**: All pods deleted during updates
**Solution**: Define PDBs
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: app-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: myapp
```

### 5. Not Planning for Node Failures
**Problem**: Single point of failure
**Solution**: Use node anti-affinity
```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values:
          - myapp
      topologyKey: kubernetes.io/hostname
```

## Kubernetes Ecosystem: Essential Tools

### Development Tools
- **Skaffold**: Continuous development for Kubernetes
- **Tilt**: Multi-service dev environment
- **Telepresence**: Local development with remote cluster
- **k9s**: Terminal UI for Kubernetes

### Monitoring and Observability
- **Prometheus + Grafana**: Metrics and visualization
- **Elastic Stack**: Log aggregation
- **Jaeger**: Distributed tracing
- **Kube-state-metrics**: Cluster state metrics

### Security Tools
- **Falco**: Runtime security
- **OPA (Open Policy Agent)**: Policy enforcement
- **Kubesec**: Security risk analysis
- **kube-bench**: CIS benchmark checks

### CI/CD Integration
- **Tekton**: Cloud-native CI/CD
- **Jenkins X**: Kubernetes-native CI/CD
- **Spinnaker**: Multi-cloud continuous delivery

## Performance Tuning: Making Kubernetes Fly

### etcd Optimization
```bash
# Defragmentation
ETCDCTL_API=3 etcdctl defrag --endpoints=https://127.0.0.1:2379

# Compaction
rev=$(ETCDCTL_API=3 etcdctl endpoint status --write-out="json" | jq -r '.[0].Status.header.revision')
ETCDCTL_API=3 etcdctl compact $rev
```

### API Server Tuning
```yaml
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
apiServer:
  extraArgs:
    max-requests-inflight: "1000"
    max-mutating-requests-inflight: "500"
    default-watch-cache-size: "500"
    event-ttl: "1h"
```

### Network Performance
```yaml
# Enable host networking for performance-critical pods
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet
```

## Modern Capabilities Worth Knowing

Kubernetes ships a new minor release roughly every four months, so "latest features" age quickly. The capabilities below have stabilized in recent releases and are worth reaching for once you are past the basics — they change how you handle sidecars, resizing, security, and ingress.

| Capability | What it changes | Why it matters |
|------------|-----------------|----------------|
| **Native sidecar containers** | Sidecars declared as restartable init containers | Sidecars start before and stop after the main container, fixing log/proxy lifecycle bugs |
| **In-place pod vertical scaling** | Resize CPU/memory requests without restarting the pod | Right-size live workloads with no disruption |
| **CEL admission policies** | Validate/mutate objects with Common Expression Language | Policy enforcement without a webhook server to operate |
| **cgroups v2** | Unified resource hierarchy on the node | More accurate memory accounting and pressure handling |
| **Gateway API** | Role-oriented successor to Ingress | Richer, portable L4/L7 routing (see below) |

### Hardening Namespaces with Pod Security Standards

Pod Security Standards are applied per namespace through labels, in three independent modes: `enforce` rejects non-compliant pods, `warn` lets them through with a warning, and `audit` records a violation in the audit log. A common rollout is to start with `warn`/`audit` to find offenders, then flip `enforce` once the namespace is clean.

```yaml
# Enforce the Restricted profile; warn and audit on the same standard
apiVersion: v1
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### Scheduling GPU Workloads

GPUs are exposed as a schedulable [extended resource](https://kubernetes.io/docs/concepts/scheduling-eviction/) named `nvidia.com/gpu` (installed by the GPU device plugin). You request whole GPUs the same way you request CPU or memory — fractional GPUs are not supported by the standard plugin, so a request and limit of `2` reserves two entire devices for the pod.

```yaml
# Request whole GPUs via the device-plugin extended resource
apiVersion: v1
kind: Pod
metadata:
  name: ml-training
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    resources:
      limits:
        nvidia.com/gpu: 2
      requests:
        nvidia.com/gpu: 2
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "all"
```

### Gateway API: The Successor to Ingress

The Gateway API is the long-term replacement for Ingress. It splits responsibilities by role: a cluster operator owns the `Gateway` (the listener, ports, and TLS), while application teams attach `HTTPRoute` objects to it for their own routing — without touching shared infrastructure. It also handles protocols Ingress never did cleanly (gRPC, TCP, UDP). The `Gateway` below just declares an HTTPS listener; routes are defined separately and bound to it.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: prod-gateway
spec:
  gatewayClassName: nginx
  listeners:
  - name: https
    protocol: HTTPS
    port: 443
    tls:
      certificateRefs:
      - name: prod-cert
```

## Future of Kubernetes: What's Next?

Several ecosystems are maturing on top of Kubernetes rather than changing the core. These are the directions worth watching:

1. **Serverless on Kubernetes**: Knative, OpenFaaS, and KEDA for scale-to-zero and event-driven workloads
2. **Edge Computing**: K3s, KubeEdge, and Akri for resource-constrained and leaf devices
3. **WebAssembly**: Running Wasm workloads alongside containers via runwasi
4. **AI/ML Workloads**: Kubeflow, MLOps pipelines, and Ray on Kubernetes
5. **eBPF Integration**: Cilium, Pixie for advanced observability
6. **Platform Engineering**: Backstage, Crossplane for developer portals
7. **FinOps**: Cost optimization with Kubecost, OpenCost

### Kubernetes Alternatives and When to Use Them
- **Docker Swarm**: Simple container orchestration
- **Nomad**: Simpler alternative for mixed workloads
- **Amazon ECS**: AWS-native container orchestration
- **Cloud Run**: Serverless containers

Choose Kubernetes when you need:
- Multi-cloud portability
- Complex application architectures
- Fine-grained control
- Large ecosystem of tools

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Extend, Don't Fork</h4>
    <p>CRDs and Operators let you teach Kubernetes about your own resource types, so custom and built-in objects are managed the same declarative way.</p>
  </div>
  <div class="takeaway-card">
    <h4>Service Mesh for Cross-Cutting Concerns</h4>
    <p>mTLS, traffic shifting, and observability belong in the mesh (Istio, Linkerd) — not in every application's code.</p>
  </div>
  <div class="takeaway-card">
    <h4>GitOps Is the Source of Truth</h4>
    <p>Declare desired state in Git; let Argo CD or Flux reconcile the cluster. Rollbacks become a <code>git revert</code>.</p>
  </div>
  <div class="takeaway-card">
    <h4>Match the Tool to the Need</h4>
    <p>Kubernetes earns its complexity for multi-cloud, large-scale, or intricate architectures. For simpler workloads, ECS, Nomad, or Cloud Run may serve better.</p>
  </div>
</div>

The Kubernetes ecosystem evolves rapidly — serverless (Knative, KEDA), edge (K3s, KubeEdge), WebAssembly, and eBPF-based observability are all maturing fast. The durable principles remain: start simple, automate aggressively, observe everything, treat security as a requirement, and practice disaster recovery before you need it.

## See Also
- [Docker](../docker/) - Container fundamentals and image creation
- [AWS](../aws/) - Managed Kubernetes services (EKS)
- [Terraform](../terraform/) - Infrastructure as Code for Kubernetes
- [Networking](../networking.html) - Network concepts and protocols
- [Database Design](../database-design.html) - Stateful applications in Kubernetes
