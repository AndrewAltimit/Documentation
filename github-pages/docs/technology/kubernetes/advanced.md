---
layout: docs
title: "Kubernetes: Advanced Topics"
permalink: /docs/technology/kubernetes/advanced.html
toc: true
toc_sticky: true
---

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

#### Istio Architecture

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

#### Circuit Breaking and Retry Logic

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

2. **Multi-Region Failover**:
   ```yaml
   # Federation v2 example
   apiVersion: types.kubefed.io/v1beta1
   kind: FederatedDeployment
   metadata:
     name: app
     namespace: production
   spec:
     placement:
       clusters:
       - name: us-east-1
       - name: us-west-2
       - name: eu-west-1
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

## Kubernetes Updates: Latest Features and Best Practices

### What's New in Kubernetes v1.29 (Mandala)
1. **ReadinessGates for Jobs**: Better job lifecycle management
2. **Sidecar Containers**: Native support for sidecar patterns
3. **In-Place Pod Vertical Scaling**: Resize pods without restart
4. **CEL for Admission Control**: Common Expression Language support
5. **Structured Authentication Configuration**: Enhanced security

### What's New in Kubernetes v1.28 (Planternetes)
1. **Cgroups v2 GA**: Better resource isolation
2. **Mixed CPUs Support**: Heterogeneous CPU architectures
3. **Persistent Volume Last Phase Transition Time**: Better storage monitoring
4. **Non-Graceful Node Shutdown**: Improved failure handling

### Enhanced Security Features
```yaml
# Pod Security Standards (enforced by default)
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### AI/ML Workload Support
```yaml
# GPU scheduling improvements
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

### Gateway API v1.0 (GA)
```yaml
# Modern ingress replacement
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

### Performance Improvements
- **Improved etcd performance**: 30% faster for large clusters
- **API Priority and Fairness**: Better request handling
- **Efficient SELinux relabeling**: Faster pod startup
- **Memory manager improvements**: Better NUMA awareness

## Future of Kubernetes: What's Next?

### Emerging Trends (2024-2025)
1. **Serverless on Kubernetes**: Knative 1.12+, OpenFaaS, KEDA 2.13
2. **Edge Computing**: K3s, KubeEdge, Akri for leaf devices
3. **WebAssembly**: Running Wasm workloads with runwasi
4. **AI/ML Workloads**: Kubeflow 1.8, MLOps pipelines, Ray on K8s
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

## Conclusion

Kubernetes has revolutionized how we deploy and manage applications, but mastery requires understanding both its power and complexity. This guide has taken you from basic concepts to advanced production patterns. Remember:

1. Start simple, add complexity gradually
2. Automate everything possible
3. Monitor and observe religiously
4. Security is not optional
5. Practice disaster recovery before you need it

The Kubernetes ecosystem continues to evolve rapidly. Stay curious, keep learning, and remember that even experts were beginners once. Whether you're deploying your first pod or architecting multi-region clusters, Kubernetes provides the foundation for modern cloud-native applications.

## See Also
- [Docker](../docker/) - Container fundamentals and image creation
- [AWS](../aws/) - Managed Kubernetes services (EKS)
- [Terraform](../terraform/) - Infrastructure as Code for Kubernetes
- [Networking](../networking.html) - Network concepts and protocols
- [Database Design](../database-design.html) - Stateful applications in Kubernetes
