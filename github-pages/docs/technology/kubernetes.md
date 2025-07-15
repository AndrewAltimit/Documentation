---
layout: default
title: Kubernetes
---

# Kubernetes

<html><header><link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css"></header></html>

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">Kubernetes</h1>
    <p class="hero-subtitle">Container Orchestration at Scale</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. Originally developed by Google and now maintained by the Cloud Native Computing Foundation (CNCF), Kubernetes has become the de facto standard for container orchestration in production environments.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-cubes"></i>
      <h4>Container Orchestration</h4>
      <p>Automated deployment and management</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-expand-arrows-alt"></i>
      <h4>Auto-scaling</h4>
      <p>Dynamic resource allocation</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-sync-alt"></i>
      <h4>Self-healing</h4>
      <p>Automatic recovery and rollbacks</p>
    </div>
  </div>
</div>

## Core Concepts

<div class="architecture-section">
  <h3><i class="fas fa-sitemap"></i> Architecture Overview</h3>
  <p>Kubernetes follows a master-worker architecture:</p>
  
  <div class="architecture-visual">
    <svg viewBox="0 0 700 400" class="k8s-architecture">
      <!-- Control Plane -->
      <rect x="50" y="50" width="600" height="120" fill="#3498db" opacity="0.1" stroke="#3498db" stroke-width="2" />
      <text x="350" y="30" text-anchor="middle" font-size="16" font-weight="bold">Control Plane</text>
      
      <!-- API Server -->
      <rect x="70" y="70" width="100" height="80" fill="#e74c3c" opacity="0.5" stroke="#c0392b" stroke-width="2" />
      <text x="120" y="105" text-anchor="middle" font-size="11" fill="white">API Server</text>
      <text x="120" y="120" text-anchor="middle" font-size="9" fill="white">Gateway</text>
      
      <!-- etcd -->
      <rect x="190" y="70" width="100" height="80" fill="#27ae60" opacity="0.5" stroke="#229954" stroke-width="2" />
      <text x="240" y="105" text-anchor="middle" font-size="11" fill="white">etcd</text>
      <text x="240" y="120" text-anchor="middle" font-size="9" fill="white">State Store</text>
      
      <!-- Scheduler -->
      <rect x="310" y="70" width="100" height="80" fill="#f39c12" opacity="0.5" stroke="#d68910" stroke-width="2" />
      <text x="360" y="105" text-anchor="middle" font-size="11" fill="white">Scheduler</text>
      <text x="360" y="120" text-anchor="middle" font-size="9" fill="white">Pod Placement</text>
      
      <!-- Controller Manager -->
      <rect x="430" y="70" width="100" height="80" fill="#9b59b6" opacity="0.5" stroke="#7d3c98" stroke-width="2" />
      <text x="480" y="100" text-anchor="middle" font-size="11" fill="white">Controller</text>
      <text x="480" y="115" text-anchor="middle" font-size="11" fill="white">Manager</text>
      <text x="480" y="130" text-anchor="middle" font-size="9" fill="white">Controllers</text>
      
      <!-- Cloud Controller -->
      <rect x="550" y="70" width="80" height="80" fill="#1abc9c" opacity="0.5" stroke="#16a085" stroke-width="2" />
      <text x="590" y="100" text-anchor="middle" font-size="10" fill="white">Cloud</text>
      <text x="590" y="115" text-anchor="middle" font-size="10" fill="white">Controller</text>
      <text x="590" y="130" text-anchor="middle" font-size="9" fill="white">Manager</text>
      
      <!-- Worker Nodes -->
      <text x="350" y="210" text-anchor="middle" font-size="16" font-weight="bold">Worker Nodes</text>
      
      <!-- Node 1 -->
      <rect x="50" y="230" width="180" height="150" fill="#95a5a6" opacity="0.1" stroke="#7f8c8d" stroke-width="2" />
      <text x="140" y="250" text-anchor="middle" font-size="12">Node 1</text>
      
      <!-- kubelet -->
      <rect x="60" y="260" width="70" height="40" fill="#3498db" opacity="0.5" />
      <text x="95" y="285" text-anchor="middle" font-size="10" fill="white">kubelet</text>
      
      <!-- kube-proxy -->
      <rect x="150" y="260" width="70" height="40" fill="#e74c3c" opacity="0.5" />
      <text x="185" y="285" text-anchor="middle" font-size="10" fill="white">kube-proxy</text>
      
      <!-- Container runtime -->
      <rect x="60" y="310" width="160" height="40" fill="#27ae60" opacity="0.5" />
      <text x="140" y="335" text-anchor="middle" font-size="10" fill="white">Container Runtime</text>
      
      <!-- Pods -->
      <circle cx="90" cy="365" r="12" fill="#f39c12" />
      <circle cx="140" cy="365" r="12" fill="#f39c12" />
      <circle cx="190" cy="365" r="12" fill="#f39c12" />
      <text x="140" y="370" text-anchor="middle" font-size="9">Pods</text>
      
      <!-- Node 2 -->
      <rect x="260" y="230" width="180" height="150" fill="#95a5a6" opacity="0.1" stroke="#7f8c8d" stroke-width="2" />
      <text x="350" y="250" text-anchor="middle" font-size="12">Node 2</text>
      
      <!-- Node 3 -->
      <rect x="470" y="230" width="180" height="150" fill="#95a5a6" opacity="0.1" stroke="#7f8c8d" stroke-width="2" />
      <text x="560" y="250" text-anchor="middle" font-size="12">Node 3</text>
      
      <!-- Communication lines -->
      <path d="M 120 150 L 140 230" stroke="#2c3e50" stroke-width="1" stroke-dasharray="3,3" />
      <path d="M 360 150 L 350 230" stroke="#2c3e50" stroke-width="1" stroke-dasharray="3,3" />
      <path d="M 480 150 L 560 230" stroke="#2c3e50" stroke-width="1" stroke-dasharray="3,3" />
    </svg>
  </div>
  
  <div class="component-details">
    <div class="component-group control-plane">
      <h4><i class="fas fa-server"></i> Control Plane Components</h4>
      <div class="component-list">
        <div class="component-item">
          <i class="fas fa-plug"></i>
          <strong>API Server:</strong> Central management point, exposes Kubernetes API
        </div>
        <div class="component-item">
          <i class="fas fa-database"></i>
          <strong>etcd:</strong> Distributed key-value store for cluster state
        </div>
        <div class="component-item">
          <i class="fas fa-calendar-alt"></i>
          <strong>Scheduler:</strong> Assigns pods to nodes based on resource requirements
        </div>
        <div class="component-item">
          <i class="fas fa-cogs"></i>
          <strong>Controller Manager:</strong> Runs controller processes
        </div>
        <div class="component-item">
          <i class="fas fa-cloud"></i>
          <strong>Cloud Controller Manager:</strong> Integrates with cloud provider APIs
        </div>
      </div>
    </div>
    
    <div class="component-group node-components">
      <h4><i class="fas fa-microchip"></i> Node Components</h4>
      <div class="component-list">
        <div class="component-item">
          <i class="fas fa-heartbeat"></i>
          <strong>kubelet:</strong> Ensures containers are running in pods
        </div>
        <div class="component-item">
          <i class="fas fa-network-wired"></i>
          <strong>kube-proxy:</strong> Maintains network rules for pod communication
        </div>
        <div class="component-item">
          <i class="fas fa-box"></i>
          <strong>Container Runtime:</strong> Docker, containerd, or CRI-O
        </div>
      </div>
    </div>
  </div>
</div>

### Kubernetes Objects

<div class="k8s-objects-section">
  <div class="object-card pod-object">
    <div class="object-header">
      <i class="fas fa-cube"></i>
      <h4>Pods</h4>
    </div>
    <p class="object-desc">The smallest deployable unit in Kubernetes:</p>
    
    <div class="object-visual">
      <svg viewBox="0 0 300 150">
        <!-- Pod outline -->
        <rect x="50" y="30" width="200" height="90" rx="10" fill="#3498db" opacity="0.2" stroke="#3498db" stroke-width="2" />
        <text x="150" y="20" text-anchor="middle" font-size="12" font-weight="bold">Pod</text>
        
        <!-- Containers inside pod -->
        <rect x="70" y="50" width="70" height="50" fill="#e74c3c" opacity="0.5" rx="5" />
        <text x="105" y="80" text-anchor="middle" font-size="10" fill="white">Container 1</text>
        
        <rect x="160" y="50" width="70" height="50" fill="#27ae60" opacity="0.5" rx="5" />
        <text x="195" y="80" text-anchor="middle" font-size="10" fill="white">Container 2</text>
        
        <!-- Shared resources -->
        <text x="150" y="130" text-anchor="middle" font-size="9">Shared Network & Storage</text>
      </svg>
    </div>
    
    <div class="code-example">
      <div class="code-header">Pod Definition</div>
      <pre><code class="language-yaml">apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    ports:
    - containerPort: 80</code></pre>
    </div>
    
    <div class="key-features">
      <h5>Key Features:</h5>
      <div class="feature-grid">
        <div class="feature-item">
          <i class="fas fa-layer-group"></i>
          <span>One or more containers</span>
        </div>
        <div class="feature-item">
          <i class="fas fa-share-alt"></i>
          <span>Shared network and storage</span>
        </div>
        <div class="feature-item">
          <i class="fas fa-hourglass-half"></i>
          <span>Ephemeral by design</span>
        </div>
        <div class="feature-item">
          <i class="fas fa-fingerprint"></i>
          <span>Unique IP address</span>
        </div>
      </div>
    </div>
  </div>

  <div class="object-card deployment-object">
    <div class="object-header">
      <i class="fas fa-rocket"></i>
      <h4>Deployments</h4>
    </div>
    <p class="object-desc">Manages replica sets and provides declarative updates:</p>
    
    <div class="object-visual">
      <svg viewBox="0 0 400 200">
        <!-- Deployment controller -->
        <rect x="150" y="20" width="100" height="40" fill="#9b59b6" opacity="0.5" stroke="#8e44ad" stroke-width="2" />
        <text x="200" y="45" text-anchor="middle" font-size="11" fill="white">Deployment</text>
        
        <!-- ReplicaSet -->
        <rect x="125" y="80" width="150" height="40" fill="#3498db" opacity="0.3" stroke="#2980b9" stroke-width="2" />
        <text x="200" y="105" text-anchor="middle" font-size="10">ReplicaSet</text>
        
        <!-- Pods -->
        <circle cx="120" cy="160" r="20" fill="#e74c3c" opacity="0.5" />
        <text x="120" y="165" text-anchor="middle" font-size="9" fill="white">Pod</text>
        
        <circle cx="200" cy="160" r="20" fill="#e74c3c" opacity="0.5" />
        <text x="200" y="165" text-anchor="middle" font-size="9" fill="white">Pod</text>
        
        <circle cx="280" cy="160" r="20" fill="#e74c3c" opacity="0.5" />
        <text x="280" y="165" text-anchor="middle" font-size="9" fill="white">Pod</text>
        
        <!-- Arrows -->
        <path d="M 200 60 L 200 75" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
        <path d="M 150 120 L 120 135" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
        <path d="M 200 120 L 200 135" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
        <path d="M 250 120 L 280 135" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
        
        <text x="330" y="105" font-size="10">Replicas: 3</text>
      </svg>
    </div>
    
    <div class="code-example">
      <div class="code-header">Deployment Definition</div>
      <pre><code class="language-yaml">apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80</code></pre>
    </div>
    
    <div class="deployment-features">
      <h5>Features:</h5>
      <div class="feature-cards">
        <div class="feature-card">
          <i class="fas fa-sync-alt"></i>
          <h6>Rolling Updates</h6>
          <p>Zero-downtime deployments</p>
        </div>
        <div class="feature-card">
          <i class="fas fa-undo"></i>
          <h6>Rollback</h6>
          <p>Revert to previous versions</p>
        </div>
        <div class="feature-card">
          <i class="fas fa-expand-arrows-alt"></i>
          <h6>Scaling</h6>
          <p>Adjust replica count</p>
        </div>
        <div class="feature-card">
          <i class="fas fa-heartbeat"></i>
          <h6>Self-healing</h6>
          <p>Automatic pod recovery</p>
        </div>
      </div>
    </div>
  </div>

  <div class="object-card service-object">
    <div class="object-header">
      <i class="fas fa-network-wired"></i>
      <h4>Services</h4>
    </div>
    <p class="object-desc">Provides stable network endpoint for pods:</p>
    
    <div class="service-types-visual">
      <h5>Service Types</h5>
      <div class="service-type-grid">
        <div class="service-type clusterip">
          <svg viewBox="0 0 150 120">
            <rect x="30" y="30" width="90" height="60" fill="#3498db" opacity="0.2" stroke="#3498db" stroke-width="2" />
            <text x="75" y="20" text-anchor="middle" font-size="10" font-weight="bold">ClusterIP</text>
            <circle cx="50" cy="60" r="8" fill="#e74c3c" />
            <circle cx="75" cy="60" r="8" fill="#e74c3c" />
            <circle cx="100" cy="60" r="8" fill="#e74c3c" />
            <text x="75" y="105" text-anchor="middle" font-size="9">Internal Only</text>
          </svg>
        </div>
        
        <div class="service-type nodeport">
          <svg viewBox="0 0 150 120">
            <rect x="30" y="30" width="90" height="60" fill="#27ae60" opacity="0.2" stroke="#27ae60" stroke-width="2" />
            <text x="75" y="20" text-anchor="middle" font-size="10" font-weight="bold">NodePort</text>
            <circle cx="75" cy="60" r="8" fill="#e74c3c" />
            <line x1="75" y1="52" x2="75" y2="10" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow)" />
            <text x="75" y="105" text-anchor="middle" font-size="9">Node IP:Port</text>
          </svg>
        </div>
        
        <div class="service-type loadbalancer">
          <svg viewBox="0 0 150 120">
            <ellipse cx="75" cy="15" rx="40" ry="10" fill="#f39c12" opacity="0.3" />
            <text x="75" y="18" text-anchor="middle" font-size="9">LB</text>
            <rect x="30" y="40" width="90" height="50" fill="#f39c12" opacity="0.2" stroke="#f39c12" stroke-width="2" />
            <text x="75" y="35" text-anchor="middle" font-size="10" font-weight="bold">LoadBalancer</text>
            <circle cx="75" cy="65" r="8" fill="#e74c3c" />
            <text x="75" y="105" text-anchor="middle" font-size="9">External LB</text>
          </svg>
        </div>
        
        <div class="service-type externalname">
          <svg viewBox="0 0 150 120">
            <rect x="30" y="30" width="90" height="60" fill="#9b59b6" opacity="0.2" stroke="#9b59b6" stroke-width="2" />
            <text x="75" y="20" text-anchor="middle" font-size="10" font-weight="bold">ExternalName</text>
            <text x="75" y="60" text-anchor="middle" font-size="16">DNS</text>
            <text x="75" y="105" text-anchor="middle" font-size="9">Maps to DNS</text>
          </svg>
        </div>
      </div>
    </div>
    
    <div class="code-example">
      <div class="code-header">Service Definition</div>
      <pre><code class="language-yaml">apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer</code></pre>
    </div>
  </div>

#### ConfigMaps and Secrets

**ConfigMap Example:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgres://localhost:5432/mydb"
  api_key: "public-api-key"
```

**Secret Example:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  username: YWRtaW4=  # base64 encoded
  password: cGFzc3dvcmQ=  # base64 encoded
```

  <div class="object-card namespace-object">
    <div class="object-header">
      <i class="fas fa-folder"></i>
      <h4>Namespaces</h4>
    </div>
    <p class="object-desc">Logical isolation within a cluster:</p>
    
    <div class="namespace-visual">
      <svg viewBox="0 0 400 250">
        <!-- Cluster boundary -->
        <rect x="20" y="20" width="360" height="210" fill="none" stroke="#2c3e50" stroke-width="2" stroke-dasharray="5,5" />
        <text x="200" y="15" text-anchor="middle" font-size="12" font-weight="bold">Kubernetes Cluster</text>
        
        <!-- Default namespace -->
        <rect x="40" y="40" width="150" height="80" fill="#3498db" opacity="0.2" stroke="#3498db" stroke-width="2" />
        <text x="115" y="60" text-anchor="middle" font-size="11" font-weight="bold">default</text>
        <circle cx="70" cy="90" r="8" fill="#e74c3c" />
        <circle cx="100" cy="90" r="8" fill="#e74c3c" />
        <circle cx="130" cy="90" r="8" fill="#e74c3c" />
        <text x="100" y="110" text-anchor="middle" font-size="9">User Apps</text>
        
        <!-- kube-system namespace -->
        <rect x="210" y="40" width="150" height="80" fill="#e74c3c" opacity="0.2" stroke="#e74c3c" stroke-width="2" />
        <text x="285" y="60" text-anchor="middle" font-size="11" font-weight="bold">kube-system</text>
        <rect cx="240" cy="85" width="12" height="12" fill="#c0392b" />
        <rect cx="270" cy="85" width="12" height="12" fill="#c0392b" />
        <rect cx="300" cy="85" width="12" height="12" fill="#c0392b" />
        <text x="270" y="110" text-anchor="middle" font-size="9">System Pods</text>
        
        <!-- Development namespace -->
        <rect x="40" y="140" width="150" height="70" fill="#27ae60" opacity="0.2" stroke="#27ae60" stroke-width="2" />
        <text x="115" y="160" text-anchor="middle" font-size="11" font-weight="bold">development</text>
        <circle cx="70" cy="185" r="8" fill="#229954" />
        <circle cx="100" cy="185" r="8" fill="#229954" />
        <text x="85" y="205" text-anchor="middle" font-size="9">Dev Apps</text>
        
        <!-- Production namespace -->
        <rect x="210" y="140" width="150" height="70" fill="#f39c12" opacity="0.2" stroke="#f39c12" stroke-width="2" />
        <text x="285" y="160" text-anchor="middle" font-size="11" font-weight="bold">production</text>
        <circle cx="240" cy="185" r="8" fill="#d68910" />
        <circle cx="270" cy="185" r="8" fill="#d68910" />
        <text x="255" y="205" text-anchor="middle" font-size="9">Prod Apps</text>
      </svg>
    </div>
    
    <div class="code-example">
      <div class="code-header">Namespace Definition</div>
      <pre><code class="language-yaml">apiVersion: v1
kind: Namespace
metadata:
  name: development</code></pre>
    </div>
    
    <div class="default-namespaces">
      <h5>Default Namespaces:</h5>
      <div class="namespace-list">
        <div class="namespace-item">
          <code>default</code>
          <span>Default namespace for objects</span>
        </div>
        <div class="namespace-item">
          <code>kube-system</code>
          <span>Kubernetes system objects</span>
        </div>
        <div class="namespace-item">
          <code>kube-public</code>
          <span>Publicly accessible data</span>
        </div>
        <div class="namespace-item">
          <code>kube-node-lease</code>
          <span>Node heartbeat data</span>
        </div>
      </div>
    </div>
  </div>
</div>

## Workload Resources

### StatefulSets

For stateful applications requiring stable identities:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

**Features:**
- Ordered deployment and scaling
- Stable network identities
- Persistent storage
- Ordered termination

### DaemonSets

Ensures pods run on all (or selected) nodes:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluentd:v1.14
        resources:
          limits:
            memory: 200Mi
```

**Use Cases:**
- Log collection
- Monitoring agents
- Network plugins
- Storage drivers

### Jobs and CronJobs

**Job Example:**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: backup-job
spec:
  template:
    spec:
      containers:
      - name: backup
        image: backup-tool:latest
        command: ["./backup.sh"]
      restartPolicy: OnFailure
  backoffLimit: 3
```

**CronJob Example:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:latest
          restartPolicy: OnFailure
```

## Networking

### Cluster Networking

**Network Model Requirements:**
- All pods can communicate with each other
- All nodes can communicate with all pods
- No NAT between pods

### Ingress

HTTP/HTTPS routing to services:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### Network Policies

Fine-grained network access control:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-netpol
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

## Storage

### Container Storage Interface (CSI) Architecture

The Container Storage Interface defines a standard for exposing block and file storage systems to containerized workloads:

```go
// CSI Controller Service - Manages volumes at cluster level
type ControllerServer interface {
    // Volume lifecycle operations
    CreateVolume(context.Context, *CreateVolumeRequest) (*CreateVolumeResponse, error)
    DeleteVolume(context.Context, *DeleteVolumeRequest) (*DeleteVolumeResponse, error)
    
    // Snapshot operations
    CreateSnapshot(context.Context, *CreateSnapshotRequest) (*CreateSnapshotResponse, error)
    DeleteSnapshot(context.Context, *DeleteSnapshotRequest) (*DeleteSnapshotResponse, error)
    
    // Volume attachment
    ControllerPublishVolume(context.Context, *ControllerPublishVolumeRequest) (*ControllerPublishVolumeResponse, error)
    ControllerUnpublishVolume(context.Context, *ControllerUnpublishVolumeRequest) (*ControllerUnpublishVolumeResponse, error)
    
    // Capabilities
    ControllerGetCapabilities(context.Context, *ControllerGetCapabilitiesRequest) (*ControllerGetCapabilitiesResponse, error)
}

// CSI Node Service - Manages volumes on individual nodes
type NodeServer interface {
    // Node operations
    NodeStageVolume(context.Context, *NodeStageVolumeRequest) (*NodeStageVolumeResponse, error)
    NodeUnstageVolume(context.Context, *NodeUnstageVolumeRequest) (*NodeUnstageVolumeResponse, error)
    
    // Mount operations
    NodePublishVolume(context.Context, *NodePublishVolumeRequest) (*NodePublishVolumeResponse, error)
    NodeUnpublishVolume(context.Context, *NodeUnpublishVolumeRequest) (*NodeUnpublishVolumeResponse, error)
    
    // Node info
    NodeGetInfo(context.Context, *NodeGetInfoRequest) (*NodeGetInfoResponse, error)
    NodeGetCapabilities(context.Context, *NodeGetCapabilitiesRequest) (*NodeGetCapabilitiesResponse, error)
}
```

### Storage Orchestration Theory

#### Volume Lifecycle State Machine

```python
from enum import Enum
from typing import Dict, List, Optional, Set
import asyncio

class VolumeState(Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    AVAILABLE = "available"
    BINDING = "binding"
    BOUND = "bound"
    RELEASING = "releasing"
    FAILED = "failed"
    DELETED = "deleted"

class VolumeLifecycleManager:
    """Manages volume state transitions with formal verification"""
    
    def __init__(self):
        # State transition graph
        self.transitions = {
            VolumeState.PENDING: {VolumeState.PROVISIONING, VolumeState.FAILED},
            VolumeState.PROVISIONING: {VolumeState.AVAILABLE, VolumeState.FAILED},
            VolumeState.AVAILABLE: {VolumeState.BINDING, VolumeState.DELETED},
            VolumeState.BINDING: {VolumeState.BOUND, VolumeState.AVAILABLE, VolumeState.FAILED},
            VolumeState.BOUND: {VolumeState.RELEASING},
            VolumeState.RELEASING: {VolumeState.AVAILABLE, VolumeState.DELETED, VolumeState.FAILED},
            VolumeState.FAILED: {VolumeState.DELETED},
            VolumeState.DELETED: set()
        }
        
        self.volumes: Dict[str, VolumeState] = {}
        self.state_lock = asyncio.Lock()
    
    async def transition_volume(self, volume_id: str, new_state: VolumeState) -> bool:
        """Atomically transition volume state with validation"""
        async with self.state_lock:
            current_state = self.volumes.get(volume_id, VolumeState.PENDING)
            
            # Verify transition is valid
            if new_state not in self.transitions[current_state]:
                raise ValueError(f"Invalid transition: {current_state} → {new_state}")
            
            # Perform state transition
            self.volumes[volume_id] = new_state
            
            # Trigger side effects
            await self._handle_state_change(volume_id, current_state, new_state)
            
            return True
    
    async def _handle_state_change(self, volume_id: str, 
                                  old_state: VolumeState, 
                                  new_state: VolumeState):
        """Handle side effects of state transitions"""
        if new_state == VolumeState.PROVISIONING:
            await self._provision_volume(volume_id)
        elif new_state == VolumeState.BOUND:
            await self._attach_volume(volume_id)
        elif new_state == VolumeState.DELETED:
            await self._cleanup_volume(volume_id)
```

### Advanced Storage Patterns

#### Distributed Volume Replication

```python
import hashlib
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VolumeReplica:
    """Represents a replica of a volume"""
    node_id: str
    replica_id: str
    state: str  # primary, secondary, syncing
    last_sync: float
    checksum: str

class DistributedVolumeManager:
    """Manages distributed volume replication with consistency guarantees"""
    
    def __init__(self, replication_factor: int = 3):
        self.replication_factor = replication_factor
        self.replicas: Dict[str, List[VolumeReplica]] = {}
        self.quorum_size = (replication_factor // 2) + 1
    
    def calculate_placement(self, volume_id: str, nodes: List[str]) -> List[str]:
        """
        Deterministic replica placement using consistent hashing
        with failure domain awareness
        """
        # Sort nodes for deterministic ordering
        sorted_nodes = sorted(nodes)
        
        # Use consistent hashing for placement
        placements = []
        for i in range(self.replication_factor):
            hash_input = f"{volume_id}:{i}".encode()
            hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
            node_index = hash_value % len(sorted_nodes)
            
            # Ensure replicas are on different nodes
            selected_node = sorted_nodes[node_index]
            attempts = 0
            while selected_node in placements and attempts < len(sorted_nodes):
                node_index = (node_index + 1) % len(sorted_nodes)
                selected_node = sorted_nodes[node_index]
                attempts += 1
            
            if selected_node not in placements:
                placements.append(selected_node)
        
        return placements[:self.replication_factor]
    
    async def write_with_quorum(self, volume_id: str, data: bytes) -> bool:
        """
        Write data to volume with quorum consensus
        """
        replicas = self.replicas.get(volume_id, [])
        if len(replicas) < self.quorum_size:
            raise ValueError(f"Insufficient replicas for quorum: {len(replicas)} < {self.quorum_size}")
        
        # Parallel writes to all replicas
        write_tasks = []
        for replica in replicas:
            if replica.state in ["primary", "secondary"]:
                write_tasks.append(self._write_to_replica(replica, data))
        
        # Wait for quorum
        results = await asyncio.gather(*write_tasks, return_exceptions=True)
        successful_writes = sum(1 for r in results if r is True)
        
        return successful_writes >= self.quorum_size
```

### Storage Performance Optimization

#### I/O Scheduler and Cache Management

```python
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

@dataclass
class IORequest:
    """Represents an I/O request with QoS parameters"""
    request_id: str
    volume_id: str
    offset: int
    length: int
    operation: str  # read, write
    priority: int
    deadline: float
    submitted_at: float
    
    def __lt__(self, other):
        # For priority queue ordering
        return self.deadline < other.deadline

class StorageScheduler:
    """
    Advanced I/O scheduler implementing deadline-aware scheduling
    with fairness guarantees
    """
    
    def __init__(self, queue_depth: int = 128):
        self.queue_depth = queue_depth
        self.pending_requests: List[IORequest] = []  # Min-heap by deadline
        self.volume_queues: Dict[str, Deque[IORequest]] = {}
        self.volume_tokens: Dict[str, int] = {}  # Token bucket for fairness
        
    def submit_request(self, request: IORequest):
        """Submit I/O request with deadline and priority"""
        # Add to priority queue
        heapq.heappush(self.pending_requests, request)
        
        # Add to per-volume queue for fairness
        if request.volume_id not in self.volume_queues:
            self.volume_queues[request.volume_id] = deque()
            self.volume_tokens[request.volume_id] = 10  # Initial tokens
        
        self.volume_queues[request.volume_id].append(request)
    
    def get_next_request(self) -> Optional[IORequest]:
        """
        Get next request using Earliest Deadline First (EDF)
        with token-based fairness
        """
        if not self.pending_requests:
            return None
        
        # Check deadline violations
        current_time = time.time()
        while self.pending_requests:
            candidate = self.pending_requests[0]
            
            # Check if volume has tokens (fairness)
            if self.volume_tokens.get(candidate.volume_id, 0) > 0:
                # Deduct token and process request
                self.volume_tokens[candidate.volume_id] -= 1
                return heapq.heappop(self.pending_requests)
            else:
                # Move to next deadline
                heapq.heappop(self.pending_requests)
                heapq.heappush(self.pending_requests, candidate)
                break
        
        # Replenish tokens periodically
        self._replenish_tokens()
        
        return None
    
    def _replenish_tokens(self):
        """Token bucket algorithm for fairness"""
        for volume_id in self.volume_tokens:
            self.volume_tokens[volume_id] = min(
                self.volume_tokens[volume_id] + 1,
                10  # Max tokens
            )
```

### Storage Classes with Advanced Features

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: high-performance-replicated
provisioner: csi.storage.k8s.io/advanced-driver
parameters:
  # Performance tier
  type: nvme-ssd
  iops: "50000"
  throughput: "1000Mi"
  
  # Replication settings
  replicationFactor: "3"
  replicationMode: "synchronous"  # synchronous, asynchronous
  consistencyLevel: "strong"      # strong, eventual
  
  # Data protection
  encryption: "aes-256-gcm"
  checksumAlgorithm: "crc32c"
  
  # Advanced features
  deduplication: "true"
  compression: "lz4"
  snapshotSchedule: "0 */6 * * *"  # Every 6 hours
  
  # QoS settings
  qosClass: "guaranteed"
  minIOPS: "10000"
  maxIOPS: "50000"
  burstDuration: "60s"
  
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

### Volume Snapshot Controller Implementation

```go
// VolumeSnapshotController manages snapshot lifecycle
type VolumeSnapshotController struct {
    client           kubernetes.Interface
    snapshotClient   snapclient.Interface
    csiClient        csi.ControllerClient
    queue            workqueue.RateLimitingInterface
    snapshotStore    cache.Store
    volumeStore      cache.Store
}

func (c *VolumeSnapshotController) syncSnapshot(key string) error {
    namespace, name, err := cache.SplitMetaNamespaceKey(key)
    if err != nil {
        return err
    }
    
    snapshot, err := c.snapshotStore.Get(key)
    if err != nil {
        return err
    }
    
    vs := snapshot.(*v1.VolumeSnapshot)
    
    switch vs.Status.Phase {
    case v1.SnapshotPending:
        return c.createSnapshot(vs)
    case v1.SnapshotCreating:
        return c.checkSnapshotStatus(vs)
    case v1.SnapshotReady:
        return c.bindSnapshotContent(vs)
    case v1.SnapshotFailed:
        return c.handleSnapshotFailure(vs)
    }
    
    return nil
}

func (c *VolumeSnapshotController) createSnapshot(vs *v1.VolumeSnapshot) error {
    // Get source PVC
    pvc, err := c.getPVC(vs.Spec.Source.PersistentVolumeClaimName, vs.Namespace)
    if err != nil {
        return err
    }
    
    // Get CSI driver from PV
    pv, err := c.getPV(pvc.Spec.VolumeName)
    if err != nil {
        return err
    }
    
    // Create snapshot via CSI
    req := &csi.CreateSnapshotRequest{
        SourceVolumeId: pv.Spec.CSI.VolumeHandle,
        Name:          vs.Name,
        Parameters:    vs.Spec.Parameters,
    }
    
    resp, err := c.csiClient.CreateSnapshot(context.TODO(), req)
    if err != nil {
        return c.updateSnapshotStatus(vs, v1.SnapshotFailed, err.Error())
    }
    
    // Update status
    return c.updateSnapshotStatus(vs, v1.SnapshotCreating, resp.Snapshot.SnapshotId)
}
```

## Configuration Management

### Resource Management

**Resource Requests and Limits:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: resource-demo
spec:
  containers:
  - name: app
    image: myapp:latest
    resources:
      requests:
        memory: "256Mi"
        cpu: "500m"
      limits:
        memory: "512Mi"
        cpu: "1000m"
```

### Horizontal Pod Autoscaling

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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaling

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

## Security

### Role-Based Access Control (RBAC)

**Role Definition:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
```

**RoleBinding:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: default
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### Pod Security Standards

**Pod Security Policy Example:**
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### Service Accounts

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: default
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  template:
    spec:
      serviceAccountName: app-service-account
      containers:
      - name: app
        image: myapp:latest
```

## Observability

### Liveness and Readiness Probes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:latest
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

### Monitoring with Prometheus

**ServiceMonitor Example:**
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
    path: /metrics
```

## kubectl Commands

### Basic Commands

```bash
# Cluster info
kubectl cluster-info
kubectl get nodes

# Working with resources
kubectl get pods
kubectl get services
kubectl get deployments
kubectl describe pod <pod-name>

# Creating resources
kubectl create -f manifest.yaml
kubectl apply -f manifest.yaml

# Updating resources
kubectl edit deployment <deployment-name>
kubectl scale deployment <deployment-name> --replicas=5

# Deleting resources
kubectl delete pod <pod-name>
kubectl delete -f manifest.yaml
```

### Advanced Commands

```bash
# Port forwarding
kubectl port-forward pod/<pod-name> 8080:80

# Executing commands in pods
kubectl exec -it <pod-name> -- /bin/bash

# Viewing logs
kubectl logs <pod-name>
kubectl logs -f <pod-name>  # Follow logs

# Copying files
kubectl cp <pod-name>:/path/to/file ./local-file
kubectl cp ./local-file <pod-name>:/path/to/file

# Resource usage
kubectl top nodes
kubectl top pods

# Debugging
kubectl describe pod <pod-name>
kubectl get events --sort-by=.metadata.creationTimestamp
```

## Best Practices

### Resource Organization
- Use namespaces for environment separation
- Label resources consistently
- Use resource quotas and limits
- Implement pod disruption budgets

### Security
- Enable RBAC
- Use network policies
- Run containers as non-root
- Scan images for vulnerabilities
- Use secrets for sensitive data

### High Availability
- Run multiple replicas
- Use pod anti-affinity rules
- Implement health checks
- Use horizontal pod autoscaling

### Monitoring and Logging
- Centralize logging
- Implement comprehensive monitoring
- Set up alerts
- Use distributed tracing

## Helm - Kubernetes Package Manager

### Chart Structure
```
mychart/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
└── charts/
```

### Basic Helm Commands
```bash
# Install a chart
helm install myrelease ./mychart

# Upgrade a release
helm upgrade myrelease ./mychart

# List releases
helm list

# Rollback
helm rollback myrelease 1

# Uninstall
helm uninstall myrelease
```

## Common Patterns

### Sidecar Pattern
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-sidecar
spec:
  containers:
  - name: app
    image: myapp:latest
  - name: sidecar
    image: logging-agent:latest
    volumeMounts:
    - name: shared-logs
      mountPath: /var/log
  volumes:
  - name: shared-logs
    emptyDir: {}
```

### Init Containers
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-init
spec:
  initContainers:
  - name: init-db
    image: busybox:1.28
    command: ['sh', '-c', 'until nc -z db-service 5432; do sleep 1; done']
  containers:
  - name: app
    image: myapp:latest
```

## Troubleshooting

### Common Issues

1. **ImagePullBackOff**
   - Check image name and tag
   - Verify registry credentials
   - Check network connectivity

2. **CrashLoopBackOff**
   - Check application logs
   - Verify resource limits
   - Check liveness probe configuration

3. **Pending Pods**
   - Check resource availability
   - Verify node selectors
   - Check persistent volume claims

4. **Service Connection Issues**
   - Verify service selectors
   - Check network policies
   - Validate DNS resolution

Kubernetes provides a robust platform for deploying and managing containerized applications at scale. Its declarative approach, combined with powerful abstractions and extensive ecosystem, makes it an essential tool for modern cloud-native development.

## See Also
- [Docker](docker.html) - Container fundamentals and image creation
- [AWS](aws.html) - Managed Kubernetes services (EKS)
- [Terraform](terraform.html) - Infrastructure as Code for Kubernetes
- [Networking](networking.html) - Network concepts and protocols
- [Database Design](database-design.html) - Stateful applications in Kubernetes
- [Cybersecurity](cybersecurity.html) - Container and cluster security