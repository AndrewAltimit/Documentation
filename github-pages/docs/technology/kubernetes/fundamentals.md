---
layout: docs
title: "Kubernetes: Fundamentals"
permalink: /docs/technology/kubernetes/fundamentals.html
toc: true
toc_sticky: true
---

## Documentation Structure

This guide covers Kubernetes comprehensively across the following topics:

1. **Quick Start** - Essential commands and operations
2. **Core Concepts** - Fundamental Kubernetes principles
3. **Kubernetes Objects** - API object specifications
4. **Workload Resources** - Application deployment patterns
5. **Networking & Storage** - Infrastructure integration
6. **Configuration & Security** - Production configurations
7. **Advanced Topics** - Complex deployment scenarios
8. **Real-World Applications** - Implementation case studies
9. **Latest Updates** - Recent features and changes

## Quick Start Guide

Kubernetes functions as an operating system for distributed systems - it manages applications (containers) across multiple computers (nodes) similar to how an OS manages programs on a single computer. This section demonstrates essential operations.

### Requirements
- Container technology knowledge (Docker)
- Kubernetes cluster access (minikube, kind, k3s, or cloud provider)
- kubectl CLI v1.28+ installed
- Optional: Helm 3.x for package management

### Your First Deployment

```bash
# 1. Check your cluster is running and version
kubectl cluster-info
kubectl version --short

# 2. Deploy a simple application
kubectl create deployment hello-world --image=nginx:alpine

# 3. Expose it to the internet
kubectl expose deployment hello-world --type=LoadBalancer --port=80

# 4. Check it's running with more details
kubectl get pods -o wide
kubectl get services
kubectl describe deployment hello-world

# 5. Scale it up
kubectl scale deployment hello-world --replicas=3

# 6. See the magic - kill a pod and watch it resurrect
kubectl get pods
kubectl delete pod <pod-name>
kubectl get pods  # Notice a new pod replaced the deleted one!

# 7. Clean up
kubectl delete deployment hello-world
kubectl delete service hello-world
```

### Core Concepts in Plain English

**Pods**: The smallest unit in Kubernetes. Like a wrapper around your container(s) that provides shared storage and network.

**Deployments**: Manages your pods. Tells Kubernetes "I want 3 copies of my app running at all times" and Kubernetes makes it happen.

**Services**: Provides a stable address for your pods. Even as pods die and restart with new IPs, the service address stays the same.

**Nodes**: The physical or virtual machines that run your containers.

**Cluster**: A group of nodes managed by Kubernetes.

That's it! You now understand the basics. The rest of this guide will take you deep into each concept and show you how to build production-ready systems.

## Understanding Kubernetes: From Containers to Orchestration

Before diving into Kubernetes architecture, let's understand why we need container orchestration. When you run a single container on your laptop, Docker is perfectly adequate. But what happens when you need to run hundreds or thousands of containers across multiple servers? How do you ensure they're always running, properly networked, and efficiently using resources? This is where Kubernetes comes in.

### The Evolution: From Bare Metal to Kubernetes

1. **Bare Metal Era**: One application per server, massive waste of resources
2. **Virtualization Era**: Multiple VMs per server, better resource usage but heavy overhead
3. **Container Era**: Lightweight isolation, but manual management becomes complex at scale
4. **Orchestration Era**: Kubernetes automates container management, enabling true cloud-native applications

### Why Kubernetes Wins

- **Declarative Configuration**: You describe the desired state, Kubernetes makes it happen
- **Self-Healing**: Automatically replaces failed containers
- **Service Discovery**: Built-in DNS and load balancing
- **Storage Orchestration**: Automatically mounts storage systems
- **Automated Rollouts**: Deploy new versions without downtime
- **Secret Management**: Secure handling of sensitive data
- **Horizontal Scaling**: Scale with a simple command or automatically based on CPU/memory usage

## Core Concepts

<div class="architecture-section">
  <h3><i class="fas fa-sitemap"></i> Architecture Overview</h3>
  <p>Kubernetes operates like a distributed operating system for your containers. It follows a master-worker architecture where a control plane manages the entire cluster while worker nodes run your actual applications. Think of it as a conductor (control plane) orchestrating an orchestra (worker nodes) to create harmonious music (your running applications).</p>
  
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

### Kubernetes Objects: The Building Blocks

Now that we understand how Kubernetes is structured, let's explore the objects we use to define our applications. These objects are like LEGO blocks - each serves a specific purpose, and when combined, they create complete application architectures. We'll start with the most fundamental unit and build up to more complex constructs.

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

#### ConfigMaps and Secrets: Managing Application Configuration

As your applications grow, you'll need to manage configuration separately from your container images. This separation allows you to deploy the same image across different environments (development, staging, production) with different configurations. ConfigMaps handle non-sensitive data, while Secrets manage sensitive information like passwords and API keys.

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

## Workload Resources: Beyond Basic Deployments

While Deployments handle stateless applications beautifully, real-world systems often need more specialized controllers. Let's explore how Kubernetes handles different workload patterns, each designed to solve specific operational challenges.

### StatefulSets: When Order and Identity Matter

Imagine deploying a database cluster where each instance needs a persistent identity and stable storage. Regular Deployments treat pods as interchangeable cattle, but StatefulSets treat them as pets with names and persistent characteristics. This is crucial for applications like databases, message queues, and distributed systems that rely on stable network identities and ordered operations.

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

### DaemonSets: One Pod Per Node

Some tasks need to run on every node in your cluster - think of monitoring agents, log collectors, or network plugins. DaemonSets solve this by automatically deploying exactly one pod per node, even as nodes are added or removed from the cluster. It's like having a building superintendent in every apartment building of a large complex.

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

### Jobs and CronJobs: Task Automation

Not all workloads run continuously. Sometimes you need to run a task once (like a database migration) or on a schedule (like nightly backups). Jobs ensure tasks complete successfully, while CronJobs handle recurring schedules - bringing Unix-like cron functionality to your containerized workloads.

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

## Networking: Connecting Your Applications

Kubernetes networking might seem complex at first, but it follows simple principles designed to make communication between containers as straightforward as possible. Let's explore how Kubernetes creates a flat network where every pod can communicate with every other pod, regardless of which node they're on.

### Cluster Networking Fundamentals

**Network Model Requirements:**
- All pods can communicate with each other
- All nodes can communicate with all pods
- No NAT between pods

### Ingress: Your Cluster's Front Door

While Services provide internal load balancing, Ingress acts as your cluster's intelligent front door. It handles HTTP/HTTPS routing, SSL termination, and path-based routing - essentially replacing the need for multiple LoadBalancer services with a single entry point that can route to multiple backend services based on hostnames and paths.

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

### Network Policies: Microsegmentation for Security

In a flat network where everything can talk to everything, security becomes paramount. Network Policies act like firewall rules at the pod level, allowing you to control which pods can communicate with each other. This microsegmentation approach is crucial for implementing zero-trust security models in your cluster.

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

