---
layout: docs
title: Quick Reference Guide
description: Comprehensive quick reference for commands, formulas, algorithms, and best practices
hide_title: true
toc: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Quick Reference Guide</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Your comprehensive resource for commands, formulas, algorithms, and best practices</p>
</div>

<div class="intro-card" markdown="1">
<p class="lead-text">A single-page cheat sheet for the things you look up constantly — CLI commands, physics constants and equations, Big-O complexity, API conventions, regex, and pre-flight checklists. Skim the cards, or jump straight to a section below. <strong>Tip:</strong> use <kbd>Ctrl</kbd>+<kbd>F</kbd> (<kbd>Cmd</kbd>+<kbd>F</kbd> on Mac) to find anything instantly.</p>
</div>

## Quick Navigation

<div class="command-grid">
  <div class="nav-card" markdown="1">
**[Command Line](#command-line-references)**
Git · Docker · Kubernetes · AWS CLI · Terraform
  </div>
  <div class="nav-card" markdown="1">
**[Physics & Math](#physics-formulas--constants)**
Constants · mechanics · QM · EM · calculus · linear algebra
  </div>
  <div class="nav-card" markdown="1">
**[Algorithms](#algorithms--data-structures)**
Big-O tables · patterns · templates
  </div>
  <div class="nav-card" markdown="1">
**[API & Network](#api-reference-patterns)**
REST · status codes · ports · regex
  </div>
  <div class="nav-card" markdown="1">
**[Troubleshooting](#troubleshooting-flowcharts)**
Docker · Git · deployment fixes
  </div>
  <div class="nav-card" markdown="1">
**[Best Practices](#best-practices-checklists)**
Code review · deployment · security
  </div>
</div>

---

## Command Line References

<div class="reference-card" markdown="1">

#### Git Commands

```bash
# Repository Management
git init                      # Initialize new repository
git clone <url>              # Clone remote repository
git remote -v                # List remote repositories
git remote add origin <url>  # Add remote repository

# Basic Operations
git add .                    # Stage all changes
git add <file>              # Stage specific file
git commit -m "message"     # Commit staged changes
git push origin <branch>    # Push to remote
git pull origin <branch>    # Pull from remote
git fetch                   # Fetch remote changes

# Branching
git branch                  # List branches
git branch <name>          # Create new branch
git checkout <branch>      # Switch branch
git checkout -b <branch>   # Create and switch
git merge <branch>         # Merge branch
git branch -d <branch>     # Delete branch

# History & Inspection
git status                 # Show working tree status
git log --oneline         # Show commit history
git diff                  # Show unstaged changes
git diff --staged         # Show staged changes
git show <commit>         # Show specific commit

# Undoing Changes
git reset HEAD~1          # Undo last commit (keep changes)
git reset --hard HEAD~1   # Undo last commit (discard changes)
git checkout -- <file>    # Discard file changes
git revert <commit>       # Create revert commit
git stash                 # Temporarily store changes
git stash pop            # Apply stashed changes

# Advanced
git rebase <branch>       # Rebase current branch
git cherry-pick <commit>  # Apply specific commit
git bisect start          # Start binary search
git reflog               # Show reference log
```
</div>

---

## Docker Commands

<div class="reference-card" markdown="1">

#### Docker Command Reference

```bash
# Container Management
docker run <image>                    # Run container
docker run -d <image>                # Run detached
docker run -it <image> /bin/bash     # Interactive shell
docker ps                            # List running containers
docker ps -a                         # List all containers
docker stop <container>              # Stop container
docker start <container>             # Start container
docker rm <container>                # Remove container
docker logs <container>              # View logs
docker exec -it <container> /bin/bash # Enter container

# Image Management
docker images                        # List images
docker pull <image>                  # Pull image
docker build -t <tag> .             # Build image
docker push <image>                 # Push image
docker rmi <image>                  # Remove image
docker tag <source> <target>        # Tag image

# Docker Compose
docker-compose up                    # Start services
docker-compose up -d                # Start detached
docker-compose down                 # Stop services
docker-compose ps                   # List services
docker-compose logs -f              # Follow logs
docker-compose exec <service> bash  # Enter service

# System Management
docker system prune                 # Clean unused resources
docker volume ls                    # List volumes
docker network ls                   # List networks
docker inspect <object>             # Inspect object
docker stats                        # Show resource usage
```
</div>

---

## Kubernetes Commands

<div class="reference-card" markdown="1">

#### kubectl Quick Reference

```bash
# Cluster Information
kubectl cluster-info              # Display cluster info
kubectl config view              # View config
kubectl get nodes                # List nodes
kubectl describe node <name>     # Node details

# Resource Management
kubectl get pods                 # List pods
kubectl get svc                  # List services
kubectl get deployments         # List deployments
kubectl get all                 # List all resources
kubectl get all -A              # All namespaces

# Creating Resources
kubectl create -f <file>        # Create from file
kubectl apply -f <file>         # Apply configuration
kubectl create deployment <name> --image=<image>
kubectl expose deployment <name> --port=<port>

# Debugging
kubectl logs <pod>              # View pod logs
kubectl logs -f <pod>           # Follow logs
kubectl exec -it <pod> -- /bin/bash  # Enter pod
kubectl describe pod <pod>      # Pod details
kubectl get events              # Cluster events

# Scaling & Updates
kubectl scale deployment <name> --replicas=<n>
kubectl set image deployment/<name> <container>=<image>
kubectl rollout status deployment/<name>
kubectl rollout undo deployment/<name>

# Deletion
kubectl delete pod <pod>        # Delete pod
kubectl delete -f <file>        # Delete from file
kubectl delete deployment <name> # Delete deployment
```
</div>

---

## AWS CLI Commands

<div class="reference-card" markdown="1">

#### AWS CLI Essential Commands

```bash
# S3 Operations
aws s3 ls                           # List buckets
aws s3 ls s3://bucket              # List objects
aws s3 cp file s3://bucket/        # Upload file
aws s3 cp s3://bucket/file .      # Download file
aws s3 sync . s3://bucket/         # Sync directory
aws s3 rm s3://bucket/file         # Delete file
aws s3 mb s3://bucket              # Make bucket
aws s3 rb s3://bucket              # Remove bucket

# EC2 Operations
aws ec2 describe-instances          # List instances
aws ec2 start-instances --instance-ids <id>
aws ec2 stop-instances --instance-ids <id>
aws ec2 terminate-instances --instance-ids <id>
aws ec2 describe-images --owners self
aws ec2 create-snapshot --volume-id <id>

# IAM Operations
aws iam list-users                  # List users
aws iam list-roles                  # List roles
aws iam list-policies               # List policies
aws iam get-user --user-name <name>
aws iam create-user --user-name <name>

# Lambda Operations
aws lambda list-functions           # List functions
aws lambda invoke --function-name <name> output.json
aws lambda update-function-code --function-name <name> --zip-file fileb://function.zip

# CloudFormation
aws cloudformation list-stacks
aws cloudformation create-stack --stack-name <name> --template-body file://template.yaml
aws cloudformation update-stack --stack-name <name> --template-body file://template.yaml
aws cloudformation delete-stack --stack-name <name>
```
</div>

---

## Terraform CLI Commands

<div class="reference-card" markdown="1">

#### Terraform Essential Commands

```bash
# Initialization & Setup
terraform init                  # Initialize working directory
terraform init -upgrade         # Update provider plugins
terraform version              # Show Terraform version

# Planning & Preview
terraform plan                 # Preview changes
terraform plan -out=tfplan     # Save plan to file
terraform plan -var="key=value" # Plan with variable
terraform plan -target=resource.name # Plan specific resource

# Apply Changes
terraform apply                # Apply changes (with prompt)
terraform apply -auto-approve  # Apply without confirmation
terraform apply tfplan         # Apply saved plan
terraform apply -var-file="vars.tfvars" # Apply with variable file

# Validation & Formatting
terraform validate             # Validate configuration
terraform fmt                  # Format configuration files
terraform fmt -check           # Check if formatting needed
terraform fmt -recursive       # Format all subdirectories

# State Management
terraform state list           # List resources in state
terraform state show <resource> # Show resource details
terraform state mv <src> <dst> # Move resource in state
terraform state rm <resource>  # Remove resource from state
terraform state pull          # Download remote state
terraform state push          # Upload local state

# Workspace Management
terraform workspace list       # List workspaces
terraform workspace new <name> # Create workspace
terraform workspace select <name> # Switch workspace
terraform workspace show      # Show current workspace
terraform workspace delete <name> # Delete workspace

# Import & Output
terraform import <resource> <id> # Import existing resource
terraform output              # Show all outputs
terraform output <name>       # Show specific output
terraform output -json        # Output as JSON

# Destruction
terraform destroy             # Destroy all resources
terraform destroy -target=resource.name # Destroy specific resource
terraform destroy -auto-approve # Destroy without confirmation

# Debugging & Troubleshooting
terraform show                # Show current state
terraform graph               # Generate dependency graph
terraform console             # Interactive console
terraform providers           # Show provider dependencies

# Advanced Operations
terraform taint <resource>    # Mark resource for recreation
terraform untaint <resource>  # Remove taint mark
terraform refresh             # Update state from real infrastructure
terraform force-unlock <lock-id> # Force unlock state

# Environment Variables
export TF_LOG=DEBUG           # Enable debug logging
export TF_LOG_PATH=terraform.log # Set log file path
export TF_VAR_name=value      # Set variable via environment
```
</div>

---

## Physics Formulas & Constants

### Fundamental Constants

<div class="reference-card" markdown="1">

#### Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Speed of light | $c$ | $2.998 \times 10^{8}$ | m/s |
| Planck constant | $h$ | $6.626 \times 10^{-34}$ | J·s |
| Reduced Planck constant | $\hbar$ | $1.055 \times 10^{-34}$ | J·s |
| Gravitational constant | $G$ | $6.674 \times 10^{-11}$ | N·m²/kg² |
| Elementary charge | $e$ | $1.602 \times 10^{-19}$ | C |
| Electron mass | $m_e$ | $9.109 \times 10^{-31}$ | kg |
| Proton mass | $m_p$ | $1.673 \times 10^{-27}$ | kg |
| Boltzmann constant | $k_B$ | $1.381 \times 10^{-23}$ | J/K |
| Avogadro's number | $N_A$ | $6.022 \times 10^{23}$ | mol⁻¹ |
| Fine structure constant | $\alpha$ | $1/137.036$ | dimensionless |
| Vacuum permittivity | $\varepsilon_0$ | $8.854 \times 10^{-12}$ | F/m |
| Vacuum permeability | $\mu_0$ | $4\pi \times 10^{-7}$ | H/m |
</div>

### Key Physics Equations

<div class="reference-card" markdown="1">

#### Classical Mechanics

| Quantity | Equation |
|----------|----------|
| Newton's second law | $\vec{F} = m\vec{a}$ |
| Newton's third law | $\vec{F}_{12} = -\vec{F}_{21}$ |
| Velocity (constant $a$) | $v = v_0 + at$ |
| Position (constant $a$) | $x = x_0 + v_0 t + \tfrac{1}{2}at^2$ |
| Velocity–position | $v^2 = v_0^2 + 2a(x - x_0)$ |
| Kinetic energy | $KE = \tfrac{1}{2}mv^2$ |
| Gravitational PE | $PE = mgh$ |
| Spring PE | $PE = \tfrac{1}{2}kx^2$ |
| Work | $W = \vec{F}\cdot\vec{d} = Fd\cos\theta$ |
| Linear momentum | $\vec{p} = m\vec{v}$ |
| Angular momentum | $\vec{L} = \vec{r}\times\vec{p}$ |
| Torque | $\vec{\tau} = \vec{r}\times\vec{F}$ |
</div>

<div class="reference-card" markdown="1">

#### Quantum Mechanics

**Fundamental equations**

$$i\hbar\,\frac{\partial \psi}{\partial t} = \hat{H}\psi \quad\text{(Schrödinger equation)}$$

$$\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V \quad\text{(Hamiltonian)}$$

$$[\hat{x}, \hat{p}] = i\hbar \quad\text{(canonical commutation)}$$

$$\Delta x\,\Delta p \geq \frac{\hbar}{2} \quad\text{(uncertainty principle)}$$

**Quantum states**

$$|\psi\rangle = \sum_i c_i |i\rangle, \qquad \langle\psi|\psi\rangle = 1, \qquad P = |\langle\phi|\psi\rangle|^2$$

**Hydrogen atom**

$$E_n = -\frac{13.6\ \text{eV}}{n^2}, \qquad r_n = n^2 a_0, \qquad a_0 = 0.529\ \text{Å}$$
</div>

<div class="reference-card" markdown="1">

#### Electromagnetism

**Maxwell's equations**

$$\nabla\cdot\vec{E} = \frac{\rho}{\varepsilon_0} \quad\text{(Gauss's law)}$$

$$\nabla\cdot\vec{B} = 0 \quad\text{(no magnetic monopoles)}$$

$$\nabla\times\vec{E} = -\frac{\partial \vec{B}}{\partial t} \quad\text{(Faraday's law)}$$

$$\nabla\times\vec{B} = \mu_0\!\left(\vec{J} + \varepsilon_0\frac{\partial \vec{E}}{\partial t}\right) \quad\text{(Ampère–Maxwell law)}$$

**Field relations**

$$\vec{F} = q(\vec{E} + \vec{v}\times\vec{B}), \qquad \vec{E} = -\nabla\phi - \frac{\partial \vec{A}}{\partial t}, \qquad \vec{B} = \nabla\times\vec{A}, \qquad c = \frac{1}{\sqrt{\mu_0\varepsilon_0}}$$

**Wave equation**

$$\nabla^2\vec{E} - \frac{1}{c^2}\frac{\partial^2 \vec{E}}{\partial t^2} = 0$$
</div>

## Algorithms & Data Structures

### Big O Complexity Reference

<div class="reference-card" markdown="1">


#### Time Complexity Cheat Sheet

**Sorting Algorithms**

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |

**Searching Algorithms**

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Linear Search | O(1) | O(n) | O(n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |

**Data Structures**

| Structure | Access | Insert/Delete | Search | Space |
|-----------|--------|---------------|--------|-------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(1) | O(n) | O(n) |
| Hash Table | O(1) | O(1) | O(1) avg | O(n) |
| Binary Tree | O(log n) | O(log n) | O(log n) | O(n) |
| B-Tree | O(log n) | O(log n) | O(log n) | O(n) |

</div>

### Common Algorithm Patterns

<div class="reference-card" markdown="1">

#### Algorithm Templates

```python
# Two Pointers
def two_pointers(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Process
        if condition:
            left += 1
        else:
            right -= 1

# Sliding Window
def sliding_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# DFS (Recursive)
def dfs(node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in node.neighbors:
        if neighbor not in visited:
            dfs(neighbor, visited)

# BFS (Iterative)
from collections import deque
def bfs(start):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        # Process node
        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Dynamic Programming
def dp_fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```
</div>

## API Reference Patterns

### RESTful API Conventions

<div class="reference-card" markdown="1">

#### REST API Best Practices

```
# Resource Naming
GET    /users              # List all users
GET    /users/{id}         # Get specific user
POST   /users              # Create new user
PUT    /users/{id}         # Update user (full)
PATCH  /users/{id}         # Update user (partial)
DELETE /users/{id}         # Delete user

# Nested Resources
GET    /users/{id}/posts   # User's posts
POST   /users/{id}/posts   # Create post for user

# Query Parameters
GET    /users?page=2&limit=20         # Pagination
GET    /users?sort=name&order=asc     # Sorting
GET    /users?filter[status]=active   # Filtering
GET    /users?fields=id,name,email    # Field selection

# HTTP Status Codes
200 OK                     # Successful GET, PUT
201 Created               # Successful POST
204 No Content            # Successful DELETE
400 Bad Request           # Invalid request
401 Unauthorized          # Authentication required
403 Forbidden             # No permission
404 Not Found             # Resource not found
409 Conflict              # Resource conflict
500 Internal Server Error # Server error

# Headers
Content-Type: application/json
Authorization: Bearer <token>
Accept: application/json
X-API-Version: v1
X-Request-ID: <uuid>
```
</div>

### Common API Response Formats

<div class="reference-card" markdown="1">

#### API Response Patterns

```json
// Successful Response
{
  "status": "success",
  "data": {
    "id": 123,
    "name": "John Doe",
    "email": "john@example.com"
  },
  "meta": {
    "timestamp": "2024-01-01T00:00:00Z"
  }
}

// Error Response
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "details": {
      "field": "email",
      "value": "invalid-email"
    }
  }
}

// Paginated Response
{
  "status": "success",
  "data": [...],
  "pagination": {
    "page": 2,
    "per_page": 20,
    "total": 100,
    "total_pages": 5,
    "links": {
      "first": "/users?page=1",
      "prev": "/users?page=1",
      "next": "/users?page=3",
      "last": "/users?page=5"
    }
  }
}
```
</div>

## Troubleshooting Flowcharts

### Docker Troubleshooting

<div class="reference-card" markdown="1">

#### Docker Issue Resolution

```
Container Won't Start?
├─ Check logs: docker logs <container>
├─ Verify image: docker images
├─ Check resources: docker system df
└─ Inspect container: docker inspect <container>

Build Fails?
├─ Check Dockerfile syntax
├─ Verify base image exists
├─ Check build context size
└─ Review build cache: docker build --no-cache

Network Issues?
├─ List networks: docker network ls
├─ Inspect network: docker network inspect <network>
├─ Check container network: docker inspect <container> | grep Network
└─ Test connectivity: docker exec <container> ping <target>

Permission Errors?
├─ Check user in Dockerfile
├─ Verify volume permissions
├─ Use --user flag in docker run
└─ Check SELinux/AppArmor settings
```
</div>

### Git Troubleshooting

<div class="reference-card" markdown="1">

#### Common Git Issues

```
Merge Conflicts?
├─ Identify conflicts: git status
├─ Open conflicted files
├─ Resolve conflicts manually
├─ Stage resolved files: git add <file>
└─ Complete merge: git commit

Accidentally Committed?
├─ Undo last commit: git reset HEAD~1
├─ Keep changes staged: git reset --soft HEAD~1
├─ Discard changes: git reset --hard HEAD~1
└─ Revert public commit: git revert <commit>

Wrong Branch?
├─ Stash changes: git stash
├─ Switch branch: git checkout <correct-branch>
├─ Apply changes: git stash pop
└─ Or cherry-pick: git cherry-pick <commit>

Lost Commits?
├─ Check reflog: git reflog
├─ Find lost commit
├─ Restore: git checkout <commit-hash>
└─ Create branch: git checkout -b recovered-branch
```
</div>

## Best Practices Checklists

### Code Review Checklist

<div class="reference-card" markdown="1">

#### Code Review Guidelines

- [ ] **Functionality**
  - [ ] Code accomplishes intended purpose
  - [ ] Edge cases handled
  - [ ] Error handling implemented
  - [ ] No obvious bugs

- [ ] **Code Quality**
  - [ ] Clear variable/function names
  - [ ] DRY principle followed
  - [ ] SOLID principles applied
  - [ ] Appropriate abstractions

- [ ] **Testing**
  - [ ] Unit tests included
  - [ ] Tests cover edge cases
  - [ ] Integration tests if needed
  - [ ] Tests are maintainable

- [ ] **Performance**
  - [ ] No obvious inefficiencies
  - [ ] Appropriate data structures
  - [ ] Database queries optimized
  - [ ] Caching implemented where needed

- [ ] **Security**
  - [ ] Input validation
  - [ ] No hardcoded secrets
  - [ ] SQL injection prevention
  - [ ] XSS prevention

- [ ] **Documentation**
  - [ ] Complex logic documented
  - [ ] API documentation updated
  - [ ] README updated if needed
  - [ ] Inline comments where helpful
</div>

### Deployment Checklist

<div class="reference-card" markdown="1">

#### Pre-Deployment Verification

- [ ] **Code Preparation**
  - [ ] All tests passing
  - [ ] Code reviewed and approved
  - [ ] Version bumped
  - [ ] Changelog updated

- [ ] **Environment Check**
  - [ ] Environment variables set
  - [ ] Secrets configured
  - [ ] Dependencies updated
  - [ ] Database migrations ready

- [ ] **Monitoring Setup**
  - [ ] Logging configured
  - [ ] Alerts configured
  - [ ] Health checks enabled
  - [ ] Metrics collection setup

- [ ] **Rollback Plan**
  - [ ] Previous version tagged
  - [ ] Rollback procedure documented
  - [ ] Database rollback plan
  - [ ] Communication plan ready

- [ ] **Post-Deployment**
  - [ ] Smoke tests executed
  - [ ] Monitoring dashboards checked
  - [ ] Performance validated
  - [ ] Stakeholders notified
</div>

## Mathematical Reference

### Calculus Formulas

<div class="reference-card" markdown="1">

#### Derivatives and Integrals

**Common derivatives**

$$\frac{d}{dx}x^n = nx^{n-1}, \quad \frac{d}{dx}e^x = e^x, \quad \frac{d}{dx}\ln x = \frac{1}{x}$$

$$\frac{d}{dx}\sin x = \cos x, \quad \frac{d}{dx}\cos x = -\sin x, \quad \frac{d}{dx}\tan x = \sec^2 x$$

**Differentiation rules**

$$\underbrace{(uv)' = u'v + uv'}_{\text{product}}, \quad \underbrace{\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x)}_{\text{chain}}, \quad \underbrace{\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}}_{\text{quotient}}$$

**Common integrals**

$$\int x^n\,dx = \frac{x^{n+1}}{n+1} + C \;\; (n \neq -1), \quad \int \frac{1}{x}\,dx = \ln|x| + C, \quad \int e^x\,dx = e^x + C$$

$$\int \sin x\,dx = -\cos x + C, \quad \int \cos x\,dx = \sin x + C, \quad \int u\,dv = uv - \int v\,du$$
</div>

### Linear Algebra

<div class="reference-card" markdown="1">

#### Matrix Operations

**Multiplication, determinants, and invariants**

$$(AB)_{ij} = \sum_k A_{ik}B_{kj}$$

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

$$\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

$$\det(A - \lambda I) = 0 \;\;(\text{eigenvalues}), \qquad \operatorname{tr}(A) = \sum_i A_{ii}, \qquad (A^\mathsf{T})_{ij} = A_{ji}$$

**Special matrices**

$$\underbrace{I_{ij} = \delta_{ij}}_{\text{identity}}, \quad \underbrace{Q^\mathsf{T}Q = QQ^\mathsf{T} = I}_{\text{orthogonal}}, \quad \underbrace{A^\dagger = A}_{\text{Hermitian}}, \quad \underbrace{U^\dagger U = UU^\dagger = I}_{\text{unitary}}$$
</div>

## Network Protocols

<div class="reference-card" markdown="1">

#### Common Port Numbers

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| SSH | 22 | TCP | Secure Shell |
| Telnet | 23 | TCP | Unencrypted remote access |
| SMTP | 25 | TCP | Email sending |
| DNS | 53 | TCP/UDP | Domain name resolution |
| HTTP | 80 | TCP | Web traffic |
| HTTPS | 443 | TCP | Secure web traffic |
| FTP | 20-21 | TCP | File transfer |
| MySQL | 3306 | TCP | MySQL database |
| PostgreSQL | 5432 | TCP | PostgreSQL database |
| Redis | 6379 | TCP | Redis cache |
| MongoDB | 27017 | TCP | MongoDB database |
| Elasticsearch | 9200 | TCP | Elasticsearch API |
| Kubernetes API | 6443 | TCP | K8s API server |
</div>

## Regular Expressions

<div class="reference-card" markdown="1">

#### Regex Quick Reference

```
# Character Classes
.        Any character except newline
\d       Digit (0-9)
\D       Non-digit
\w       Word character (a-z, A-Z, 0-9, _)
\W       Non-word character
\s       Whitespace
\S       Non-whitespace
[abc]    Character set
[^abc]   Negated set
[a-z]    Range

# Quantifiers
*        0 or more
+        1 or more
?        0 or 1
{n}      Exactly n
{n,}     n or more
{n,m}    Between n and m

# Anchors
^        Start of string
$        End of string
\b       Word boundary
\B       Non-word boundary

# Groups
(...)    Capturing group
(?:...)  Non-capturing group
(?=...)  Positive lookahead
(?!...)  Negative lookahead

# Common Patterns
Email: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
URL: https?://[^\s]+
IP: \b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b
Phone: ^\+?1?\d{9,15}$
```
</div>

## Quick Links

### Documentation Resources
- [Git Documentation](https://git-scm.com/doc)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [Python Documentation](https://docs.python.org/3/)
- [MDN Web Docs](https://developer.mozilla.org/)

### Online Tools
- [Regex101](https://regex101.com/) - Regex testing
- [JWT.io](https://jwt.io/) - JWT decoder
- [Base64 Encode/Decode](https://www.base64encode.org/)
- [JSON Formatter](https://jsonformatter.curiousconcept.com/)
- [Crontab Guru](https://crontab.guru/) - Cron expression helper
- [YAML Validator](https://www.yamllint.com/)

### Performance Tools
- [GTmetrix](https://gtmetrix.com/) - Web performance
- [WebPageTest](https://www.webpagetest.org/) - Performance testing
- [Can I Use](https://caniuse.com/) - Browser compatibility
- [Bundle Phobia](https://bundlephobia.com/) - NPM package size

---

<div class="reference-footer">
<h2>Contributing to This Reference</h2>
<p>This reference guide is continuously updated. Found an error or have a suggestion? <a href="https://github.com/AndrewAltimit/Documentation">Contribute on GitHub</a>.</p>
<p>For detailed explanations and tutorials, explore the main documentation sections:</p>
<p class="footer-links">
<a href="../#technology">Technology</a> ·
<a href="../#physics">Physics</a> ·
<a href="../ai-ml/">AI/ML</a> ·
<a href="../artificial-intelligence/">AI Hub</a> ·
<a href="../quantum-computing/">Quantum Computing</a> ·
<a href="../distributed-systems/">Distributed Systems</a>
</p>
</div>

## Related References

- **[Git Command Reference](../technology/git-reference.html)** - Comprehensive Git guide
- **[Docker Essentials](../technology/docker-essentials.html)** - Complete Docker reference
- **[Terraform Documentation](../technology/terraform/)** - Infrastructure as Code guide
- **[AI/ML Model Reference](../ai-ml/model-types.html)** - Model architectures explained
- **[Advanced Mathematics](../advanced/ai-mathematics/)** - Graduate-level formulas

---

*Last updated: 2025 | Quick tip: Use Ctrl+F (Cmd+F on Mac) to search this page instantly*