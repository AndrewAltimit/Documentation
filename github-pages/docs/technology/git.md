---
layout: docs
title: Git Version Control
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
difficulty_level: advanced
section: technology
---

{% include learning-breadcrumb.html 
   path=site.data.breadcrumbs.technology 
   current="Git Version Control"
   alternatives=site.data.alternatives.git_advanced 
%}

{% include skill-level-navigation.html 
   current_level="advanced"
   topic="Git"
   beginner_link="/docs/technology/git-crash-course/"
   intermediate_link="/docs/technology/branching/"
%}

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">Git Version Control</h1>
    <p class="hero-subtitle">A Comprehensive Guide to Distributed Version Control with Git</p>
  </div>
</div>

{% include difficulty-helper.html 
   current_level="advanced"
   easier_link="/docs/technology/branching/"
   prerequisites=site.data.prerequisites.git_advanced
   related_topics=site.data.related_topics.git_advanced
%}

<div class="intro-card">
  <div class="notice--info">
    <h4>📚 Comprehensive Git Guide</h4>
    <p>This page provides a complete Git reference from basics to advanced internals. Whether you're just starting or looking to master Git's inner workings, you'll find valuable content here. Use the table of contents to jump to your level of expertise.</p>
  </div>
  
  <p class="lead-text">Git is a distributed version control system that tracks changes in your code over time. Unlike traditional version control systems that rely on a central server, Git gives every developer a complete copy of the project history. This revolutionary approach, combined with its elegant design based on cryptographic principles, has made Git the de facto standard for modern software development.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-project-diagram"></i>
      <h4>DAG-Based History</h4>
      <p>Directed acyclic graph for commits</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-shield-alt"></i>
      <h4>Cryptographic Integrity</h4>
      <p>SHA-1/SHA-256 content addressing (SHA-256 recommended)</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-network-wired"></i>
      <h4>Distributed Architecture</h4>
      <p>Peer-to-peer repository model</p>
    </div>
  </div>
</div>

## What is Git?

Git is a **distributed version control system** created by Linus Torvalds in 2005 for Linux kernel development. Unlike centralized systems (SVN, Perforce), Git:

- **Stores complete history locally**: Every clone is a full backup
- **Works offline**: Most operations don't need network access
- **Branches are lightweight**: Creating/merging branches is fast and easy
- **Guarantees data integrity**: Uses SHA-1/SHA-256 checksums for all data
- **Supports non-linear development**: Multiple parallel branches and complex merges

### Why Use Version Control?

Version control solves fundamental problems in software development:

1. **Collaboration**: Multiple developers can work on the same project without conflicts
2. **History**: Track who changed what, when, and why
3. **Backup**: Distributed copies protect against data loss
4. **Experimentation**: Try new ideas in branches without affecting stable code
5. **Time Travel**: Revert to any previous state of the project
6. **Blame/Annotation**: Understand why code was written a certain way

## Git Crash Course: From Zero to Productive

This section gets you productive with Git in minutes. We'll cover the essential commands you need for daily work, then gradually introduce more advanced concepts.

### Installing Git

**macOS:**
```bash
# Using Homebrew
brew install git

# Or download from git-scm.com
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install git

# Fedora
sudo dnf install git

# Arch
sudo pacman -S git
```

**Windows:**
- Download Git for Windows from [git-scm.com](https://git-scm.com)
- Or use WSL2 with Linux instructions

### First-Time Setup

```bash
# Configure your identity (required)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set your preferred editor
git config --global core.editor "code --wait"  # VS Code
# or
git config --global core.editor "vim"          # Vim
# or  
git config --global core.editor "nano"         # Nano

# Improve command output
git config --global color.ui auto

# Set default branch name for new repos
git config --global init.defaultBranch main

# Helpful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
```

### Your First Repository

**Starting a New Project:**
```bash
# Create a new directory
mkdir my-project
cd my-project

# Initialize Git repository
git init

# Create your first file
echo "# My Project" > README.md

# Check repository status
git status
# Output: Shows README.md as untracked

# Add file to staging area
git add README.md

# Commit with a message
git commit -m "Initial commit: Add README"

# View commit history
git log
```

**Cloning an Existing Project:**
```bash
# Clone via HTTPS
git clone https://github.com/username/repository.git

# Clone via SSH (requires SSH key setup)
git clone git@github.com:username/repository.git

# Clone into specific directory
git clone https://github.com/username/repository.git my-local-name
```

### The Basic Workflow

Git has three main states for your files:

1. **Working Directory**: Where you edit files
2. **Staging Area (Index)**: Where you prepare commits
3. **Repository**: Where Git stores committed snapshots

```bash
# 1. Make changes to files
echo "New feature" >> feature.txt

# 2. Check what changed
git status
git diff              # Shows unstaged changes

# 3. Stage changes
git add feature.txt   # Stage specific file
git add .            # Stage all changes
git add -p           # Stage interactively (choose hunks)

# 4. Review staged changes
git diff --staged    # Shows what will be committed

# 5. Commit
git commit -m "Add new feature"

# Or open editor for detailed message
git commit
```

### Working with Branches

Branches are Git's killer feature. They let you work on features in isolation:

```bash
# List branches
git branch              # Local branches
git branch -r          # Remote branches  
git branch -a          # All branches

# Create new branch
git branch feature/login

# Switch to branch
git checkout feature/login
# Or create and switch in one command
git checkout -b feature/login

# Make changes and commit
echo "Login form" > login.html
git add login.html
git commit -m "Add login form"

# Switch back to main
git checkout main

# Merge feature branch
git merge feature/login

# Delete merged branch
git branch -d feature/login
```

### Remote Repositories

Git is distributed - you collaborate by syncing with remote repositories:

```bash
# Add remote repository
git remote add origin https://github.com/username/repo.git

# View remotes
git remote -v

# Push to remote
git push origin main
# Or set upstream and just use 'git push'
git push -u origin main

# Fetch changes from remote
git fetch origin

# Pull (fetch + merge) changes
git pull origin main

# Push new branch
git push -u origin feature/login
```

### Common Scenarios and Solutions

**Scenario 1: Undo Changes**
```bash
# Discard changes in working directory
git checkout -- file.txt           # Single file
git checkout -- .                 # All files

# Unstage files (keep changes)
git reset HEAD file.txt

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes) 
git reset --hard HEAD~1           # DANGEROUS!
```

**Scenario 2: Fix Commit Message**
```bash
# Change last commit message
git commit --amend -m "Better message"

# Add forgotten file to last commit
git add forgotten.txt
git commit --amend --no-edit
```

**Scenario 3: Save Work Temporarily**
```bash
# Stash current changes
git stash

# Do other work, then restore
git stash pop

# List all stashes
git stash list

# Apply specific stash
git stash apply stash@{2}
```

**Scenario 4: Resolve Merge Conflicts**
```bash
# After merge conflict
git status                    # See conflicted files

# Edit files to resolve conflicts
# Look for markers:
# <<<<<<< HEAD
# Your changes
# =======  
# Their changes
# >>>>>>> branch-name

# After resolving
git add resolved-file.txt
git commit                   # Completes the merge
```

### Quick Reference Card

| Task | Command |
|------|--------|
| Initialize repo | `git init` |
| Clone repo | `git clone <url>` |
| Check status | `git status` |
| View changes | `git diff` |
| Stage files | `git add <file>` or `git add .` |
| Commit | `git commit -m "message"` |
| View history | `git log --oneline` |
| Create branch | `git checkout -b <branch>` |
| Switch branch | `git checkout <branch>` |
| Merge branch | `git merge <branch>` |
| Push changes | `git push` |
| Pull changes | `git pull` |
| Stash changes | `git stash` |
| Undo changes | `git checkout -- <file>` |

## Understanding Git's Architecture

Now that you can use Git effectively, let's understand how it works under the hood. This knowledge helps you troubleshoot problems, optimize workflows, and truly master Git.

## Common Pitfalls and How to Avoid Them

### 1. Committing Large Binary Files

**Problem**: Git stores complete copies of binary files, bloating repository size.

**Solution**: 
```bash
# Use Git LFS for large files
git lfs track "*.psd"
git lfs track "*.zip" 
git add .gitattributes
git commit -m "Configure Git LFS"

# Or add to .gitignore
echo "*.psd" >> .gitignore
echo "build/" >> .gitignore
```

### 2. Committing Sensitive Data

**Problem**: Accidentally committed passwords, API keys, or secrets.

**Solution**:
```bash
# If not pushed yet
git reset --soft HEAD~1
# Remove sensitive file
git rm --cached sensitive.txt

# If already pushed (requires rewriting history)
# Use BFG Repo-Cleaner
java -jar bfg.jar --delete-files passwords.txt
git push --force

# Prevention: Use .gitignore
echo ".env" >> .gitignore
echo "config/secrets.yml" >> .gitignore
```

### 3. Merge Conflicts from Outdated Branches

**Problem**: Long-lived feature branches accumulate conflicts.

**Solution**:
```bash
# Regularly sync with main branch
git checkout feature/long-running
git fetch origin
git rebase origin/main  # Or merge if you prefer

# Alternative: Merge main into feature periodically
git merge origin/main
```

### 4. Lost Commits After Reset

**Problem**: Accidentally reset --hard and lost commits.

**Solution**:
```bash
# Git keeps a reflog of all HEAD movements
git reflog
# Find your lost commit SHA
# Example output:
# abc123 HEAD@{0}: reset: moving to HEAD~3
# def456 HEAD@{1}: commit: Important work

# Recover the commit
git checkout def456
# Or create branch from it
git checkout -b recovered-work def456
```

### 5. Messy Commit History

**Problem**: Many small, unclear commits make history hard to follow.

**Solution**:
```bash
# Before merging, clean up with interactive rebase
git rebase -i origin/main

# In the editor:
# pick abc123 WIP
# squash def456 Fix typo
# squash ghi789 More fixes
# reword jkl012 Add login feature

# Result: Clean, logical commits
```

### 6. Working on Wrong Branch

**Problem**: Made commits on main instead of feature branch.

**Solution**:
```bash
# Create new branch with current changes
git branch feature/new-work

# Reset main to origin
git reset --hard origin/main

# Switch to feature branch
git checkout feature/new-work
```

### 7. Pushing to Wrong Remote

**Problem**: Accidentally pushed to upstream instead of fork.

**Solution**:
```bash
# Set up remotes clearly
git remote add upstream https://github.com/original/repo.git
git remote add origin https://github.com/yourfork/repo.git

# Always verify before pushing
git remote -v
git push origin feature-branch  # Explicitly specify remote
```

### 8. File Permission Changes

**Problem**: Git tracks file permissions, causing unnecessary diffs.

**Solution**:
```bash
# Ignore file mode changes
git config core.fileMode false

# Or globally
git config --global core.fileMode false
```

### 9. Line Ending Issues

**Problem**: CRLF/LF differences between Windows and Unix systems.

**Solution**:
```bash
# Configure line endings
# Windows:
git config --global core.autocrlf true

# Mac/Linux:
git config --global core.autocrlf input

# Per-repository settings in .gitattributes:
echo "* text=auto" >> .gitattributes
echo "*.sh text eol=lf" >> .gitattributes
echo "*.bat text eol=crlf" >> .gitattributes
```

### 10. Detached HEAD State

**Problem**: Accidentally working in detached HEAD state.

**Solution**:
```bash
# If you made commits in detached HEAD
# Create a branch to save them
git branch save-detached-work

# Or if you want to discard
git checkout main

# To avoid: Always work on branches
git checkout -b new-feature
# Instead of
git checkout <commit-sha>
```

## Mathematical Foundations

### Content-Addressable Storage Theory

Git implements a content-addressable filesystem where objects are stored and retrieved by their SHA-1 hash. This provides cryptographic integrity and enables efficient storage through deduplication.

**Key Components:**
- **GitObject**: Base class for all Git objects (blob, tree, commit, tag)
- **SHA-1 Hashing**: Content addressing using cryptographic hashes
  - **⚠️ Security Notice**: SHA-1 is cryptographically broken and deprecated. While Git still uses SHA-1 by default for backward compatibility, it's recommended to migrate to SHA-256 for new repositories using `git init --object-format=sha256`
- **Compression**: zlib compression for efficient storage
- **Merkle Trees**: Tree objects form a Merkle tree structure
- **Commit DAG**: Directed Acyclic Graph for version history

**Object Model:**
```python
# Example: Creating a Git object
header = f"blob {len(content)}\0".encode()
sha1 = hashlib.sha1(header + content).hexdigest()
compressed = zlib.compress(header + content)
```

> **Code Reference**: For complete implementation of Git's object model, Merkle trees, and DAG operations, see [`content_addressable_storage.py`](../../code-examples/technology/git/content_addressable_storage.py)

### Three-Way Merge Algorithm

Git's three-way merge algorithm combines changes from two branches using their common ancestor as a reference point. This enables automatic resolution of non-conflicting changes.

**Algorithm Steps:**
1. Find common ancestor (merge base)
2. Compute diffs: base→ours and base→theirs
3. Apply non-conflicting changes automatically
4. Mark conflicting changes for manual resolution

**Merge Cases:**
- **No conflict**: Changes in different files or parts
- **Auto-merge**: Same changes in both branches
- **Conflict**: Different changes to same lines

**Example merge conflict markers:**
```
<<<<<<< ours
Our changes
=======
Their changes
>>>>>>> theirs
```

> **Code Reference**: For complete three-way merge implementation with conflict detection and advanced merge strategies, see [`three_way_merge.py`](../../code-examples/technology/git/three_way_merge.py)

## Git Internals

### Object Storage Implementation

Git uses a content-addressable storage system where objects are identified by their SHA-1 hash. The storage system includes:

**Key Components:**
- **Loose Objects**: Individual compressed files stored in `.git/objects/`
- **Pack Files**: Compressed archives containing multiple objects with delta compression
- **Pack Indexes**: Binary indexes for fast object lookup within pack files
- **Bitmap Indexes**: Reachability bitmaps for optimized operations

**Storage Format:**
- Objects are compressed using zlib
- Directory structure uses first 2 characters of SHA-1 for sharding
- Pack files use delta compression to reduce storage size

> **Code Reference**: For complete object storage implementation including loose objects, pack files, and index structures, see [`object_storage.py`](../../code-examples/technology/git/object_storage.py)

### Index (Staging Area) Structure

```c
// Git index format
struct index_header {
    char signature[4];      // "DIRC"
    uint32_t version;       // 2, 3, or 4
    uint32_t entries;       // Number of entries
};

struct index_entry {
    struct stat_data {
        uint32_t ctime_sec;
        uint32_t ctime_nsec;
        uint32_t mtime_sec;
        uint32_t mtime_nsec;
        uint32_t dev;
        uint32_t ino;
        uint32_t mode;
        uint32_t uid;
        uint32_t gid;
        uint32_t size;
    } stat;
    uint8_t sha1[20];
    uint16_t flags;
    char path[]; // Variable length
};

// Extensions
struct index_extension {
    char signature[4];
    uint32_t size;
    // Extension-specific data
};
```

### Reference Management

Git references are human-readable names that point to commit objects. The reference system provides:

**Reference Types:**
- **Branches**: Mutable references to commits (`refs/heads/*`)
- **Tags**: Immutable references to objects (`refs/tags/*`)
- **Remote refs**: References to remote repository state (`refs/remotes/*`)
- **Symbolic refs**: References to other references (like HEAD)

**Storage Mechanisms:**
- **Loose refs**: Individual files containing commit SHA
- **Packed refs**: Single file containing multiple references for efficiency
- **Ref transactions**: Atomic updates to multiple references

**Atomic Updates:**
- Lock files ensure concurrent safety
- Compare-and-swap operations prevent race conditions
- Reflogs track reference history

> **Code Reference**: For complete reference management implementation with atomic updates and transactions, see [`reference_management.py`](../../code-examples/technology/git/reference_management.py)

## Advanced Command Implementation

### Repository Operations at the Protocol Level

Git's network protocol enables efficient distributed repository synchronization through:

**Protocol Phases:**
1. **Reference Discovery**: Client queries available refs from server
2. **Capability Negotiation**: Client and server agree on protocol features
3. **Pack Negotiation**: Client specifies wants/haves for minimal data transfer
4. **Pack Transfer**: Server sends optimized pack file with only needed objects

**Protocol Features:**
- **Smart HTTP Protocol**: Efficient bidirectional communication
- **Pack Protocol**: Optimized object transfer format
- **Shallow Cloning**: Fetch limited history depth
- **Partial Cloning**: Fetch subset of repository objects
- **Protocol Extensions**: Side-band, progress reporting, atomic push

**Optimization Techniques:**
- Delta compression between similar objects
- Bitmap indexes for reachability queries
- Multi-pack indexes for large repositories

> **Code Reference**: For complete Git protocol implementation including clone, fetch, and push operations, see [`repository_operations.py`](../../code-examples/technology/git/repository_operations.py)

### Configuration

```bash
# Set user information
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default editor
git config --global core.editor "vim"

# Set default branch name
git config --global init.defaultBranch main

# List all configurations
git config --list

# Configure line endings
git config --global core.autocrlf true  # Windows
git config --global core.autocrlf input # Mac/Linux
```

### Basic Workflow

```bash
# Check status
git status

# Add files to staging
git add file.txt                # Specific file
git add .                       # All files
git add -p                      # Interactive staging

# Commit changes
git commit -m "Add feature X"
git commit -am "Update file"    # Add and commit tracked files
git commit --amend             # Modify last commit

# View history
git log
git log --oneline
git log --graph --all --decorate
git log --follow file.txt       # File history including renames
```

## Advanced Branching and Merge Strategies

### Merge Algorithm Implementation

Git implements sophisticated merge algorithms to combine divergent development branches:

**Merge Strategies:**
- **Recursive (default)**: Handles multiple merge bases by recursively merging them
- **Octopus**: Merges multiple branches simultaneously (for integration branches)
- **Ours**: Keeps current branch content, ignoring other branches
- **Subtree**: Adjusts for directory structure differences
- **Resolve**: Older two-head merge strategy

**Three-Way Merge Process:**
1. Find common ancestor(s) of branches
2. For multiple bases, recursively merge them to create virtual base
3. Compare base, ours, and theirs for each file
4. Auto-merge non-conflicting changes
5. Mark conflicts for manual resolution

**Conflict Detection:**
- Concurrent modifications to same file regions
- File vs directory conflicts
- Add/add conflicts (different files at same path)
- Rename detection and handling

> **Code Reference**: For complete merge strategy implementations including recursive, octopus, and subtree strategies, see [`merge_strategies.py`](../../code-examples/technology/git/merge_strategies.py)

## Branching and Merging

### Branch Management

```bash
# List branches
git branch                      # Local branches
git branch -r                   # Remote branches
git branch -a                   # All branches

# Create branch
git branch feature/new-feature
git checkout -b feature/new-feature  # Create and switch

# Switch branches
git checkout develop
git switch develop              # Newer syntax

# Delete branch
git branch -d feature/completed  # Safe delete
git branch -D feature/abandoned  # Force delete

# Rename branch
git branch -m old-name new-name
```

### Merging

```bash
# Merge branch
git checkout main
git merge feature/new-feature

# Merge with no fast-forward
git merge --no-ff feature/new-feature

# Abort merge
git merge --abort

# Resolve conflicts
# 1. Edit conflicted files
# 2. Mark as resolved
git add resolved-file.txt
# 3. Complete merge
git commit
```

### Rebasing

```bash
# Rebase current branch onto main
git rebase main

# Interactive rebase
git rebase -i HEAD~3

# Common interactive commands:
# pick   - use commit
# reword - change commit message
# squash - combine with previous
# fixup  - combine, discard message
# drop   - remove commit

# Abort rebase
git rebase --abort

# Continue after resolving conflicts
git rebase --continue
```

## Distributed Repository Synchronization

### Remote Protocol Implementation

Git's distributed nature relies on efficient protocols for synchronizing repositories:

**Protocol Capabilities:**
- **multi_ack_detailed**: Optimized negotiation with multiple acknowledgments
- **side-band-64k**: Multiplexed data streams for progress and errors
- **ofs-delta**: Offset-based delta encoding for better compression
- **shallow**: Support for shallow clones and fetches
- **filter**: Partial clone with object filtering
- **atomic**: All-or-nothing push transactions

**Fetch Operation Phases:**
1. **Reference Discovery**: Query remote for available refs and capabilities
2. **Negotiation**: Find common commits to minimize transfer
3. **Pack Transfer**: Receive optimized pack with only missing objects
4. **Reference Update**: Atomically update local refs

**Push Operation Phases:**
1. **Permission Check**: Verify write access to remote
2. **Pack Building**: Create pack with objects missing on remote
3. **Atomic Transaction**: Upload pack and update refs atomically
4. **Hook Execution**: Server-side hooks for policy enforcement

**Delta Compression:**
- Rolling hash for efficient block matching
- Copy/insert instructions for minimal delta size
- Window-based search for similar objects

> **Code Reference**: For complete remote protocol implementation with pack negotiation and delta compression, see [`remote_protocol.py`](../../code-examples/technology/git/remote_protocol.py)

## Remote Repositories

### Remote Management

```bash
# List remotes
git remote -v

# Add remote
git remote add origin https://github.com/user/repo.git

# Change remote URL
git remote set-url origin git@github.com:user/repo.git

# Remove remote
git remote remove origin

# Rename remote
git remote rename origin upstream
```

### Fetching and Pulling

```bash
# Fetch changes
git fetch origin
git fetch --all

# Pull changes
git pull origin main
git pull --rebase origin main

# Set upstream branch
git push -u origin feature/new-feature
git branch --set-upstream-to=origin/main
```

### Pushing

```bash
# Push to remote
git push origin main
git push -u origin feature/new-feature  # Set upstream

# Push all branches
git push --all origin

# Push tags
git push origin v1.0.0          # Specific tag
git push origin --tags          # All tags

# Force push (use with caution)
git push --force origin feature/rebased
git push --force-with-lease     # Safer force push
```

## Advanced Git Algorithms

### Rebase and Bisect Algorithms

**Interactive Rebase:**
Rebase rewrites commit history by replaying commits onto a new base:

**Rebase Commands:**
- **pick**: Use commit as-is
- **reword**: Edit commit message
- **edit**: Stop for amending
- **squash**: Combine with previous commit
- **fixup**: Like squash but discard message
- **exec**: Run shell command
- **drop**: Remove commit
- **label/reset**: Advanced scripting commands
- **merge**: Create merge commit during rebase

**Rebase Process:**
1. Save original HEAD position
2. Checkout onto target
3. Cherry-pick each commit according to todo list
4. Handle conflicts by pausing for resolution
5. Update branch reference when complete

**Binary Search (Bisect):**
Efficiently find commit that introduced a bug:

**Bisect Algorithm:**
1. Mark known good and bad commits
2. Find commits between good and bad
3. Select optimal commit that best bisects the graph
4. Test and mark as good/bad
5. Repeat until first bad commit is found

**Optimization:**
- Weight commits by reachability to minimize search steps
- Skip untestable commits
- Handle non-linear history with merge commits

> **Code Reference**: For complete rebase and bisect implementations with conflict handling, see [`rebase_bisect.py`](../../code-examples/technology/git/rebase_bisect.py)

## Advanced Features

### Stashing

```bash
# Save changes temporarily
git stash
git stash save "Work in progress"

# List stashes
git stash list

# Apply stash
git stash apply                 # Most recent
git stash apply stash@{2}       # Specific stash
git stash pop                   # Apply and remove

# Show stash contents
git stash show -p stash@{0}

# Create branch from stash
git stash branch new-feature stash@{0}

# Clear stashes
git stash drop stash@{0}        # Specific stash
git stash clear                 # All stashes
```

### Cherry-picking

```bash
# Apply specific commit
git cherry-pick abc123

# Cherry-pick multiple commits
git cherry-pick abc123 def456

# Cherry-pick without committing
git cherry-pick -n abc123
```

### Reset and Revert

```bash
# Soft reset (keep changes staged)
git reset --soft HEAD~1

# Mixed reset (keep changes unstaged)
git reset HEAD~1

# Hard reset (discard changes)
git reset --hard HEAD~1

# Reset to specific commit
git reset --hard abc123

# Revert commit (creates new commit)
git revert abc123

# Revert merge commit
git revert -m 1 merge-commit-hash
```

### Searching and Filtering

```bash
# Search in files
git grep "pattern"
git grep -n "pattern"           # With line numbers

# Find commits
git log --grep="bug fix"
git log --author="John"
git log --since="2 weeks ago"
git log --until="2023-01-01"

# Find commit that introduced bug
git bisect start
git bisect bad                  # Current commit is bad
git bisect good abc123          # Known good commit
# Test and mark commits as good/bad
git bisect reset                # When done
```

## Advanced Workflow Patterns

### Formal Workflow Modeling

Git workflows can be formally modeled as state machines to enforce policies and guide development:

**Workflow Components:**
- **States**: Development stages (feature, staging, production)
- **Transitions**: Valid moves between states
- **Branch Mappings**: Patterns mapping branches to states
- **Validation Rules**: Policies enforcing workflow compliance

**Common Workflow Patterns:**

**GitFlow:**
- Long-lived branches: main/master, develop
- Feature branches from develop
- Release branches for staging
- Hotfix branches from production
- Structured release process

**GitHub Flow:**
- Single main branch
- Feature branches for all changes
- Pull requests for code review
- Deploy from feature branches or main
- Simple and continuous deployment friendly

**GitLab Flow:**
- Environment branches (staging, production)
- Main branch for development
- Cherry-pick or merge to promote changes
- Supports multiple deployment environments

**Monorepo Workflows:**
- Project-aware branching strategies
- Dependency analysis for affected projects
- Selective CI/CD based on changes
- Cross-project atomic commits

> **Code Reference**: For complete workflow modeling implementations with state machines and validation, see [`workflow_modeling.py`](../../code-examples/technology/git/workflow_modeling.py)

## Git Workflows

### Git Flow

```bash
# Initialize git flow
git flow init

# Feature branches
git flow feature start new-feature
git flow feature finish new-feature

# Release branches
git flow release start 1.0.0
git flow release finish 1.0.0

# Hotfix branches
git flow hotfix start critical-fix
git flow hotfix finish critical-fix
```

### GitHub Flow

Simple workflow:
1. Create branch from main
2. Add commits
3. Open pull request
4. Discuss and review
5. Merge to main
6. Deploy

### GitLab Flow

Environment branches:
- `main` → `pre-production` → `production`
- Feature branches merge to main
- Cherry-pick to environment branches

## Working with Large Files

### Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track file types
git lfs track "*.psd"
git lfs track "*.zip"

# List tracked patterns
git lfs track

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"

# View LFS files
git lfs ls-files
```

## Submodules

```bash
# Add submodule
git submodule add https://github.com/user/library.git libs/library

# Initialize submodules after clone
git submodule init
git submodule update

# Clone with submodules
git clone --recursive https://github.com/user/repo.git

# Update submodules
git submodule update --remote

# Remove submodule
git submodule deinit libs/library
git rm libs/library
rm -rf .git/modules/libs/library
```

## Hooks

### Client-side Hooks

Create executable scripts in `.git/hooks/`:

**pre-commit:**
```bash
#!/bin/sh
# Run tests before commit
npm test
```

**commit-msg:**
```bash
#!/bin/sh
# Enforce commit message format
grep -qE "^(feat|fix|docs|style|refactor|test|chore): " "$1" || {
    echo "Commit message must start with type: feat|fix|docs|style|refactor|test|chore"
    exit 1
}
```

### Server-side Hooks

- **pre-receive:** Enforce policies
- **update:** Validate branch updates
- **post-receive:** Trigger deployments

## Best Practices

### Commit Messages

```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:
```
feat(auth): add OAuth2 integration

Implemented Google and GitHub OAuth2 providers
with automatic token refresh capability.

Closes #123
```

### Branch Naming

- `feature/user-authentication`
- `bugfix/login-error`
- `hotfix/security-patch`
- `release/v2.0.0`
- `chore/update-dependencies`

### .gitignore

```gitignore
# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Dependencies
node_modules/
vendor/

# Build outputs
dist/
build/
*.exe
*.dll

# Logs
*.log
logs/

# Environment
.env
.env.local

# Temporary files
*.tmp
*.cache
```

### Git Aliases

```bash
# Useful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

## Troubleshooting

### Common Issues

**Detached HEAD:**
```bash
# Create branch from current position
git checkout -b new-branch

# Return to previous branch
git checkout -
```

**Undo operations:**
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo git add
git reset HEAD file.txt

# Discard local changes
git checkout -- file.txt

# Recover deleted branch
git reflog
git checkout -b recovered-branch abc123
```

**Clean repository:**
```bash
# Show what would be removed
git clean -n

# Remove untracked files
git clean -f

# Remove untracked files and directories
git clean -fd

# Remove ignored files too
git clean -fdx
```

## Security

### Signing Commits

```bash
# Configure GPG
git config --global user.signingkey YOUR_GPG_KEY_ID

# Sign commits
git commit -S -m "Signed commit"

# Always sign commits
git config --global commit.gpgsign true

# Verify signatures
git log --show-signature
```

### Sensitive Data

```bash
# Remove file from all history (use BFG Repo-Cleaner)
bfg --delete-files passwords.txt

# Alternative with filter-branch (slower)
git filter-branch --tree-filter 'rm -f passwords.txt' HEAD

# Force push to update remote
git push --force --all
```

## Performance Optimization Theory

### Git Performance Analysis

Git provides sophisticated tools and algorithms for optimizing repository performance:

**Performance Metrics:**
- **Object Database**: Loose objects vs pack files ratio
- **Pack Efficiency**: Compression ratios and delta chain depths
- **Reference Performance**: Loose vs packed refs
- **Working Tree**: Large files and untracked content
- **Network Operations**: Pack negotiation efficiency

**Optimization Strategies:**

**Geometric Repacking:**
- Groups pack files by size in geometric progression
- Each pack ~2x size of previous
- Reduces pack file count while maintaining efficiency
- Optimal for large repositories with many packs

**Bitmap Indexes:**
- EWAH compressed bitmaps for reachability queries
- Dramatically speeds up counting and traversal operations
- Optimal commit selection for coverage
- Used in fetch/push negotiations

**Optimization Techniques:**
- Multi-pack indexes for very large repositories
- Commit graphs for faster traversal
- Bloom filters for changed paths
- Delta islands for fork networks

**Performance Tuning Parameters:**
- `pack.window`: Delta search window size
- `pack.depth`: Maximum delta chain length
- `pack.threads`: Parallel compression threads
- `core.preloadIndex`: Parallel index loading

> **Code Reference**: For complete performance analysis and optimization implementations, see [`performance_optimization.py`](../../code-examples/technology/git/performance_optimization.py)

## Performance

### Optimization

```bash
# Garbage collection
git gc

# Aggressive garbage collection
git gc --aggressive

# Verify repository integrity
git fsck

# Count objects
git count-objects -v

# Prune old objects
git prune
```

### Large Repositories

```bash
# Shallow clone
git clone --depth 1 https://github.com/large/repo.git

# Partial clone (Git 2.17+)
git clone --filter=blob:none https://github.com/large/repo.git

# Sparse checkout
git sparse-checkout init
git sparse-checkout set src/ docs/
```

## Research Frontiers in Version Control

### Emerging Paradigms

The future of version control explores novel approaches beyond traditional DAG-based systems:

**Quantum Version Control:**
- Superposition of multiple development branches
- Quantum entanglement for correlated changes
- Amplitude amplification for optimal path selection
- Theoretical framework: |ψ⟩ = Σ αᵢ|branchᵢ⟩

**CRDT-Based VCS:**
- Conflict-free Replicated Data Types guarantee convergence
- No coordination required for merging
- Eventual consistency without central authority
- Operation-based and state-based CRDT approaches

**Blockchain Version Control:**
- Immutable commit history with cryptographic proof
- Smart contracts for automated governance
- Decentralized trust without central repository
- Proof-of-work or proof-of-stake for commit validation

### Machine Learning Integration

**Conflict Prediction:**
- Feature extraction from branch histories
- Neural networks trained on historical conflicts
- Proactive warnings before problematic merges
- Integration with CI/CD pipelines

**Automated Commit Messages:**
- Transformer models analyzing code diffs
- Context-aware message generation
- Conventional commit format compliance
- Multi-language support

**Merge Strategy Optimization:**
- Reinforcement learning for strategy selection
- State encoding of repository structure
- Reward based on merge success and conflicts
- Adaptive to project-specific patterns

**Code Review Automation:**
- AI-powered vulnerability detection
- Style and pattern suggestions
- Performance impact analysis
- Test coverage recommendations

> **Code Reference**: For experimental implementations of quantum VCS, CRDT-based systems, blockchain VCS, and ML enhancements, see [`research_frontiers.py`](../../code-examples/technology/git/research_frontiers.py)

## References and Further Reading

### Academic Papers
- "A Formal Model of Git Version Control" - ICSE 2019
- "Distributed Version Control as a Distributed Database" - VLDB 2018
- "Merkle Trees and Content-Addressable Storage" - IEEE 2017
- "Conflict-Free Replicated Data Types for Distributed Version Control" - PODC 2020

### Books
- "Pro Git" - Scott Chacon and Ben Straub
- "Version Control with Git" - Jon Loeliger and Matthew McCullough
- "Git Internals" - Scott Chacon
- "Building Git" - James Coglan

### Advanced Topics
- libgit2: Git core library implementation
- JGit: Pure Java implementation of Git
- Dulwich: Pure Python implementation of Git
- Git wire protocol specification
- Pack file format specification
- Index file format specification

### Research Projects
- **Pijul**: Patch-based version control with formal theory
- **Darcs**: Patch theory and commutation
- **Fossil**: Distributed VCS with integrated wiki and issue tracking
- **Mercurial**: Alternative DVCS with different design choices

## Conclusion

Git has revolutionized software development by providing a distributed, efficient, and reliable version control system. From its elegant mathematical foundations to its practical everyday commands, Git offers tools for every level of developer.

### Key Takeaways

1. **Start Simple**: You don't need to understand everything to be productive. Master the basic workflow first.

2. **Branches are Cheap**: Use branches liberally for features, experiments, and bug fixes.

3. **Commit Often**: Small, focused commits are easier to understand and revert if needed.

4. **Write Good Messages**: Your future self will thank you for clear commit messages.

5. **Learn the Internals**: Understanding how Git works helps you recover from mistakes and optimize workflows.

6. **Practice Recovery**: Knowing how to undo, reset, and recover gives confidence to experiment.

### Next Steps

- **Practice**: Create a test repository and try different commands
- **Explore**: Read the Pro Git book for comprehensive coverage
- **Contribute**: Join open source projects to experience collaborative workflows
- **Customize**: Set up aliases and hooks for your workflow
- **Stay Updated**: Git continues to evolve with new features and improvements

Git's elegant design, based on solid computer science principles, provides a foundation for complex workflows while maintaining data integrity through cryptographic guarantees. Whether you're fixing a typo or architecting a complex feature across multiple branches, Git has the tools to support your development process. Understanding Git deeply transforms it from a necessary tool into a powerful ally in creating better software.