# Git Version Control

<html><header><link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css"></header></html>

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">Git</h1>
    <p class="hero-subtitle">Distributed Version Control: Theory and Implementation</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Git represents a paradigm shift in version control systems, implementing a content-addressable filesystem with a distributed architecture based on cryptographic principles. Built on directed acyclic graphs (DAGs), Merkle trees, and SHA-1 hashing, Git provides mathematical guarantees about data integrity while enabling sophisticated workflows through its elegant object model.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-project-diagram"></i>
      <h4>DAG-Based History</h4>
      <p>Directed acyclic graph for commits</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-shield-alt"></i>
      <h4>Cryptographic Integrity</h4>
      <p>SHA-1/SHA-256 content addressing</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-network-wired"></i>
      <h4>Distributed Architecture</h4>
      <p>Peer-to-peer repository model</p>
    </div>
  </div>
</div>

## Mathematical Foundations

### Content-Addressable Storage Theory

Git implements a content-addressable filesystem where objects are stored and retrieved by their SHA-1 hash. This provides cryptographic integrity and enables efficient storage through deduplication.

**Key Components:**
- **GitObject**: Base class for all Git objects (blob, tree, commit, tag)
- **SHA-1 Hashing**: Content addressing using cryptographic hashes
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

```python
class GitObjectStore:
    """Low-level Git object storage implementation"""
    
    def __init__(self, git_dir: str = ".git"):
        self.git_dir = git_dir
        self.objects_dir = os.path.join(git_dir, "objects")
    
    def write_object(self, obj: GitObject) -> str:
        """Write object to disk using content addressing"""
        sha = obj.sha1
        compressed = obj.compress()
        
        # Objects stored as .git/objects/ab/cdef...
        dir_name = sha[:2]
        file_name = sha[2:]
        
        obj_dir = os.path.join(self.objects_dir, dir_name)
        os.makedirs(obj_dir, exist_ok=True)
        
        obj_path = os.path.join(obj_dir, file_name)
        with open(obj_path, 'wb') as f:
            f.write(compressed)
        
        return sha
    
    def read_object(self, sha: str) -> GitObject:
        """Read and decompress object from disk"""
        dir_name = sha[:2]
        file_name = sha[2:]
        
        obj_path = os.path.join(self.objects_dir, dir_name, file_name)
        with open(obj_path, 'rb') as f:
            compressed = f.read()
        
        decompressed = zlib.decompress(compressed)
        
        # Parse header
        null_idx = decompressed.index(b'\0')
        header = decompressed[:null_idx].decode()
        content = decompressed[null_idx + 1:]
        
        obj_type, size = header.split(' ')
        return GitObject(ObjectType(obj_type), content)

class PackFile:
    """Git packfile format for efficient storage"""
    
    def __init__(self, pack_path: str):
        self.pack_path = pack_path
        self.idx_path = pack_path.replace('.pack', '.idx')
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, int]:
        """Load pack index for fast lookups"""
        # Pack index format (v2):
        # - Header (8 bytes)
        # - Fanout table (256 * 4 bytes)
        # - SHA1 entries (20 bytes each)
        # - CRC32 checksums (4 bytes each)
        # - Offsets (4 or 8 bytes each)
        # - 64-bit offsets (if needed)
        # - Pack checksum (20 bytes)
        # - Index checksum (20 bytes)
        
        index = {}
        with open(self.idx_path, 'rb') as f:
            # Parse index file
            header = f.read(8)
            if header != b'\xff\x74\x4f\x63\x00\x00\x00\x02':
                raise ValueError("Invalid pack index version")
            
            # Read fanout table
            fanout = []
            for _ in range(256):
                fanout.append(struct.unpack('>I', f.read(4))[0])
            
            num_objects = fanout[-1]
            
            # Read SHA1s
            shas = []
            for _ in range(num_objects):
                shas.append(f.read(20).hex())
            
            # Skip CRC32s
            f.seek(4 * num_objects, 1)
            
            # Read offsets
            for i, sha in enumerate(shas):
                offset = struct.unpack('>I', f.read(4))[0]
                if offset & 0x80000000:
                    # Large offset stored separately
                    offset = self._read_large_offset(f, offset & 0x7fffffff)
                index[sha] = offset
        
        return index

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

```python
class RefManager:
    """Manages Git references (branches, tags, etc.)"""
    
    def __init__(self, git_dir: str):
        self.git_dir = git_dir
        self.refs_dir = os.path.join(git_dir, "refs")
    
    def read_ref(self, ref_name: str) -> Optional[str]:
        """Read reference value (commit SHA)"""
        # Check packed refs first
        packed_ref = self._read_packed_ref(ref_name)
        if packed_ref:
            return packed_ref
        
        # Check loose ref
        ref_path = os.path.join(self.git_dir, ref_name)
        if os.path.exists(ref_path):
            with open(ref_path, 'r') as f:
                return f.read().strip()
        
        return None
    
    def update_ref(self, ref_name: str, new_sha: str, 
                  old_sha: Optional[str] = None) -> bool:
        """Atomically update reference"""
        ref_path = os.path.join(self.git_dir, ref_name)
        
        # Ensure atomic update
        if old_sha is not None:
            current = self.read_ref(ref_name)
            if current != old_sha:
                return False  # Concurrent modification
        
        # Write to temporary file and rename
        tmp_path = f"{ref_path}.lock"
        with open(tmp_path, 'w') as f:
            f.write(new_sha + '\n')
        
        os.rename(tmp_path, ref_path)
        return True
    
    def _read_packed_ref(self, ref_name: str) -> Optional[str]:
        """Read from packed-refs file"""
        packed_refs_path = os.path.join(self.git_dir, "packed-refs")
        if not os.path.exists(packed_refs_path):
            return None
        
        with open(packed_refs_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('^'):
                    continue
                sha, ref = line.strip().split(' ', 1)
                if ref == ref_name:
                    return sha
        
        return None
```

## Advanced Command Implementation

### Repository Operations at the Protocol Level

```python
class GitProtocol:
    """Git wire protocol implementation"""
    
    def __init__(self, transport: 'Transport'):
        self.transport = transport
        self.capabilities = set()
    
    def discover_refs(self) -> Dict[str, str]:
        """Discover references using smart HTTP protocol"""
        # Send: GET /info/refs?service=git-upload-pack
        response = self.transport.get("/info/refs", 
                                    params={"service": "git-upload-pack"})
        
        refs = {}
        for line in response.split('\n'):
            if line.startswith('#'):
                continue
            if ' ' in line:
                sha, ref = line.split(' ', 1)
                refs[ref] = sha
                
                # Parse capabilities from first line
                if '\0' in ref:
                    ref, caps = ref.split('\0', 1)
                    self.capabilities = set(caps.split(' '))
        
        return refs
    
    def negotiate_pack(self, wants: List[str], 
                      haves: List[str]) -> bytes:
        """Negotiate pack file using smart protocol"""
        # Build want lines
        request = []
        for i, want in enumerate(wants):
            if i == 0 and self.capabilities:
                caps = ' '.join(self.capabilities)
                request.append(f"want {want} {caps}\n")
            else:
                request.append(f"want {want}\n")
        
        # Build have lines
        for have in haves:
            request.append(f"have {have}\n")
        
        request.append("done\n")
        
        # Send request
        response = self.transport.post("/git-upload-pack", 
                                     data=''.join(request))
        
        # Parse pack file
        return self._parse_pack_response(response)

class CloneOperation:
    """High-level clone implementation"""
    
    def __init__(self, url: str, target_dir: str):
        self.url = url
        self.target_dir = target_dir
        self.protocol = GitProtocol(HTTPTransport(url))
    
    def clone(self, branch: Optional[str] = None, 
             depth: Optional[int] = None):
        """Perform repository clone"""
        # Create directory structure
        os.makedirs(self.target_dir)
        git_dir = os.path.join(self.target_dir, ".git")
        os.makedirs(git_dir)
        
        # Initialize repository structure
        self._init_repo_structure(git_dir)
        
        # Discover refs
        refs = self.protocol.discover_refs()
        
        # Determine what to fetch
        if branch:
            want_ref = f"refs/heads/{branch}"
            if want_ref not in refs:
                raise ValueError(f"Branch {branch} not found")
            wants = [refs[want_ref]]
        else:
            # Fetch all branches
            wants = [sha for ref, sha in refs.items() 
                    if ref.startswith("refs/heads/")]
        
        # Negotiate pack
        if depth:
            # Shallow clone
            self.protocol.capabilities.add(f"deepen {depth}")
        
        pack_data = self.protocol.negotiate_pack(wants, [])
        
        # Process pack file
        self._process_pack(git_dir, pack_data)
        
        # Update references
        self._update_refs(git_dir, refs)
        
        # Checkout HEAD
        self._checkout_head(git_dir)
```

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

```python
class MergeStrategy:
    """Base class for merge strategies"""
    
    def merge(self, base: 'Commit', ours: 'Commit', 
             theirs: 'Commit') -> 'MergeResult':
        raise NotImplementedError

class RecursiveMergeStrategy(MergeStrategy):
    """Git's default recursive merge strategy"""
    
    def merge(self, base: 'Commit', ours: 'Commit', 
             theirs: 'Commit') -> 'MergeResult':
        # Handle multiple merge bases
        merge_bases = self.find_merge_bases(ours, theirs)
        
        if len(merge_bases) > 1:
            # Recursively merge the merge bases
            virtual_base = self.merge_bases(merge_bases)
            return self.three_way_merge(virtual_base, ours, theirs)
        else:
            # Simple three-way merge
            return self.three_way_merge(merge_bases[0], ours, theirs)
    
    def three_way_merge(self, base: 'Commit', ours: 'Commit', 
                       theirs: 'Commit') -> 'MergeResult':
        """Perform three-way merge on trees"""
        result = MergeResult()
        
        # Get trees
        base_tree = base.tree
        our_tree = ours.tree
        their_tree = theirs.tree
        
        # Merge trees recursively
        merged_tree = self.merge_trees(base_tree, our_tree, their_tree, result)
        
        if not result.conflicts:
            # Create merge commit
            merge_commit = Commit(
                tree=merged_tree,
                parents=[ours.sha, theirs.sha],
                message=f"Merge {theirs.sha[:7]} into {ours.sha[:7]}"
            )
            result.commit = merge_commit
        
        return result
    
    def merge_trees(self, base: 'Tree', ours: 'Tree', 
                   theirs: 'Tree', result: 'MergeResult') -> 'Tree':
        """Recursively merge directory trees"""
        merged_entries = {}
        
        # Get all unique paths
        all_paths = set()
        all_paths.update(base.entries.keys())
        all_paths.update(ours.entries.keys())
        all_paths.update(theirs.entries.keys())
        
        for path in all_paths:
            base_entry = base.entries.get(path)
            our_entry = ours.entries.get(path)
            their_entry = theirs.entries.get(path)
            
            merged_entry = self.merge_entry(
                path, base_entry, our_entry, their_entry, result
            )
            
            if merged_entry:
                merged_entries[path] = merged_entry
        
        return Tree(entries=merged_entries)

class OctopusMergeStrategy(MergeStrategy):
    """Strategy for merging multiple branches"""
    
    def merge_multiple(self, head: 'Commit', 
                      branches: List['Commit']) -> 'MergeResult':
        """Merge multiple branches simultaneously"""
        # Find common base for all branches
        common_base = self.find_common_base_all([head] + branches)
        
        # Build multi-way merge tree
        merge_tree = MultiWayMergeTree(common_base)
        merge_tree.add_branch(head)
        for branch in branches:
            merge_tree.add_branch(branch)
        
        # Perform merge
        result = merge_tree.merge()
        
        if result.success:
            # Create octopus merge commit
            merge_commit = Commit(
                tree=result.tree,
                parents=[head.sha] + [b.sha for b in branches],
                message=f"Merge branches {', '.join(b.ref for b in branches)}"
            )
            result.commit = merge_commit
        
        return result
```

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

```python
class RemoteProtocol:
    """Git remote protocol handler"""
    
    def __init__(self, transport: 'Transport'):
        self.transport = transport
        self.capabilities = {
            'multi_ack_detailed',
            'side-band-64k',
            'ofs-delta',
            'agent=git/2.x.x',
            'allow-tip-sha1-in-want',
            'allow-reachable-sha1-in-want'
        }
    
    def fetch_pack(self, remote: 'Remote', 
                  refs: List[str]) -> 'FetchResult':
        """Fetch objects from remote"""
        # Phase 1: Reference discovery
        remote_refs = self.discover_refs()
        
        # Phase 2: Negotiation
        common_commits = self.negotiate_common_commits(remote_refs)
        
        # Phase 3: Pack transfer
        pack_data = self.receive_pack(remote_refs, common_commits)
        
        # Phase 4: Update local refs
        return self.update_local_refs(pack_data, remote_refs)
    
    def push_pack(self, remote: 'Remote', 
                 refspecs: List['RefSpec']) -> 'PushResult':
        """Push objects to remote"""
        # Validate push permissions
        if not self.check_push_permissions(remote):
            raise PermissionError("Push access denied")
        
        # Build pack of missing objects
        pack = self.build_push_pack(refspecs)
        
        # Send pack with atomic transaction
        with self.atomic_push_transaction() as txn:
            # Upload pack
            txn.send_pack(pack)
            
            # Update remote refs
            for refspec in refspecs:
                txn.update_ref(refspec)
            
            # Commit or rollback
            if txn.verify():
                txn.commit()
                return PushResult(success=True)
            else:
                txn.rollback()
                return PushResult(success=False, 
                               reason="Verification failed")

class DeltaCompression:
    """Delta compression for efficient transfer"""
    
    def __init__(self):
        self.window_size = 10
        self.max_delta_size = 1024 * 1024  # 1MB
    
    def compute_delta(self, source: bytes, target: bytes) -> bytes:
        """Compute binary delta between objects"""
        # Use xdelta algorithm
        delta = []
        
        # Build rolling hash table for source
        source_hashes = self.build_hash_table(source)
        
        i = 0
        while i < len(target):
            # Try to find matching block
            match = self.find_match(target[i:], source_hashes, source)
            
            if match:
                # Emit copy instruction
                delta.append(DeltaOp.copy(match.offset, match.length))
                i += match.length
            else:
                # Emit insert instruction
                delta.append(DeltaOp.insert(target[i]))
                i += 1
        
        return self.encode_delta(delta)
```

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

### Rebase Implementation

```python
class RebaseOperation:
    """Interactive rebase implementation"""
    
    def __init__(self, onto: str, upstream: str, branch: str):
        self.onto = onto
        self.upstream = upstream
        self.branch = branch
        self.todo_list: List['RebaseAction'] = []
    
    def prepare_rebase(self) -> List['RebaseAction']:
        """Prepare rebase todo list"""
        # Find commits to rebase
        commits = self.find_commits_to_rebase()
        
        # Generate todo list
        for commit in commits:
            action = RebaseAction('pick', commit)
            self.todo_list.append(action)
        
        return self.todo_list
    
    def execute_rebase(self) -> 'RebaseResult':
        """Execute rebase operations"""
        # Save original HEAD
        orig_head = self.get_head()
        
        # Checkout onto commit
        self.checkout(self.onto)
        
        # Apply commits
        for action in self.todo_list:
            result = self.apply_action(action)
            
            if result.conflict:
                # Pause for conflict resolution
                self.save_rebase_state(action)
                return RebaseResult(
                    status='conflict',
                    message=f"Conflict while applying {action.commit[:7]}"
                )
        
        # Update branch ref
        self.update_ref(self.branch, self.get_head())
        
        return RebaseResult(status='success')
    
    def apply_action(self, action: 'RebaseAction') -> 'ActionResult':
        """Apply single rebase action"""
        if action.command == 'pick':
            return self.cherry_pick(action.commit)
        elif action.command == 'reword':
            result = self.cherry_pick(action.commit)
            if result.success:
                self.amend_commit_message(action.new_message)
            return result
        elif action.command == 'squash':
            return self.squash_with_previous(action.commit)
        elif action.command == 'fixup':
            return self.squash_with_previous(action.commit, 
                                           keep_message=False)
        elif action.command == 'drop':
            return ActionResult(success=True)  # Skip commit
        else:
            raise ValueError(f"Unknown rebase command: {action.command}")

class BisectAlgorithm:
    """Binary search for bug introduction"""
    
    def __init__(self, good: str, bad: str):
        self.good_commits = {good}
        self.bad_commits = {bad}
        self.tested_commits = set()
    
    def next_commit(self) -> Optional[str]:
        """Find next commit to test using binary search"""
        # Build commit graph between good and bad
        graph = self.build_commit_graph()
        
        # Find midpoint using graph distance
        untested = self.find_untested_commits(graph)
        if not untested:
            return None
        
        # Weight commits by reachability
        weights = self.calculate_commit_weights(untested, graph)
        
        # Select commit that best bisects the graph
        return self.select_optimal_commit(weights)
    
    def calculate_commit_weights(self, commits: Set[str], 
                               graph: 'CommitGraph') -> Dict[str, float]:
        """Calculate bisection efficiency for each commit"""
        weights = {}
        
        for commit in commits:
            # Count reachable commits from this point
            reachable_good = self.count_reachable(commit, 
                                                self.good_commits, graph)
            reachable_bad = self.count_reachable(commit, 
                                               self.bad_commits, graph)
            
            # Optimal weight minimizes |reachable_good - reachable_bad|
            weight = abs(reachable_good - reachable_bad)
            weights[commit] = weight
        
        return weights
```

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

```python
from enum import Enum
from typing import Dict, List, Set, Optional
import networkx as nx

class WorkflowState(Enum):
    """States in a Git workflow state machine"""
    DEVELOPMENT = "development"
    FEATURE = "feature"
    INTEGRATION = "integration"
    STAGING = "staging"
    PRODUCTION = "production"
    HOTFIX = "hotfix"

class GitWorkflow:
    """Abstract base for Git workflow implementations"""
    
    def __init__(self):
        self.state_graph = nx.DiGraph()
        self.branch_mapping: Dict[str, WorkflowState] = {}
        self.transition_rules: List['TransitionRule'] = []
        self._define_workflow()
    
    def _define_workflow(self):
        """Define workflow states and transitions"""
        raise NotImplementedError
    
    def validate_transition(self, from_branch: str, 
                          to_branch: str) -> bool:
        """Validate if transition is allowed"""
        from_state = self.branch_mapping.get(from_branch)
        to_state = self.branch_mapping.get(to_branch)
        
        if not from_state or not to_state:
            return False
        
        # Check if path exists in state graph
        return nx.has_path(self.state_graph, from_state, to_state)
    
    def suggest_next_actions(self, current_branch: str) -> List[str]:
        """Suggest valid next actions based on current state"""
        current_state = self.branch_mapping.get(current_branch)
        if not current_state:
            return []
        
        # Find reachable states
        reachable = nx.descendants(self.state_graph, current_state)
        
        actions = []
        for state in reachable:
            branches = [b for b, s in self.branch_mapping.items() 
                       if s == state]
            actions.extend(branches)
        
        return actions

class GitFlowWorkflow(GitWorkflow):
    """GitFlow workflow implementation"""
    
    def _define_workflow(self):
        # Define states
        states = [
            WorkflowState.DEVELOPMENT,
            WorkflowState.FEATURE,
            WorkflowState.INTEGRATION,
            WorkflowState.STAGING,
            WorkflowState.PRODUCTION,
            WorkflowState.HOTFIX
        ]
        
        self.state_graph.add_nodes_from(states)
        
        # Define transitions
        self.state_graph.add_edges_from([
            (WorkflowState.DEVELOPMENT, WorkflowState.FEATURE),
            (WorkflowState.FEATURE, WorkflowState.DEVELOPMENT),
            (WorkflowState.DEVELOPMENT, WorkflowState.STAGING),
            (WorkflowState.STAGING, WorkflowState.PRODUCTION),
            (WorkflowState.PRODUCTION, WorkflowState.HOTFIX),
            (WorkflowState.HOTFIX, WorkflowState.PRODUCTION),
            (WorkflowState.HOTFIX, WorkflowState.DEVELOPMENT)
        ])
        
        # Map branches to states
        self.branch_mapping = {
            'develop': WorkflowState.DEVELOPMENT,
            'master': WorkflowState.PRODUCTION,
            'main': WorkflowState.PRODUCTION,
            'release/*': WorkflowState.STAGING,
            'feature/*': WorkflowState.FEATURE,
            'hotfix/*': WorkflowState.HOTFIX
        }
    
    def create_feature_branch(self, feature_name: str) -> 'Branch':
        """Create feature branch following GitFlow"""
        # Features branch from develop
        source = 'develop'
        branch_name = f'feature/{feature_name}'
        
        # Validate source exists
        if not self.ref_exists(source):
            raise ValueError(f"Source branch {source} not found")
        
        # Create branch
        branch = Branch(branch_name, source)
        self.checkout(branch_name, create=True, start_point=source)
        
        return branch

class MonorepoWorkflow(GitWorkflow):
    """Workflow for monorepo management"""
    
    def __init__(self, projects: List[str]):
        self.projects = projects
        self.project_dependencies = self._analyze_dependencies()
        super().__init__()
    
    def _analyze_dependencies(self) -> nx.DiGraph:
        """Analyze inter-project dependencies"""
        dep_graph = nx.DiGraph()
        
        for project in self.projects:
            deps = self._get_project_dependencies(project)
            for dep in deps:
                dep_graph.add_edge(project, dep)
        
        return dep_graph
    
    def affected_projects(self, changed_files: List[str]) -> Set[str]:
        """Determine projects affected by changes"""
        directly_affected = set()
        
        # Find directly affected projects
        for file in changed_files:
            project = self._file_to_project(file)
            if project:
                directly_affected.add(project)
        
        # Find transitively affected projects
        all_affected = set(directly_affected)
        for project in directly_affected:
            # All projects that depend on this one
            dependents = nx.ancestors(self.project_dependencies, project)
            all_affected.update(dependents)
        
        return all_affected
```

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

```python
class PerformanceAnalyzer:
    """Analyze and optimize Git repository performance"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.metrics = {}
    
    def analyze_repository(self) -> 'PerformanceReport':
        """Comprehensive performance analysis"""
        report = PerformanceReport()
        
        # Object database metrics
        report.object_stats = self.analyze_object_database()
        
        # Pack file efficiency
        report.pack_stats = self.analyze_pack_files()
        
        # Reference performance
        report.ref_stats = self.analyze_references()
        
        # Working tree analysis
        report.worktree_stats = self.analyze_working_tree()
        
        # Generate recommendations
        report.recommendations = self.generate_recommendations(report)
        
        return report
    
    def analyze_object_database(self) -> 'ObjectStats':
        """Analyze object database performance"""
        stats = ObjectStats()
        
        # Count objects by type
        for obj_file in self.iter_loose_objects():
            obj = self.read_object(obj_file)
            stats.count_by_type[obj.type] += 1
            stats.total_size += obj.size
            
            # Track large objects
            if obj.size > 10 * 1024 * 1024:  # 10MB
                stats.large_objects.append(obj)
        
        # Analyze object reachability
        stats.unreachable_objects = self.find_unreachable_objects()
        
        return stats
    
    def optimize_pack_files(self) -> 'OptimizationResult':
        """Optimize pack file organization"""
        # Repack with optimal parameters
        repack_config = {
            'window': 250,          # Larger window for better compression
            'depth': 50,            # Deeper delta chains
            'threads': os.cpu_count(),
            'compression': 9,       # Maximum compression
            'delta_base_offset': True
        }
        
        # Use geometric repacking for large repos
        if self.is_large_repository():
            return self.geometric_repack(repack_config)
        else:
            return self.standard_repack(repack_config)
    
    def geometric_repack(self, config: Dict) -> 'OptimizationResult':
        """Geometric repacking for better performance"""
        # Group packs by size with geometric progression
        packs = self.get_pack_files()
        
        # Sort by size
        packs.sort(key=lambda p: p.size)
        
        # Create geometric series of pack sizes
        # Each pack is ~2x the size of previous
        result = OptimizationResult()
        current_group = []
        target_size = 1024 * 1024  # Start at 1MB
        
        for pack in packs:
            current_group.append(pack)
            
            if sum(p.size for p in current_group) >= target_size:
                # Repack this group
                new_pack = self.repack_group(current_group, config)
                result.new_packs.append(new_pack)
                
                # Reset for next group
                current_group = []
                target_size *= 2  # Geometric progression
        
        return result

class BitmapIndex:
    """Reachability bitmap implementation for fast operations"""
    
    def __init__(self, pack_file: 'PackFile'):
        self.pack_file = pack_file
        self.bitmaps: Dict[str, 'EWAHBitmap'] = {}
        self._build_bitmaps()
    
    def _build_bitmaps(self):
        """Build reachability bitmaps for commits"""
        # Select commits for bitmap coverage
        selected_commits = self.select_bitmap_commits()
        
        for commit in selected_commits:
            # Build reachability bitmap
            bitmap = EWAHBitmap()
            reachable = self.find_reachable_objects(commit)
            
            for obj_idx in reachable:
                bitmap.set(obj_idx)
            
            # Compress bitmap
            bitmap.compress()
            self.bitmaps[commit] = bitmap
    
    def select_bitmap_commits(self) -> List[str]:
        """Select optimal commits for bitmap coverage"""
        # Use commit selection algorithm
        # Goals: minimize total bitmaps while maximizing coverage
        
        commits = self.get_all_commits()
        selected = []
        covered_objects = set()
        
        while len(covered_objects) < self.total_objects * 0.95:
            best_commit = None
            best_coverage = 0
            
            for commit in commits:
                if commit in selected:
                    continue
                
                # Calculate new coverage
                reachable = self.find_reachable_objects(commit)
                new_coverage = len(reachable - covered_objects)
                
                if new_coverage > best_coverage:
                    best_commit = commit
                    best_coverage = new_coverage
            
            if best_commit:
                selected.append(best_commit)
                covered_objects.update(
                    self.find_reachable_objects(best_commit)
                )
            else:
                break
        
        return selected
```

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

### Merkle DAG Alternatives

```python
class QuantumVersionControl:
    """
    Theoretical: Quantum superposition for version control
    """
    
    def __init__(self):
        self.quantum_states = {}
    
    def create_superposition_branch(self, branches: List[str]):
        """
        Create quantum superposition of multiple branch states
        """
        # |ψ⟩ = Σ αᵢ|branchᵢ⟩
        # Allows exploring multiple development paths simultaneously
        pass
    
    def collapse_to_optimal(self, objective_function):
        """
        Collapse superposition to optimal branch based on criteria
        """
        # Measure quantum state to select best development path
        pass

class CRDTBasedVCS:
    """
    Conflict-free Replicated Data Types for version control
    """
    
    def __init__(self):
        self.operation_log = OperationCRDT()
        self.file_states = StateCRDT()
    
    def merge_without_conflicts(self, replicas: List['Replica']):
        """
        Automatically merge any number of replicas without conflicts
        """
        # CRDTs guarantee convergence without coordination
        merged_state = self.file_states.empty()
        
        for replica in replicas:
            merged_state = merged_state.merge(replica.state)
        
        return merged_state

class BlockchainVCS:
    """
    Blockchain-based version control for trust and immutability
    """
    
    def __init__(self):
        self.blockchain = Blockchain()
        self.smart_contracts = {}
    
    def commit_with_proof_of_work(self, changes: 'Changes'):
        """
        Commit changes with blockchain proof of work
        """
        block = Block(
            index=self.blockchain.height + 1,
            timestamp=time.time(),
            data=changes.serialize(),
            previous_hash=self.blockchain.last_block.hash
        )
        
        # Mine block
        proof = self.proof_of_work(block)
        block.nonce = proof
        
        # Add to blockchain
        self.blockchain.add_block(block)
        
        return block.hash
```

### Machine Learning for Version Control

```python
class MLEnhancedGit:
    """
    Machine learning enhancements for Git workflows
    """
    
    def __init__(self):
        self.conflict_predictor = self.load_conflict_model()
        self.commit_suggester = self.load_suggestion_model()
        self.merge_optimizer = self.load_merge_model()
    
    def predict_merge_conflicts(self, branch1: str, 
                              branch2: str) -> float:
        """
        Predict probability of merge conflicts
        """
        # Extract features
        features = self.extract_branch_features(branch1, branch2)
        
        # Predict using trained model
        conflict_probability = self.conflict_predictor.predict_proba(
            features
        )[0][1]
        
        return conflict_probability
    
    def suggest_commit_message(self, diff: 'Diff') -> str:
        """
        Generate commit message using transformer model
        """
        # Tokenize diff
        tokens = self.tokenize_diff(diff)
        
        # Generate message
        message = self.commit_suggester.generate(
            tokens,
            max_length=72,
            temperature=0.7
        )
        
        return message
    
    def optimize_merge_strategy(self, commits: List['Commit']) -> 'Strategy':
        """
        Use reinforcement learning to find optimal merge strategy
        """
        state = self.encode_repository_state()
        
        # Use trained RL agent
        action = self.merge_optimizer.act(state)
        
        return self.decode_strategy(action)
```

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

Git's elegant design, based on solid computer science principles, has revolutionized software development. Its content-addressable storage, distributed architecture, and powerful branching model provide a foundation for complex workflows while maintaining data integrity through cryptographic guarantees. Understanding Git at this deep level enables developers to leverage its full potential and contribute to the evolution of version control systems.