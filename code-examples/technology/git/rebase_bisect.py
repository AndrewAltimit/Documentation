"""
Git Rebase and Bisect Algorithms

Implementation of advanced Git operations:
- Interactive rebase with various commands
- Cherry-pick operation
- Binary search (bisect) for finding bugs
- Conflict resolution during rebase
"""

from typing import List, Optional, Dict, Set, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import os
import json
from collections import deque


class RebaseCommand(Enum):
    """Interactive rebase commands"""
    PICK = "pick"        # Use commit
    REWORD = "reword"    # Use commit, edit message
    EDIT = "edit"        # Use commit, stop for amending
    SQUASH = "squash"    # Use commit, meld into previous
    FIXUP = "fixup"      # Like squash, discard message
    EXEC = "exec"        # Run command
    DROP = "drop"        # Remove commit
    LABEL = "label"      # Label current HEAD
    RESET = "reset"      # Reset HEAD to label
    MERGE = "merge"      # Create merge commit


@dataclass
class RebaseAction:
    """Single action in rebase todo list"""
    command: RebaseCommand
    commit: str
    message: Optional[str] = None
    new_message: Optional[str] = None
    exec_command: Optional[str] = None
    label_name: Optional[str] = None


@dataclass
class RebaseState:
    """Current state of rebase operation"""
    onto: str
    upstream: str  
    branch: str
    todo_list: List[RebaseAction]
    done_list: List[RebaseAction]
    current_action: Optional[RebaseAction]
    orig_head: str
    head: str
    conflicts: List[str]
    
    def save(self, git_dir: str):
        """Save rebase state to disk"""
        rebase_dir = os.path.join(git_dir, "rebase-merge")
        os.makedirs(rebase_dir, exist_ok=True)
        
        # Save state files
        with open(os.path.join(rebase_dir, "onto"), 'w') as f:
            f.write(self.onto)
        
        with open(os.path.join(rebase_dir, "orig-head"), 'w') as f:
            f.write(self.orig_head)
        
        with open(os.path.join(rebase_dir, "head"), 'w') as f:
            f.write(self.head)
        
        # Save todo list
        with open(os.path.join(rebase_dir, "git-rebase-todo"), 'w') as f:
            for action in self.todo_list:
                f.write(f"{action.command.value} {action.commit} {action.message or ''}\n")
    
    @classmethod
    def load(cls, git_dir: str) -> Optional['RebaseState']:
        """Load rebase state from disk"""
        rebase_dir = os.path.join(git_dir, "rebase-merge")
        if not os.path.exists(rebase_dir):
            return None
        
        # Load state files
        with open(os.path.join(rebase_dir, "onto"), 'r') as f:
            onto = f.read().strip()
        
        with open(os.path.join(rebase_dir, "orig-head"), 'r') as f:
            orig_head = f.read().strip()
        
        # Parse todo list
        todo_list = []
        todo_path = os.path.join(rebase_dir, "git-rebase-todo")
        if os.path.exists(todo_path):
            with open(todo_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split(' ', 2)
                        if len(parts) >= 2:
                            cmd = RebaseCommand(parts[0])
                            commit = parts[1]
                            msg = parts[2] if len(parts) > 2 else None
                            todo_list.append(RebaseAction(cmd, commit, msg))
        
        return cls(
            onto=onto,
            upstream="",  # Would load from state
            branch="",    # Would load from state
            todo_list=todo_list,
            done_list=[],
            current_action=None,
            orig_head=orig_head,
            head=onto,
            conflicts=[]
        )


@dataclass
class RebaseResult:
    """Result of rebase operation"""
    status: str  # 'success', 'conflict', 'error'
    message: Optional[str] = None
    conflicts: List[str] = None


@dataclass
class ActionResult:
    """Result of applying single rebase action"""
    success: bool
    conflict: bool = False
    new_commit: Optional[str] = None


class RebaseOperation:
    """Interactive rebase implementation"""
    
    def __init__(self, onto: str, upstream: str, branch: str):
        self.onto = onto
        self.upstream = upstream
        self.branch = branch
        self.todo_list: List[RebaseAction] = []
        self.state: Optional[RebaseState] = None
    
    def prepare_rebase(self) -> List[RebaseAction]:
        """Prepare rebase todo list"""
        # Find commits to rebase
        commits = self.find_commits_to_rebase()
        
        # Generate todo list
        for commit in commits:
            action = RebaseAction(RebaseCommand.PICK, commit.sha, commit.message)
            self.todo_list.append(action)
        
        return self.todo_list
    
    def execute_rebase(self) -> RebaseResult:
        """Execute rebase operations"""
        # Save original HEAD
        orig_head = self.get_head()
        
        # Initialize state
        self.state = RebaseState(
            onto=self.onto,
            upstream=self.upstream,
            branch=self.branch,
            todo_list=list(self.todo_list),
            done_list=[],
            current_action=None,
            orig_head=orig_head,
            head=self.onto,
            conflicts=[]
        )
        
        # Checkout onto commit
        self.checkout(self.onto)
        
        # Apply commits
        while self.state.todo_list:
            action = self.state.todo_list.pop(0)
            self.state.current_action = action
            
            result = self.apply_action(action)
            
            if result.conflict:
                # Save state and pause for conflict resolution
                self.state.conflicts = self.get_conflicted_files()
                self.state.save(".git")
                
                return RebaseResult(
                    status='conflict',
                    message=f"Conflict while applying {action.commit[:7]}",
                    conflicts=self.state.conflicts
                )
            
            if result.success:
                self.state.done_list.append(action)
                if result.new_commit:
                    self.state.head = result.new_commit
            else:
                # Error occurred
                return RebaseResult(
                    status='error',
                    message=f"Failed to apply {action.commit[:7]}"
                )
        
        # Update branch ref
        self.update_ref(self.branch, self.state.head)
        
        # Clean up state
        self.cleanup_rebase_state()
        
        return RebaseResult(status='success')
    
    def continue_rebase(self) -> RebaseResult:
        """Continue rebase after conflict resolution"""
        # Load state
        self.state = RebaseState.load(".git")
        if not self.state:
            return RebaseResult(status='error', message='No rebase in progress')
        
        # Check if conflicts are resolved
        if self.get_conflicted_files():
            return RebaseResult(
                status='error',
                message='Conflicts still present'
            )
        
        # Complete current action
        if self.state.current_action:
            # Commit the resolved changes
            new_commit = self.commit_resolved_changes(self.state.current_action)
            self.state.head = new_commit
            self.state.done_list.append(self.state.current_action)
        
        # Continue with remaining actions
        return self.execute_rebase()
    
    def abort_rebase(self) -> RebaseResult:
        """Abort rebase and restore original state"""
        # Load state
        self.state = RebaseState.load(".git")
        if not self.state:
            return RebaseResult(status='error', message='No rebase in progress')
        
        # Restore original HEAD
        self.checkout(self.state.orig_head)
        self.update_ref(self.branch, self.state.orig_head)
        
        # Clean up
        self.cleanup_rebase_state()
        
        return RebaseResult(status='success', message='Rebase aborted')
    
    def apply_action(self, action: RebaseAction) -> ActionResult:
        """Apply single rebase action"""
        if action.command == RebaseCommand.PICK:
            return self.cherry_pick(action.commit)
            
        elif action.command == RebaseCommand.REWORD:
            result = self.cherry_pick(action.commit)
            if result.success and result.new_commit:
                self.amend_commit_message(result.new_commit, action.new_message)
            return result
            
        elif action.command == RebaseCommand.EDIT:
            result = self.cherry_pick(action.commit)
            if result.success:
                # Pause for editing
                self.state.save(".git")
                print(f"Stopped at {action.commit[:7]}... {action.message}")
                print("You can amend the commit now")
            return result
            
        elif action.command == RebaseCommand.SQUASH:
            return self.squash_with_previous(action.commit)
            
        elif action.command == RebaseCommand.FIXUP:
            return self.squash_with_previous(action.commit, keep_message=False)
            
        elif action.command == RebaseCommand.DROP:
            # Skip this commit
            return ActionResult(success=True)
            
        elif action.command == RebaseCommand.EXEC:
            # Execute shell command
            success = self.execute_command(action.exec_command)
            return ActionResult(success=success)
            
        else:
            raise ValueError(f"Unknown rebase command: {action.command}")
    
    def cherry_pick(self, commit_sha: str) -> ActionResult:
        """Cherry-pick a commit"""
        # Get commit object
        commit = self.get_commit(commit_sha)
        
        # Get parent tree
        parent = self.get_commit(commit.parents[0]) if commit.parents else None
        parent_tree = parent.tree if parent else None
        
        # Get current tree
        current_tree = self.get_tree("HEAD")
        
        # Perform three-way merge
        merge_result = self.three_way_merge(parent_tree, current_tree, commit.tree)
        
        if merge_result.conflicts:
            return ActionResult(success=False, conflict=True)
        
        # Create new commit
        new_commit = self.create_commit(
            tree=merge_result.tree,
            parents=[self.get_head()],
            message=commit.message,
            author=commit.author
        )
        
        return ActionResult(success=True, new_commit=new_commit)
    
    def squash_with_previous(self, commit_sha: str, 
                           keep_message: bool = True) -> ActionResult:
        """Squash commit with previous"""
        if not self.state.done_list:
            # No previous commit to squash with
            return self.cherry_pick(commit_sha)
        
        # Get the commit to squash
        commit = self.get_commit(commit_sha)
        
        # Apply changes without creating commit
        result = self.apply_changes(commit)
        if not result.success:
            return result
        
        # Amend previous commit
        prev_commit = self.get_commit(self.state.head)
        
        if keep_message:
            # Combine messages
            combined_message = f"{prev_commit.message}\n\n{commit.message}"
        else:
            # Keep only previous message
            combined_message = prev_commit.message
        
        # Amend commit
        self.amend_commit(self.state.head, message=combined_message)
        
        return ActionResult(success=True, new_commit=self.state.head)
    
    def find_commits_to_rebase(self) -> List['Commit']:
        """Find commits to rebase"""
        # Find commits between upstream and branch
        commits = []
        
        # Simplified - would use git rev-list
        # git rev-list upstream..branch
        
        # Mock implementation
        for i in range(3):
            commits.append(Commit(
                sha=f"commit{i}",
                message=f"Commit {i}",
                parents=[f"parent{i}"],
                tree=None,
                author="Developer"
            ))
        
        return commits
    
    # Helper methods (simplified implementations)
    def get_head(self) -> str:
        """Get current HEAD commit"""
        return "HEAD"
    
    def checkout(self, ref: str):
        """Checkout a commit/branch"""
        pass
    
    def update_ref(self, ref: str, commit: str):
        """Update reference to point to commit"""
        pass
    
    def get_commit(self, sha: str) -> 'Commit':
        """Get commit object"""
        return Commit(sha, f"Message for {sha}", [], None, "Author")
    
    def get_tree(self, ref: str) -> 'Tree':
        """Get tree object for ref"""
        return None
    
    def three_way_merge(self, base, ours, theirs) -> 'MergeResult':
        """Perform three-way merge"""
        # Simplified - would use merge algorithm
        from dataclasses import dataclass
        
        @dataclass
        class MergeResult:
            tree: any
            conflicts: List[str]
        
        return MergeResult(tree=None, conflicts=[])
    
    def create_commit(self, tree, parents, message, author) -> str:
        """Create new commit"""
        return f"new-commit-{len(message)}"
    
    def get_conflicted_files(self) -> List[str]:
        """Get list of conflicted files"""
        return []
    
    def cleanup_rebase_state(self):
        """Remove rebase state directory"""
        import shutil
        rebase_dir = os.path.join(".git", "rebase-merge")
        if os.path.exists(rebase_dir):
            shutil.rmtree(rebase_dir)
    
    def amend_commit_message(self, commit: str, new_message: str):
        """Amend commit message"""
        pass
    
    def execute_command(self, command: str) -> bool:
        """Execute shell command"""
        import subprocess
        result = subprocess.run(command, shell=True)
        return result.returncode == 0
    
    def apply_changes(self, commit) -> ActionResult:
        """Apply commit changes without committing"""
        return ActionResult(success=True)
    
    def amend_commit(self, commit: str, message: str):
        """Amend existing commit"""
        pass
    
    def commit_resolved_changes(self, action: RebaseAction) -> str:
        """Commit changes after conflict resolution"""
        return f"resolved-{action.commit[:7]}"


@dataclass
class Commit:
    """Simple commit representation"""
    sha: str
    message: str
    parents: List[str]
    tree: Optional['Tree']
    author: str


class BisectAlgorithm:
    """Binary search for bug introduction"""
    
    def __init__(self, good: str, bad: str):
        self.good_commits = {good}
        self.bad_commits = {bad}
        self.tested_commits = set()
        self.skip_commits = set()
        self.commit_graph: Optional['CommitGraph'] = None
    
    def start(self, commit_graph: 'CommitGraph'):
        """Start bisect operation"""
        self.commit_graph = commit_graph
        
        # Verify good is ancestor of bad
        if not commit_graph.is_ancestor(self.good_commits.pop(), 
                                      self.bad_commits.pop()):
            raise ValueError("Good commit is not ancestor of bad commit")
        
        self.good_commits = {list(self.good_commits)[0]}
        self.bad_commits = {list(self.bad_commits)[0]}
    
    def next_commit(self) -> Optional[str]:
        """Find next commit to test using binary search"""
        if not self.commit_graph:
            return None
        
        # Find untested commits
        untested = self.find_untested_commits()
        if not untested:
            return None
        
        # Weight commits by bisection efficiency
        weights = self.calculate_commit_weights(untested)
        
        # Select commit that best bisects the graph
        return self.select_optimal_commit(weights)
    
    def mark_good(self, commit: str):
        """Mark commit as good"""
        self.good_commits.add(commit)
        self.tested_commits.add(commit)
        
        # All ancestors are also good
        ancestors = self.commit_graph.get_ancestors(commit)
        self.good_commits.update(ancestors)
    
    def mark_bad(self, commit: str):
        """Mark commit as bad"""
        self.bad_commits.add(commit)
        self.tested_commits.add(commit)
        
        # All descendants are also bad
        descendants = self.commit_graph.get_descendants(commit)
        self.bad_commits.update(descendants)
    
    def mark_skip(self, commit: str):
        """Skip commit (untestable)"""
        self.skip_commits.add(commit)
        self.tested_commits.add(commit)
    
    def get_result(self) -> Optional[str]:
        """Get the first bad commit"""
        # Find bad commits with only good parents
        first_bad = None
        
        for bad in self.bad_commits:
            parents = self.commit_graph.get_parents(bad)
            if all(p in self.good_commits for p in parents):
                if not first_bad or self.is_better_candidate(bad, first_bad):
                    first_bad = bad
        
        return first_bad
    
    def find_untested_commits(self) -> Set[str]:
        """Find commits that haven't been tested"""
        # Get all commits between good and bad
        all_commits = self.commit_graph.get_commits_between(
            self.good_commits, self.bad_commits
        )
        
        # Remove tested and skipped commits
        untested = all_commits - self.tested_commits - self.skip_commits
        
        # Remove commits we can infer
        untested -= self.good_commits
        untested -= self.bad_commits
        
        return untested
    
    def calculate_commit_weights(self, commits: Set[str]) -> Dict[str, float]:
        """Calculate bisection efficiency for each commit"""
        weights = {}
        
        total_untested = len(self.find_untested_commits())
        
        for commit in commits:
            # Simulate marking this commit as good/bad
            # Count how many commits would be eliminated
            
            # If marked good
            good_ancestors = len(self.commit_graph.get_ancestors(commit))
            
            # If marked bad  
            bad_descendants = len(self.commit_graph.get_descendants(commit))
            
            # Optimal weight minimizes the maximum remaining commits
            # This is the worst-case scenario
            worst_case = max(
                total_untested - good_ancestors,
                total_untested - bad_descendants
            )
            
            # Lower weight is better
            weights[commit] = worst_case
        
        return weights
    
    def select_optimal_commit(self, weights: Dict[str, float]) -> str:
        """Select commit with best bisection properties"""
        if not weights:
            return None
        
        # Find commits with minimum weight
        min_weight = min(weights.values())
        candidates = [c for c, w in weights.items() if w == min_weight]
        
        # If multiple candidates, prefer merge commits
        merge_commits = [c for c in candidates 
                        if len(self.commit_graph.get_parents(c)) > 1]
        
        if merge_commits:
            return merge_commits[0]
        
        # Otherwise, return any candidate
        return candidates[0]
    
    def is_better_candidate(self, commit1: str, commit2: str) -> bool:
        """Compare two commits for first bad commit"""
        # Prefer commit with fewer untested ancestors
        ancestors1 = self.commit_graph.get_ancestors(commit1)
        ancestors2 = self.commit_graph.get_ancestors(commit2)
        
        untested1 = len(ancestors1 - self.tested_commits)
        untested2 = len(ancestors2 - self.tested_commits)
        
        return untested1 < untested2
    
    def estimate_steps_remaining(self) -> int:
        """Estimate number of bisection steps remaining"""
        untested = len(self.find_untested_commits())
        if untested <= 1:
            return untested
        
        # Binary search: log2(n) steps
        import math
        return int(math.ceil(math.log2(untested)))


class CommitGraph:
    """Simple commit graph for bisect"""
    
    def __init__(self):
        self.commits: Dict[str, Set[str]] = {}  # commit -> parents
        self.children: Dict[str, Set[str]] = {}  # commit -> children
    
    def add_commit(self, commit: str, parents: List[str]):
        """Add commit to graph"""
        self.commits[commit] = set(parents)
        
        # Update children mapping
        if commit not in self.children:
            self.children[commit] = set()
        
        for parent in parents:
            if parent not in self.children:
                self.children[parent] = set()
            self.children[parent].add(commit)
    
    def get_parents(self, commit: str) -> Set[str]:
        """Get immediate parents of commit"""
        return self.commits.get(commit, set())
    
    def get_ancestors(self, commit: str) -> Set[str]:
        """Get all ancestors of commit"""
        ancestors = set()
        to_visit = deque([commit])
        
        while to_visit:
            current = to_visit.popleft()
            if current in ancestors:
                continue
            
            ancestors.add(current)
            parents = self.get_parents(current)
            to_visit.extend(parents)
        
        ancestors.remove(commit)  # Don't include commit itself
        return ancestors
    
    def get_descendants(self, commit: str) -> Set[str]:
        """Get all descendants of commit"""
        descendants = set()
        to_visit = deque([commit])
        
        while to_visit:
            current = to_visit.popleft()
            if current in descendants:
                continue
            
            descendants.add(current)
            children = self.children.get(current, set())
            to_visit.extend(children)
        
        descendants.remove(commit)  # Don't include commit itself
        return descendants
    
    def is_ancestor(self, potential_ancestor: str, commit: str) -> bool:
        """Check if one commit is ancestor of another"""
        ancestors = self.get_ancestors(commit)
        return potential_ancestor in ancestors
    
    def get_commits_between(self, good_commits: Set[str], 
                          bad_commits: Set[str]) -> Set[str]:
        """Get all commits between good and bad"""
        # Find all ancestors of bad commits
        bad_ancestors = set()
        for bad in bad_commits:
            bad_ancestors.add(bad)
            bad_ancestors.update(self.get_ancestors(bad))
        
        # Find all ancestors of good commits  
        good_ancestors = set()
        for good in good_commits:
            good_ancestors.add(good)
            good_ancestors.update(self.get_ancestors(good))
        
        # Commits between are in bad ancestors but not good ancestors
        between = bad_ancestors - good_ancestors
        
        return between


# Example usage
def demo_rebase():
    """Demonstrate rebase operation"""
    print("Interactive Rebase Demo")
    print("-" * 40)
    
    # Create rebase operation
    rebase = RebaseOperation(
        onto="main",
        upstream="origin/main", 
        branch="feature/my-feature"
    )
    
    # Prepare todo list
    todo_list = rebase.prepare_rebase()
    
    print("Rebase todo list:")
    for i, action in enumerate(todo_list):
        print(f"{i+1}. {action.command.value} {action.commit[:7]} {action.message}")
    
    # Simulate editing todo list
    if len(todo_list) > 2:
        # Change second commit to squash
        todo_list[1].command = RebaseCommand.SQUASH
        
        # Drop third commit
        todo_list[2].command = RebaseCommand.DROP
    
    print("\nModified todo list:")
    for i, action in enumerate(todo_list):
        print(f"{i+1}. {action.command.value} {action.commit[:7]} {action.message}")
    
    # Execute rebase
    result = rebase.execute_rebase()
    print(f"\nRebase result: {result.status}")
    if result.message:
        print(f"Message: {result.message}")


def demo_bisect():
    """Demonstrate bisect algorithm"""
    print("\nBinary Search (Bisect) Demo")
    print("-" * 40)
    
    # Create commit graph
    graph = CommitGraph()
    
    # Linear history with bug introduced
    commits = []
    for i in range(10):
        commit = f"commit{i}"
        parent = [commits[-1]] if commits else []
        graph.add_commit(commit, parent)
        commits.append(commit)
    
    # Initialize bisect
    bisect = BisectAlgorithm(good=commits[2], bad=commits[8])
    bisect.start(graph)
    
    print(f"Good commit: {commits[2]}")
    print(f"Bad commit: {commits[8]}")
    print(f"Commits to test: {len(bisect.find_untested_commits())}")
    
    # Simulate bisection
    bug_commit = commits[6]  # Bug introduced here
    steps = 0
    
    while True:
        next_commit = bisect.next_commit()
        if not next_commit:
            break
        
        steps += 1
        commit_idx = commits.index(next_commit)
        
        print(f"\nStep {steps}: Testing {next_commit}")
        
        # Simulate testing
        if commit_idx >= commits.index(bug_commit):
            print(f"  Result: BAD")
            bisect.mark_bad(next_commit)
        else:
            print(f"  Result: GOOD")
            bisect.mark_good(next_commit)
        
        remaining = bisect.estimate_steps_remaining()
        print(f"  Estimated steps remaining: {remaining}")
    
    # Get result
    first_bad = bisect.get_result()
    print(f"\nFirst bad commit: {first_bad}")
    print(f"Actual bug commit: {bug_commit}")
    print(f"Found in {steps} steps")


if __name__ == "__main__":
    demo_rebase()
    demo_bisect()