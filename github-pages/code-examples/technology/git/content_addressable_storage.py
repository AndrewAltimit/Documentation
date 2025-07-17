"""
Content-Addressable Storage Theory and Git Object Model

Implementation of Git's fundamental data structures including:
- SHA-1 content addressing
- Object types (blob, tree, commit, tag)
- Merkle tree for directories
- Directed Acyclic Graph (DAG) for commit history
"""

import hashlib
import zlib
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class ObjectType(Enum):
    BLOB = "blob"
    TREE = "tree"
    COMMIT = "commit"
    TAG = "tag"


@dataclass
class GitObject:
    """Base class for Git objects with content addressing"""

    type: ObjectType
    content: bytes

    @property
    def sha1(self) -> str:
        """Compute SHA-1 hash of object"""
        header = f"{self.type.value} {len(self.content)}\0".encode()
        return hashlib.sha1(header + self.content).hexdigest()

    def compress(self) -> bytes:
        """Compress object for storage"""
        header = f"{self.type.value} {len(self.content)}\0".encode()
        return zlib.compress(header + self.content)


class MerkleTree:
    """Git's tree object implementation as a Merkle tree"""

    def __init__(self):
        self.entries: List[Tuple[str, str, str]] = []  # mode, sha, name

    def add_entry(self, mode: str, sha: str, name: str):
        """Add file or subtree entry"""
        self.entries.append((mode, sha, name))

    def compute_tree_hash(self) -> str:
        """Compute tree object hash"""
        # Sort entries by name for canonical form
        sorted_entries = sorted(self.entries, key=lambda x: x[2])

        content = b""
        for mode, sha, name in sorted_entries:
            # Format: <mode> <name>\0<20-byte sha>
            entry = f"{mode} {name}\0".encode()
            entry += bytes.fromhex(sha)
            content += entry

        tree_obj = GitObject(ObjectType.TREE, content)
        return tree_obj.sha1


@dataclass
class Commit:
    """Git commit object representation"""

    tree: str
    parents: List[str]
    author: str
    committer: str
    message: str

    def compute_hash(self) -> str:
        """Compute commit object hash"""
        content = f"tree {self.tree}\n"
        for parent in self.parents:
            content += f"parent {parent}\n"
        content += f"author {self.author}\n"
        content += f"committer {self.committer}\n"
        content += f"\n{self.message}"

        commit_obj = GitObject(ObjectType.COMMIT, content.encode())
        return commit_obj.sha1


class CommitDAG:
    """Directed Acyclic Graph for commit history"""

    def __init__(self):
        self.commits: Dict[str, Commit] = {}
        self.refs: Dict[str, str] = {}  # branch/tag -> commit sha

    def add_commit(self, commit: Commit) -> str:
        """Add commit to DAG"""
        sha = commit.compute_hash()
        self.commits[sha] = commit
        return sha

    def find_common_ancestor(self, sha1: str, sha2: str) -> Optional[str]:
        """Find merge base using BFS"""
        visited1, visited2 = set(), set()
        queue1, queue2 = deque([sha1]), deque([sha2])

        while queue1 or queue2:
            if queue1:
                current = queue1.popleft()
                if current in visited2:
                    return current
                visited1.add(current)

                commit = self.commits.get(current)
                if commit:
                    queue1.extend(p for p in commit.parents if p not in visited1)

            if queue2:
                current = queue2.popleft()
                if current in visited1:
                    return current
                visited2.add(current)

                commit = self.commits.get(current)
                if commit:
                    queue2.extend(p for p in commit.parents if p not in visited2)

        return None

    def find_all_ancestors(self, sha: str) -> set:
        """Find all ancestors of a commit"""
        ancestors = set()
        to_visit = deque([sha])

        while to_visit:
            current = to_visit.popleft()
            if current in ancestors:
                continue

            ancestors.add(current)
            commit = self.commits.get(current)
            if commit:
                to_visit.extend(commit.parents)

        return ancestors

    def is_ancestor(self, potential_ancestor: str, descendant: str) -> bool:
        """Check if one commit is an ancestor of another"""
        ancestors = self.find_all_ancestors(descendant)
        return potential_ancestor in ancestors

    def find_merge_bases(self, *commits: str) -> List[str]:
        """Find all merge bases for multiple commits"""
        if len(commits) < 2:
            return list(commits)

        # Find ancestors for each commit
        ancestor_sets = [self.find_all_ancestors(commit) for commit in commits]

        # Find common ancestors
        common_ancestors = ancestor_sets[0]
        for ancestor_set in ancestor_sets[1:]:
            common_ancestors &= ancestor_set

        # Find merge bases (common ancestors with no descendants in common ancestors)
        merge_bases = []
        for candidate in common_ancestors:
            is_merge_base = True
            for other in common_ancestors:
                if candidate != other and self.is_ancestor(candidate, other):
                    is_merge_base = False
                    break

            if is_merge_base:
                merge_bases.append(candidate)

        return merge_bases

    def topological_sort(self, commits: Optional[List[str]] = None) -> List[str]:
        """Return commits in topological order (parents before children)"""
        if commits is None:
            commits = list(self.commits.keys())

        # Build in-degree map
        in_degree = {commit: 0 for commit in commits}
        children_map = {commit: [] for commit in commits}

        for commit_sha in commits:
            commit = self.commits.get(commit_sha)
            if commit:
                for parent in commit.parents:
                    if parent in in_degree:
                        children_map[parent].append(commit_sha)
                        in_degree[commit_sha] += 1

        # Find commits with no dependencies
        queue = deque([commit for commit in commits if in_degree[commit] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Process children
            for child in children_map[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def find_divergence_point(self, branch1: str, branch2: str) -> Optional[str]:
        """Find where two branches diverged"""
        merge_base = self.find_common_ancestor(branch1, branch2)
        return merge_base


# Example usage and testing
def demo_git_objects():
    """Demonstrate Git object model"""
    # Create blob
    file_content = b"Hello, Git!"
    blob = GitObject(ObjectType.BLOB, file_content)
    print(f"Blob SHA-1: {blob.sha1}")

    # Create tree
    tree = MerkleTree()
    tree.add_entry("100644", blob.sha1, "hello.txt")
    tree.add_entry("100644", "a" * 40, "world.txt")
    tree_hash = tree.compute_tree_hash()
    print(f"Tree SHA-1: {tree_hash}")

    # Create commit DAG
    dag = CommitDAG()

    # Initial commit
    initial_commit = Commit(
        tree=tree_hash,
        parents=[],
        author="Alice <alice@example.com> 1234567890 +0000",
        committer="Alice <alice@example.com> 1234567890 +0000",
        message="Initial commit",
    )
    initial_sha = dag.add_commit(initial_commit)

    # Feature branch commit
    feature_commit = Commit(
        tree=tree_hash,
        parents=[initial_sha],
        author="Bob <bob@example.com> 1234567891 +0000",
        committer="Bob <bob@example.com> 1234567891 +0000",
        message="Add feature X",
    )
    feature_sha = dag.add_commit(feature_commit)

    # Main branch commit
    main_commit = Commit(
        tree=tree_hash,
        parents=[initial_sha],
        author="Alice <alice@example.com> 1234567892 +0000",
        committer="Alice <alice@example.com> 1234567892 +0000",
        message="Update documentation",
    )
    main_sha = dag.add_commit(main_commit)

    # Find common ancestor
    merge_base = dag.find_common_ancestor(feature_sha, main_sha)
    print(f"Merge base: {merge_base}")
    print(f"Is initial commit the merge base? {merge_base == initial_sha}")


if __name__ == "__main__":
    demo_git_objects()
