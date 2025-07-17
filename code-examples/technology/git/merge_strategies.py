"""
Advanced Git Merge Strategies

Implementation of various merge strategies including:
- Recursive merge (default)
- Octopus merge (multiple branches)
- Ours/Theirs strategies
- Subtree merge
- Custom merge drivers
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Commit:
    """Simple commit representation"""

    sha: str
    tree: "Tree"
    parents: List[str]
    message: str
    ref: Optional[str] = None


@dataclass
class Tree:
    """Tree object representation"""

    entries: Dict[str, "TreeEntry"]


@dataclass
class TreeEntry:
    """Entry in a tree (file or subtree)"""

    mode: str
    sha: str
    name: str
    content: Optional[bytes] = None


@dataclass
class MergeResult:
    """Result of a merge operation"""

    success: bool
    tree: Optional[Tree] = None
    commit: Optional[Commit] = None
    conflicts: List["Conflict"] = None
    message: Optional[str] = None


@dataclass
class Conflict:
    """Represents a merge conflict"""

    path: str
    base_content: Optional[bytes]
    our_content: Optional[bytes]
    their_content: Optional[bytes]


class MergeStrategy(ABC):
    """Base class for merge strategies"""

    @abstractmethod
    def merge(self, base: Commit, ours: Commit, theirs: Commit) -> MergeResult:
        """Perform merge operation"""
        pass

    def find_merge_bases(self, commits: List[Commit]) -> List[Commit]:
        """Find merge bases for commits"""
        # Simplified - would use CommitDAG in practice
        if not commits:
            return []

        # Find common ancestors
        ancestor_sets = []
        for commit in commits:
            ancestors = self._get_all_ancestors(commit)
            ancestor_sets.append(ancestors)

        # Find intersection
        common = ancestor_sets[0]
        for ancestors in ancestor_sets[1:]:
            common &= ancestors

        # Find merge bases (maximal elements)
        merge_bases = []
        for candidate in common:
            is_merge_base = True
            for other in common:
                if candidate != other and self._is_ancestor(candidate, other):
                    is_merge_base = False
                    break
            if is_merge_base:
                merge_bases.append(candidate)

        return merge_bases

    def _get_all_ancestors(self, commit: Commit) -> Set[str]:
        """Get all ancestors of a commit"""
        # Simplified implementation
        ancestors = {commit.sha}
        for parent in commit.parents:
            ancestors.add(parent)
        return ancestors

    def _is_ancestor(self, potential_ancestor: str, descendant: str) -> bool:
        """Check if one commit is ancestor of another"""
        # Simplified - would check actual DAG
        return False


class RecursiveMergeStrategy(MergeStrategy):
    """Git's default recursive merge strategy"""

    def merge(self, base: Commit, ours: Commit, theirs: Commit) -> MergeResult:
        """Perform recursive three-way merge"""
        # Handle multiple merge bases
        merge_bases = self.find_merge_bases([ours, theirs])

        if len(merge_bases) > 1:
            # Recursively merge the merge bases
            virtual_base = self._merge_bases(merge_bases)
            return self._three_way_merge(virtual_base, ours, theirs)
        elif merge_bases:
            # Simple three-way merge
            return self._three_way_merge(merge_bases[0], ours, theirs)
        else:
            # No common base - treat as new branches
            return self._merge_unrelated(ours, theirs)

    def _three_way_merge(
        self, base: Commit, ours: Commit, theirs: Commit
    ) -> MergeResult:
        """Perform three-way merge on trees"""
        result = MergeResult(success=True, conflicts=[])

        # Get trees
        base_tree = base.tree if base else Tree(entries={})
        our_tree = ours.tree
        their_tree = theirs.tree

        # Merge trees recursively
        merged_tree = self._merge_trees(base_tree, our_tree, their_tree, result)

        if not result.conflicts:
            # Create merge commit
            merge_commit = Commit(
                sha=self._compute_sha(merged_tree),
                tree=merged_tree,
                parents=[ours.sha, theirs.sha],
                message=f"Merge {theirs.ref or theirs.sha[:7]} into {ours.ref or ours.sha[:7]}",
            )
            result.commit = merge_commit
            result.tree = merged_tree
        else:
            result.success = False

        return result

    def _merge_trees(
        self, base: Tree, ours: Tree, theirs: Tree, result: MergeResult
    ) -> Tree:
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

            merged_entry = self._merge_entry(
                path, base_entry, our_entry, their_entry, result
            )

            if merged_entry:
                merged_entries[path] = merged_entry

        return Tree(entries=merged_entries)

    def _merge_entry(
        self,
        path: str,
        base: Optional[TreeEntry],
        ours: Optional[TreeEntry],
        theirs: Optional[TreeEntry],
        result: MergeResult,
    ) -> Optional[TreeEntry]:
        """Merge a single file/directory entry"""
        # Case 1: Same in both branches
        if self._entries_equal(ours, theirs):
            return ours

        # Case 2: Changed only in our branch
        if self._entries_equal(base, theirs):
            return ours

        # Case 3: Changed only in their branch
        if self._entries_equal(base, ours):
            return theirs

        # Case 4: Added in both branches with same content
        if base is None and ours and theirs:
            if ours.sha == theirs.sha:
                return ours

        # Case 5: Deleted in one branch
        if ours is None or theirs is None:
            if base is not None:
                # Modified in one, deleted in other - conflict
                conflict = Conflict(
                    path=path,
                    base_content=base.content if base else None,
                    our_content=ours.content if ours else None,
                    their_content=theirs.content if theirs else None,
                )
                result.conflicts.append(conflict)
                return None
            else:
                # Added in one branch only
                return ours or theirs

        # Case 6: Both modified differently - try content merge
        if ours.mode == theirs.mode and ours.mode == "100644":
            # Regular files - attempt three-way merge
            merged_content = self._merge_file_content(
                base.content if base else b"", ours.content, theirs.content
            )

            if merged_content is not None:
                return TreeEntry(
                    mode=ours.mode,
                    sha=self._compute_sha(merged_content),
                    name=ours.name,
                    content=merged_content,
                )

        # Case 7: Conflict
        conflict = Conflict(
            path=path,
            base_content=base.content if base else None,
            our_content=ours.content if ours else None,
            their_content=theirs.content if theirs else None,
        )
        result.conflicts.append(conflict)
        return None

    def _merge_bases(self, bases: List[Commit]) -> Commit:
        """Merge multiple merge bases into virtual base"""
        if len(bases) == 1:
            return bases[0]

        # Recursively merge bases pairwise
        merged = bases[0]
        for base in bases[1:]:
            result = self._three_way_merge(None, merged, base)
            if result.success:
                merged = result.commit
            else:
                # Conflict in merge bases - use first
                return bases[0]

        return merged

    def _entries_equal(self, e1: Optional[TreeEntry], e2: Optional[TreeEntry]) -> bool:
        """Check if two entries are equal"""
        if e1 is None and e2 is None:
            return True
        if e1 is None or e2 is None:
            return False
        return e1.sha == e2.sha and e1.mode == e2.mode

    def _merge_file_content(
        self, base: bytes, ours: bytes, theirs: bytes
    ) -> Optional[bytes]:
        """Attempt to merge file content"""
        # Simplified - would use three-way merge algorithm
        if ours == theirs:
            return ours
        return None

    def _compute_sha(self, content) -> str:
        """Compute SHA for content"""
        import hashlib

        return hashlib.sha1(str(content).encode()).hexdigest()

    def _merge_unrelated(self, ours: Commit, theirs: Commit) -> MergeResult:
        """Merge unrelated histories"""
        # Combine trees at root level
        merged_entries = {}
        merged_entries.update(ours.tree.entries)

        # Add their entries, checking for conflicts
        conflicts = []
        for path, entry in theirs.tree.entries.items():
            if path in merged_entries:
                if merged_entries[path].sha != entry.sha:
                    conflicts.append(
                        Conflict(
                            path=path,
                            base_content=None,
                            our_content=merged_entries[path].content,
                            their_content=entry.content,
                        )
                    )
            else:
                merged_entries[path] = entry

        if conflicts:
            return MergeResult(success=False, conflicts=conflicts)

        merged_tree = Tree(entries=merged_entries)
        merge_commit = Commit(
            sha=self._compute_sha(merged_tree),
            tree=merged_tree,
            parents=[ours.sha, theirs.sha],
            message=f"Merge {theirs.sha[:7]} into {ours.sha[:7]}",
        )

        return MergeResult(success=True, commit=merge_commit, tree=merged_tree)


class OctopusMergeStrategy(MergeStrategy):
    """Strategy for merging multiple branches simultaneously"""

    def merge(self, base: Commit, ours: Commit, theirs: Commit) -> MergeResult:
        """Not used for octopus - see merge_multiple"""
        raise NotImplementedError("Use merge_multiple for octopus merge")

    def merge_multiple(self, head: Commit, branches: List[Commit]) -> MergeResult:
        """Merge multiple branches simultaneously"""
        # Find common base for all branches
        all_commits = [head] + branches
        common_base = self._find_common_base_all(all_commits)

        if not common_base:
            return MergeResult(
                success=False, message="No common base found for octopus merge"
            )

        # Build multi-way merge tree
        merge_tree = self._multi_way_merge(common_base, all_commits)

        if merge_tree is None:
            return MergeResult(
                success=False,
                message="Conflicts detected - cannot perform octopus merge",
            )

        # Create octopus merge commit
        parent_shas = [head.sha] + [b.sha for b in branches]
        branch_names = [b.ref or b.sha[:7] for b in branches]

        merge_commit = Commit(
            sha=self._compute_sha(merge_tree),
            tree=merge_tree,
            parents=parent_shas,
            message=f"Merge branches {', '.join(branch_names)}",
        )

        return MergeResult(success=True, commit=merge_commit, tree=merge_tree)

    def _find_common_base_all(self, commits: List[Commit]) -> Optional[Commit]:
        """Find common base for all commits"""
        if not commits:
            return None

        # Find intersection of all ancestors
        ancestor_sets = [self._get_all_ancestors(c) for c in commits]
        common = ancestor_sets[0]

        for ancestors in ancestor_sets[1:]:
            common &= ancestors

        if not common:
            return None

        # Return most recent common ancestor
        # Simplified - would use commit dates
        return Commit(
            sha=list(common)[0],
            tree=Tree(entries={}),
            parents=[],
            message="Common base",
        )

    def _multi_way_merge(self, base: Commit, branches: List[Commit]) -> Optional[Tree]:
        """Perform multi-way merge"""
        # Start with base tree
        result_tree = Tree(entries=dict(base.tree.entries))

        # Apply changes from each branch
        for branch in branches:
            changes = self._compute_changes(base.tree, branch.tree)

            # Check for conflicts
            for path, change in changes.items():
                if path in result_tree.entries:
                    if result_tree.entries[path].sha != change.sha:
                        # Conflict - cannot do octopus merge
                        return None
                else:
                    result_tree.entries[path] = change

        return result_tree

    def _compute_changes(self, base: Tree, branch: Tree) -> Dict[str, TreeEntry]:
        """Compute changes between base and branch"""
        changes = {}

        for path, entry in branch.entries.items():
            base_entry = base.entries.get(path)
            if not base_entry or base_entry.sha != entry.sha:
                changes[path] = entry

        return changes

    def _compute_sha(self, content) -> str:
        """Compute SHA for content"""
        import hashlib

        return hashlib.sha1(str(content).encode()).hexdigest()


class SubtreeMergeStrategy(MergeStrategy):
    """Merge as a subtree of the current repository"""

    def __init__(self, prefix: str):
        self.prefix = prefix.rstrip("/") + "/"

    def merge(self, base: Commit, ours: Commit, theirs: Commit) -> MergeResult:
        """Merge their tree as subtree of ours"""
        # Rewrite their tree with prefix
        their_subtree = self._prefix_tree(theirs.tree, self.prefix)

        # Create synthetic base at prefix
        base_subtree = None
        if base:
            base_subtree = self._extract_subtree(base.tree, self.prefix)

        # Perform three-way merge
        recursive = RecursiveMergeStrategy()

        # Create temporary commits for merge
        if base_subtree:
            base_commit = Commit("base", base_subtree, [], "Base")
        else:
            base_commit = None

        our_commit = Commit(ours.sha, ours.tree, ours.parents, ours.message)
        their_commit = Commit(theirs.sha, their_subtree, theirs.parents, theirs.message)

        result = recursive._three_way_merge(base_commit, our_commit, their_commit)

        if result.success:
            # Adjust commit message
            result.commit.message = f"Merge {theirs.sha[:7]} as '{self.prefix}'"

        return result

    def _prefix_tree(self, tree: Tree, prefix: str) -> Tree:
        """Add prefix to all paths in tree"""
        prefixed_entries = {}

        for path, entry in tree.entries.items():
            new_path = prefix + path
            new_entry = TreeEntry(
                mode=entry.mode, sha=entry.sha, name=entry.name, content=entry.content
            )
            prefixed_entries[new_path] = new_entry

        return Tree(entries=prefixed_entries)

    def _extract_subtree(self, tree: Tree, prefix: str) -> Tree:
        """Extract subtree at prefix"""
        subtree_entries = {}

        for path, entry in tree.entries.items():
            if path.startswith(prefix):
                new_path = path[len(prefix) :]
                subtree_entries[new_path] = entry

        return Tree(entries=subtree_entries)


class OursStrategy(MergeStrategy):
    """Always use our version (fake merge)"""

    def merge(self, base: Commit, ours: Commit, theirs: Commit) -> MergeResult:
        """Create merge commit but keep our tree"""
        merge_commit = Commit(
            sha=self._compute_sha(ours.tree),
            tree=ours.tree,
            parents=[ours.sha, theirs.sha],
            message=f"Merge {theirs.sha[:7]} (ours strategy)",
        )

        return MergeResult(success=True, commit=merge_commit, tree=ours.tree)

    def _compute_sha(self, content) -> str:
        import hashlib

        return hashlib.sha1(str(content).encode()).hexdigest()


# Example usage
def demo_merge_strategies():
    """Demonstrate different merge strategies"""
    # Create sample commits
    base_tree = Tree(
        entries={
            "file1.txt": TreeEntry("100644", "a1", "file1.txt", b"Base content"),
            "file2.txt": TreeEntry("100644", "a2", "file2.txt", b"Shared file"),
        }
    )

    our_tree = Tree(
        entries={
            "file1.txt": TreeEntry("100644", "b1", "file1.txt", b"Our changes"),
            "file2.txt": TreeEntry("100644", "a2", "file2.txt", b"Shared file"),
            "file3.txt": TreeEntry("100644", "b3", "file3.txt", b"Our new file"),
        }
    )

    their_tree = Tree(
        entries={
            "file1.txt": TreeEntry("100644", "c1", "file1.txt", b"Their changes"),
            "file2.txt": TreeEntry("100644", "a2", "file2.txt", b"Shared file"),
            "file4.txt": TreeEntry("100644", "c4", "file4.txt", b"Their new file"),
        }
    )

    base = Commit("base123", base_tree, [], "Base commit")
    ours = Commit("ours456", our_tree, ["base123"], "Our commit", "main")
    theirs = Commit("theirs789", their_tree, ["base123"], "Their commit", "feature")

    # Test recursive merge
    print("Recursive Merge Strategy:")
    recursive = RecursiveMergeStrategy()
    result = recursive.merge(base, ours, theirs)

    if result.success:
        print("  Merge successful!")
        print(f"  Parents: {result.commit.parents}")
    else:
        print("  Merge failed with conflicts:")
        for conflict in result.conflicts:
            print(f"    - {conflict.path}")

    # Test ours strategy
    print("\nOurs Strategy:")
    ours_strategy = OursStrategy()
    result = ours_strategy.merge(base, ours, theirs)
    print(f"  Merge successful: {result.success}")
    print(f"  Tree unchanged: {result.tree == ours.tree}")


if __name__ == "__main__":
    demo_merge_strategies()
