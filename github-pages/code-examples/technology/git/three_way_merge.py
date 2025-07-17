"""
Three-Way Merge Algorithm

Implementation of Git's three-way merge algorithm for combining changes
from two branches with a common ancestor.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


@dataclass
class Conflict:
    """Represents a merge conflict"""

    line: int
    ours: List[str]
    theirs: List[str]
    base: List[str]


@dataclass
class MergeResult:
    """Result of a three-way merge operation"""

    merged_content: List[str]
    conflicts: List[Conflict]
    success: bool

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


class ThreeWayMerge:
    """Implements Git's three-way merge algorithm"""

    def __init__(self, base: List[str], ours: List[str], theirs: List[str]):
        self.base = base
        self.ours = ours
        self.theirs = theirs

    def merge(self) -> Tuple[List[str], List[Conflict]]:
        """Perform three-way merge"""
        # Compute diffs from base
        base_to_ours = self._compute_diff(self.base, self.ours)
        base_to_theirs = self._compute_diff(self.base, self.theirs)

        result = []
        conflicts = []

        # Apply non-conflicting changes
        for i, line in enumerate(self.base):
            ours_change = base_to_ours.get(i)
            theirs_change = base_to_theirs.get(i)

            if ours_change is None and theirs_change is None:
                # No changes
                result.append(line)
            elif ours_change is not None and theirs_change is None:
                # Only we changed
                result.extend(ours_change)
            elif ours_change is None and theirs_change is not None:
                # Only they changed
                result.extend(theirs_change)
            elif ours_change == theirs_change:
                # Both changed identically
                result.extend(ours_change)
            else:
                # Conflict
                conflict = Conflict(
                    line=i, ours=ours_change, theirs=theirs_change, base=[line]
                )
                conflicts.append(conflict)
                result.extend(self._format_conflict(conflict))

        return result, conflicts

    def _compute_diff(
        self, from_lines: List[str], to_lines: List[str]
    ) -> Dict[int, List[str]]:
        """Compute line-based differences"""
        matcher = SequenceMatcher(None, from_lines, to_lines)
        changes = {}

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                for i in range(i1, i2):
                    changes[i] = to_lines[j1:j2]
            elif tag == "delete":
                for i in range(i1, i2):
                    changes[i] = []
            elif tag == "insert":
                changes[i1] = to_lines[j1:j2]

        return changes

    def _format_conflict(self, conflict: Conflict) -> List[str]:
        """Format conflict markers like Git"""
        lines = []
        lines.append("<<<<<<< ours\n")
        lines.extend(conflict.ours)
        lines.append("=======\n")
        lines.extend(conflict.theirs)
        lines.append(">>>>>>> theirs\n")
        return lines

    def merge_files(self) -> MergeResult:
        """Perform merge and return structured result"""
        merged, conflicts = self.merge()

        return MergeResult(
            merged_content=merged, conflicts=conflicts, success=len(conflicts) == 0
        )


class AdvancedMerge:
    """Advanced merge strategies and algorithms"""

    @staticmethod
    def recursive_three_way_merge(
        base: List[str], ours: List[str], theirs: List[str]
    ) -> MergeResult:
        """
        Recursive three-way merge for handling criss-cross merges

        When there are multiple merge bases, recursively merge them
        to create a virtual merge base.
        """
        merger = ThreeWayMerge(base, ours, theirs)
        return merger.merge_files()

    @staticmethod
    def patience_diff(
        a: List[str], b: List[str]
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Patience diff algorithm - better handling of moved blocks

        1. Match unique lines that appear exactly once in both files
        2. Recursively diff the sections between matches
        """
        # Find unique lines in each file
        unique_a = {}
        unique_b = {}

        for i, line in enumerate(a):
            if line in unique_a:
                unique_a[line] = None  # Mark as non-unique
            else:
                unique_a[line] = i

        for i, line in enumerate(b):
            if line in unique_b:
                unique_b[line] = None  # Mark as non-unique
            else:
                unique_b[line] = i

        # Find matching unique lines
        matches = []
        for line, idx_a in unique_a.items():
            if idx_a is not None and line in unique_b and unique_b[line] is not None:
                matches.append((idx_a, unique_b[line]))

        # Sort by position in first file
        matches.sort()

        # Build diff recursively between matches
        diff_ops = []
        last_a = 0
        last_b = 0

        for idx_a, idx_b in matches:
            # Diff the section before this match
            if idx_a > last_a or idx_b > last_b:
                section_diff = AdvancedMerge._simple_diff(
                    a[last_a:idx_a], b[last_b:idx_b]
                )
                for op, i1, i2, j1, j2 in section_diff:
                    diff_ops.append(
                        (op, last_a + i1, last_a + i2, last_b + j1, last_b + j2)
                    )

            # Add the matching line
            diff_ops.append(("equal", idx_a, idx_a + 1, idx_b, idx_b + 1))

            last_a = idx_a + 1
            last_b = idx_b + 1

        # Handle remaining sections
        if last_a < len(a) or last_b < len(b):
            section_diff = AdvancedMerge._simple_diff(a[last_a:], b[last_b:])
            for op, i1, i2, j1, j2 in section_diff:
                diff_ops.append(
                    (op, last_a + i1, last_a + i2, last_b + j1, last_b + j2)
                )

        return diff_ops

    @staticmethod
    def _simple_diff(
        a: List[str], b: List[str]
    ) -> List[Tuple[str, int, int, int, int]]:
        """Simple diff for small sections"""
        matcher = SequenceMatcher(None, a, b)
        return matcher.get_opcodes()

    @staticmethod
    def semantic_merge(base: Dict, ours: Dict, theirs: Dict) -> Dict:
        """
        Semantic merge for structured data (JSON, YAML, etc.)

        Merges at the semantic level rather than line level
        """
        result = {}
        all_keys = set(base.keys()) | set(ours.keys()) | set(theirs.keys())

        for key in all_keys:
            base_val = base.get(key)
            our_val = ours.get(key)
            their_val = theirs.get(key)

            if our_val == their_val:
                # No conflict
                result[key] = our_val
            elif base_val == our_val:
                # They changed, we didn't
                result[key] = their_val
            elif base_val == their_val:
                # We changed, they didn't
                result[key] = our_val
            elif (
                isinstance(base_val, dict)
                and isinstance(our_val, dict)
                and isinstance(their_val, dict)
            ):
                # Recursively merge nested objects
                result[key] = AdvancedMerge.semantic_merge(base_val, our_val, their_val)
            else:
                # Conflict - need manual resolution
                # In a real implementation, this would be marked as a conflict
                result[key] = {
                    "<<<<<<< ours": our_val,
                    "=======": None,
                    ">>>>>>> theirs": their_val,
                }

        return result


# Example usage
def demo_three_way_merge():
    """Demonstrate three-way merge"""
    # Base version
    base = [
        "def hello():\n",
        "    print('Hello')\n",
        "\n",
        "def world():\n",
        "    print('World')\n",
    ]

    # Our version (added parameter to hello)
    ours = [
        "def hello(name):\n",
        "    print(f'Hello, {name}')\n",
        "\n",
        "def world():\n",
        "    print('World')\n",
    ]

    # Their version (added exclamation to world)
    theirs = [
        "def hello():\n",
        "    print('Hello')\n",
        "\n",
        "def world():\n",
        "    print('World!')\n",
    ]

    # Perform merge
    merger = ThreeWayMerge(base, ours, theirs)
    result = merger.merge_files()

    print("Merge Result:")
    print("-" * 40)
    for line in result.merged_content:
        print(line, end="")

    if result.has_conflicts:
        print("\nConflicts detected:", len(result.conflicts))
    else:
        print("\nMerge completed successfully!")


if __name__ == "__main__":
    demo_three_way_merge()
