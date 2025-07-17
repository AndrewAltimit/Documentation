"""
Git Reference Management

Implementation of Git's reference system including:
- Loose refs (files in .git/refs/)
- Packed refs (single file optimization)
- Symbolic refs (like HEAD)
- Reference transactions for atomicity
"""

import fcntl
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class RefUpdate:
    """Represents a reference update operation"""

    ref_name: str
    old_sha: Optional[str]
    new_sha: str
    message: Optional[str] = None


class RefManager:
    """Manages Git references (branches, tags, etc.)"""

    def __init__(self, git_dir: str):
        self.git_dir = git_dir
        self.refs_dir = os.path.join(git_dir, "refs")
        self.packed_refs_path = os.path.join(git_dir, "packed-refs")
        self.logs_dir = os.path.join(git_dir, "logs")

    def read_ref(self, ref_name: str) -> Optional[str]:
        """Read reference value (commit SHA)"""
        # Check symbolic ref first
        if ref_name == "HEAD":
            return self._read_head()

        # Check packed refs first
        packed_ref = self._read_packed_ref(ref_name)
        if packed_ref:
            return packed_ref

        # Check loose ref
        ref_path = os.path.join(self.git_dir, ref_name)
        if os.path.exists(ref_path):
            with open(ref_path, "r") as f:
                content = f.read().strip()
                # Handle symbolic refs
                if content.startswith("ref:"):
                    target_ref = content[4:].strip()
                    return self.read_ref(target_ref)
                return content

        return None

    def update_ref(
        self,
        ref_name: str,
        new_sha: str,
        old_sha: Optional[str] = None,
        message: Optional[str] = None,
    ) -> bool:
        """Atomically update reference"""
        ref_path = os.path.join(self.git_dir, ref_name)

        # Ensure atomic update
        if old_sha is not None:
            current = self.read_ref(ref_name)
            if current != old_sha:
                return False  # Concurrent modification

        # Create directory if needed
        ref_dir = os.path.dirname(ref_path)
        os.makedirs(ref_dir, exist_ok=True)

        # Write to temporary file and rename
        tmp_path = f"{ref_path}.lock"
        try:
            with open(tmp_path, "w") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(new_sha + "\n")

            # Atomic rename
            os.rename(tmp_path, ref_path)

            # Update reflog
            self._update_reflog(ref_name, old_sha or "0" * 40, new_sha, message)

            return True
        except Exception:
            # Cleanup on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def delete_ref(self, ref_name: str) -> bool:
        """Delete a reference"""
        ref_path = os.path.join(self.git_dir, ref_name)

        if os.path.exists(ref_path):
            old_sha = self.read_ref(ref_name)
            os.unlink(ref_path)

            # Update reflog
            self._update_reflog(ref_name, old_sha or "0" * 40, "0" * 40, "deleted")

            return True

        # Check packed refs
        if self._delete_packed_ref(ref_name):
            return True

        return False

    def create_symbolic_ref(self, name: str, target: str):
        """Create symbolic reference (like HEAD)"""
        ref_path = os.path.join(self.git_dir, name)

        with open(ref_path, "w") as f:
            f.write(f"ref: {target}\n")

    def list_refs(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """List all references with optional prefix filter"""
        refs = {}

        # Read loose refs
        if os.path.exists(self.refs_dir):
            for root, dirs, files in os.walk(self.refs_dir):
                for file in files:
                    ref_path = os.path.join(root, file)
                    ref_name = os.path.relpath(ref_path, self.git_dir)

                    if prefix and not ref_name.startswith(prefix):
                        continue

                    sha = self.read_ref(ref_name)
                    if sha:
                        refs[ref_name] = sha

        # Read packed refs
        packed_refs = self._read_all_packed_refs()
        for ref_name, sha in packed_refs.items():
            if prefix and not ref_name.startswith(prefix):
                continue
            refs[ref_name] = sha

        return refs

    def pack_refs(self):
        """Pack loose refs into packed-refs file"""
        all_refs = self.list_refs()

        # Write packed-refs file
        with open(self.packed_refs_path, "w") as f:
            f.write("# pack-refs with: peeled fully-peeled\n")

            for ref_name, sha in sorted(all_refs.items()):
                f.write(f"{sha} {ref_name}\n")

                # For annotated tags, also write peeled value
                if ref_name.startswith("refs/tags/"):
                    peeled = self._peel_tag(sha)
                    if peeled != sha:
                        f.write(f"^{peeled}\n")

        # Remove loose refs that were packed
        for ref_name in all_refs:
            ref_path = os.path.join(self.git_dir, ref_name)
            if os.path.exists(ref_path):
                os.unlink(ref_path)

    @contextmanager
    def transaction(self):
        """Create reference transaction for atomic updates"""
        txn = RefTransaction(self)
        try:
            yield txn
            txn.commit()
        except Exception:
            txn.rollback()
            raise

    def _read_head(self) -> Optional[str]:
        """Read HEAD reference"""
        head_path = os.path.join(self.git_dir, "HEAD")
        if os.path.exists(head_path):
            with open(head_path, "r") as f:
                content = f.read().strip()
                if content.startswith("ref:"):
                    # Symbolic ref
                    target = content[4:].strip()
                    return self.read_ref(target)
                else:
                    # Direct SHA
                    return content
        return None

    def _read_packed_ref(self, ref_name: str) -> Optional[str]:
        """Read from packed-refs file"""
        if not os.path.exists(self.packed_refs_path):
            return None

        with open(self.packed_refs_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    sha, ref = parts
                    if ref == ref_name:
                        return sha

        return None

    def _read_all_packed_refs(self) -> Dict[str, str]:
        """Read all packed refs"""
        refs = {}

        if not os.path.exists(self.packed_refs_path):
            return refs

        with open(self.packed_refs_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    sha, ref = parts
                    refs[ref] = sha

        return refs

    def _delete_packed_ref(self, ref_name: str) -> bool:
        """Delete ref from packed-refs file"""
        if not os.path.exists(self.packed_refs_path):
            return False

        # Read all packed refs
        lines = []
        found = False

        with open(self.packed_refs_path, "r") as f:
            skip_next_peeled = False
            for line in f:
                if skip_next_peeled and line.startswith("^"):
                    skip_next_peeled = False
                    continue

                if not line.startswith("#") and not line.startswith("^"):
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2 and parts[1] == ref_name:
                        found = True
                        skip_next_peeled = True
                        continue

                lines.append(line)

        if found:
            # Rewrite packed-refs without the deleted ref
            with open(self.packed_refs_path, "w") as f:
                f.writelines(lines)

        return found

    def _update_reflog(
        self, ref_name: str, old_sha: str, new_sha: str, message: Optional[str]
    ):
        """Update reflog for ref"""
        log_path = os.path.join(self.logs_dir, ref_name)
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

        # Get user info (simplified)
        user = os.environ.get("USER", "unknown")
        email = f"{user}@localhost"
        timestamp = int(datetime.now().timestamp())
        timezone = "+0000"

        # Format reflog entry
        entry = f"{old_sha} {new_sha} {user} <{email}> {timestamp} {timezone}\t"
        if message:
            entry += message
        else:
            entry += f"update: {ref_name}"
        entry += "\n"

        # Append to reflog
        with open(log_path, "a") as f:
            f.write(entry)

    def _peel_tag(self, sha: str) -> str:
        """Peel annotated tag to get commit SHA"""
        # In a real implementation, this would read the tag object
        # and follow the reference to the commit
        return sha


class RefTransaction:
    """Atomic reference transaction"""

    def __init__(self, ref_manager: RefManager):
        self.ref_manager = ref_manager
        self.updates: List[RefUpdate] = []
        self.locks: Dict[str, int] = {}  # ref_name -> file descriptor

    def update(
        self,
        ref_name: str,
        new_sha: str,
        old_sha: Optional[str] = None,
        message: Optional[str] = None,
    ):
        """Queue reference update"""
        self.updates.append(RefUpdate(ref_name, old_sha, new_sha, message))

    def delete(self, ref_name: str):
        """Queue reference deletion"""
        old_sha = self.ref_manager.read_ref(ref_name)
        self.updates.append(RefUpdate(ref_name, old_sha, "0" * 40, "deleted"))

    def commit(self):
        """Commit all updates atomically"""
        try:
            # Phase 1: Acquire all locks
            for update in self.updates:
                self._acquire_lock(update.ref_name)

            # Phase 2: Verify preconditions
            for update in self.updates:
                if update.old_sha is not None:
                    current = self.ref_manager.read_ref(update.ref_name)
                    if current != update.old_sha:
                        raise ValueError(f"Reference {update.ref_name} has changed")

            # Phase 3: Apply updates
            for update in self.updates:
                if update.new_sha == "0" * 40:
                    # Deletion
                    self.ref_manager.delete_ref(update.ref_name)
                else:
                    # Update
                    self.ref_manager.update_ref(
                        update.ref_name, update.new_sha, update.old_sha, update.message
                    )
        finally:
            # Release all locks
            self._release_locks()

    def rollback(self):
        """Rollback transaction"""
        self._release_locks()
        self.updates.clear()

    def _acquire_lock(self, ref_name: str):
        """Acquire lock for reference"""
        lock_path = os.path.join(self.ref_manager.git_dir, f"{ref_name}.lock")
        lock_dir = os.path.dirname(lock_path)
        os.makedirs(lock_dir, exist_ok=True)

        fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.locks[ref_name] = fd

    def _release_locks(self):
        """Release all acquired locks"""
        for ref_name, fd in self.locks.items():
            os.close(fd)
            lock_path = os.path.join(self.ref_manager.git_dir, f"{ref_name}.lock")
            if os.path.exists(lock_path):
                os.unlink(lock_path)
        self.locks.clear()


# Example usage
def demo_ref_management():
    """Demonstrate reference management"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = os.path.join(tmpdir, ".git")
        os.makedirs(git_dir)

        # Initialize ref manager
        ref_manager = RefManager(git_dir)

        # Create some refs
        ref_manager.update_ref("refs/heads/main", "a" * 40)
        ref_manager.update_ref("refs/heads/feature", "b" * 40)
        ref_manager.update_ref("refs/tags/v1.0", "c" * 40)

        # Create symbolic ref
        ref_manager.create_symbolic_ref("HEAD", "refs/heads/main")

        # Read refs
        print("HEAD:", ref_manager.read_ref("HEAD"))
        print("main:", ref_manager.read_ref("refs/heads/main"))

        # List all refs
        print("\nAll refs:")
        for ref, sha in ref_manager.list_refs().items():
            print(f"  {ref}: {sha[:8]}...")

        # Atomic transaction
        print("\nPerforming atomic transaction...")
        with ref_manager.transaction() as txn:
            txn.update("refs/heads/main", "d" * 40, "a" * 40)
            txn.update("refs/heads/feature", "e" * 40, "b" * 40)

        print("Transaction completed!")
        print("main:", ref_manager.read_ref("refs/heads/main")[:8] + "...")


if __name__ == "__main__":
    demo_ref_management()
