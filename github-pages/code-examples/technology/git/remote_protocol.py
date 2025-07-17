"""
Git Remote Protocol and Delta Compression

Implementation of Git's remote synchronization protocol including:
- Pack negotiation algorithms
- Delta compression using xdelta
- Push/pull operations
- Reference advertisement
- Capability negotiation
"""

import hashlib
import io
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Set, Tuple


@dataclass
class Remote:
    """Remote repository configuration"""

    name: str
    url: str
    fetch_refspecs: List[str]
    push_refspecs: List[str]


@dataclass
class RefSpec:
    """Reference specification for push/pull"""

    source: str
    destination: str
    force: bool = False

    @classmethod
    def parse(cls, spec: str) -> "RefSpec":
        """Parse refspec string like +refs/heads/*:refs/remotes/origin/*"""
        force = spec.startswith("+")
        if force:
            spec = spec[1:]

        if ":" in spec:
            source, dest = spec.split(":", 1)
        else:
            source = spec
            dest = spec

        return cls(source, dest, force)


@dataclass
class FetchResult:
    """Result of fetch operation"""

    updated_refs: Dict[str, Tuple[Optional[str], str]]  # ref -> (old_sha, new_sha)
    new_objects: int
    bytes_transferred: int


@dataclass
class PushResult:
    """Result of push operation"""

    success: bool
    updated_refs: Dict[str, Tuple[Optional[str], str]]  # ref -> (old_sha, new_sha)
    rejected_refs: Dict[str, str]  # ref -> reason


class RemoteProtocol:
    """Git remote protocol handler"""

    def __init__(self, transport: "Transport"):
        self.transport = transport
        self.capabilities = {
            "multi_ack_detailed",
            "side-band-64k",
            "ofs-delta",
            "agent=git/2.x.x",
            "allow-tip-sha1-in-want",
            "allow-reachable-sha1-in-want",
            "no-progress",
            "include-tag",
        }
        self.remote_capabilities = set()

    def fetch_pack(self, remote: Remote, refs: List[str]) -> FetchResult:
        """Fetch objects from remote"""
        # Phase 1: Reference discovery
        remote_refs = self.discover_refs()

        # Phase 2: Determine what we want
        want_refs = self._match_refs(refs, remote_refs, remote.fetch_refspecs)
        wants = [sha for sha in want_refs.values() if sha]

        if not wants:
            return FetchResult({}, 0, 0)

        # Phase 3: Negotiation
        common_commits = self.negotiate_common_commits(wants)

        # Phase 4: Pack transfer
        pack_data = self.receive_pack(wants, common_commits)

        # Phase 5: Update local refs
        updated_refs = self.update_local_refs(want_refs)

        return FetchResult(
            updated_refs=updated_refs,
            new_objects=self._count_objects(pack_data),
            bytes_transferred=len(pack_data),
        )

    def push_pack(self, remote: Remote, refspecs: List[RefSpec]) -> PushResult:
        """Push objects to remote"""
        # Check capabilities
        if "receive-pack" not in self.remote_capabilities:
            return PushResult(False, {}, {"*": "Remote does not support push"})

        # Build update list
        updates = []
        for refspec in refspecs:
            old_sha = self._get_remote_ref(refspec.destination)
            new_sha = self._get_local_ref(refspec.source)

            if (
                not refspec.force
                and old_sha
                and not self._is_fast_forward(old_sha, new_sha)
            ):
                return PushResult(
                    False, {}, {refspec.destination: "Non-fast-forward update"}
                )

            updates.append((refspec.destination, old_sha, new_sha))

        # Build pack of missing objects
        pack = self.build_push_pack(updates)

        # Send pack with atomic transaction
        return self._send_pack_atomic(pack, updates)

    def discover_refs(self) -> Dict[str, str]:
        """Discover remote references"""
        # Send ls-refs request
        request = self._build_ls_refs_request()
        response = self.transport.send_packet_line(request)

        refs = {}
        for line in self._parse_packet_lines(response):
            if line.startswith("#"):
                continue

            parts = line.split(" ", 1)
            if len(parts) == 2:
                sha, ref = parts

                # Parse capabilities from first ref
                if "\0" in ref:
                    ref, caps = ref.split("\0", 1)
                    self.remote_capabilities = set(caps.split())

                refs[ref] = sha

        return refs

    def negotiate_common_commits(self, wants: List[str]) -> Set[str]:
        """Negotiate common commits using multi_ack_detailed"""
        common = set()
        haves = self._get_local_commits()
        done = False
        round_num = 0
        max_rounds = 256

        while not done and round_num < max_rounds:
            # Send have lines
            request = []

            # First round includes wants
            if round_num == 0:
                for want in wants:
                    caps = " ".join(sorted(self.capabilities))
                    request.append(f"want {want} {caps}")
                request.append("")  # Flush

            # Send haves (32 per round)
            have_batch = []
            for _ in range(32):
                if haves:
                    have = haves.pop()
                    have_batch.append(have)
                    request.append(f"have {have}")

            if not have_batch:
                request.append("done")
                done = True
            else:
                request.append("")  # Flush

            # Send request
            response = self.transport.send_packet_lines(request)

            # Parse response
            for line in response:
                if line.startswith("ACK"):
                    parts = line.split()
                    if len(parts) >= 2:
                        common.add(parts[1])

                        if len(parts) >= 3:
                            if parts[2] == "ready":
                                done = True
                            elif parts[2] == "common":
                                # Continue negotiation
                                pass
                elif line == "NAK":
                    # Continue with more haves
                    pass

            round_num += 1

        return common

    def receive_pack(self, wants: List[str], common: Set[str]) -> bytes:
        """Receive pack file from remote"""
        # Build final request
        request = []

        for want in wants:
            request.append(f"want {want}")
        request.append("")  # Flush

        for have in common:
            request.append(f"have {have}")

        request.append("done")

        # Send and receive pack
        response = self.transport.send_receive_pack(request)

        # Extract pack data (skip protocol headers)
        pack_start = response.find(b"PACK")
        if pack_start >= 0:
            return response[pack_start:]

        return response

    def build_push_pack(self, updates: List[Tuple[str, Optional[str], str]]) -> bytes:
        """Build pack for push operation"""
        # Determine objects to send
        objects_to_send = set()
        have_objects = set()

        for ref, old_sha, new_sha in updates:
            if new_sha and new_sha != "0" * 40:
                # Find objects reachable from new but not old
                new_objects = self._find_reachable_objects(new_sha)
                objects_to_send.update(new_objects)

                if old_sha and old_sha != "0" * 40:
                    old_objects = self._find_reachable_objects(old_sha)
                    have_objects.update(old_objects)

        # Remove objects remote already has
        objects_to_send -= have_objects

        # Build pack file
        return self._build_pack_file(list(objects_to_send))

    def _send_pack_atomic(
        self, pack: bytes, updates: List[Tuple[str, Optional[str], str]]
    ) -> PushResult:
        """Send pack atomically with reference updates"""
        # Build push request
        request = []

        # Capability advertisement
        caps = " ".join(["report-status", "side-band-64k", "agent=git/2.x.x"])

        # Reference updates
        for i, (ref, old_sha, new_sha) in enumerate(updates):
            old = old_sha or "0" * 40
            line = f"{old} {new_sha} {ref}"
            if i == 0:
                line += f"\0{caps}"
            request.append(line)

        request.append("")  # Flush

        # Send request with pack
        response = self.transport.send_push_pack(request, pack)

        # Parse response
        updated_refs = {}
        rejected_refs = {}

        for line in response:
            if line.startswith("ok"):
                ref = line[3:]
                # Find the update for this ref
                for r, old, new in updates:
                    if r == ref:
                        updated_refs[ref] = (old, new)
                        break
            elif line.startswith("ng"):
                parts = line[3:].split(" ", 1)
                ref = parts[0]
                reason = parts[1] if len(parts) > 1 else "rejected"
                rejected_refs[ref] = reason

        success = len(rejected_refs) == 0

        return PushResult(success, updated_refs, rejected_refs)

    def _match_refs(
        self, patterns: List[str], remote_refs: Dict[str, str], refspecs: List[str]
    ) -> Dict[str, str]:
        """Match ref patterns against remote refs"""
        matched = {}

        for pattern in patterns:
            if "*" in pattern:
                # Wildcard pattern
                prefix = pattern.replace("*", "")
                for ref, sha in remote_refs.items():
                    if ref.startswith(prefix):
                        matched[ref] = sha
            else:
                # Exact match
                if pattern in remote_refs:
                    matched[pattern] = remote_refs[pattern]

        return matched

    def _get_local_commits(self) -> List[str]:
        """Get list of local commits for negotiation"""
        # Simplified - would read from local repository
        return []

    def _is_fast_forward(self, old_sha: str, new_sha: str) -> bool:
        """Check if update is fast-forward"""
        # Would check if old_sha is ancestor of new_sha
        return True

    def _get_remote_ref(self, ref: str) -> Optional[str]:
        """Get current value of remote ref"""
        # Would query remote
        return None

    def _get_local_ref(self, ref: str) -> Optional[str]:
        """Get current value of local ref"""
        # Would read from local repository
        return "a" * 40  # Mock SHA

    def _find_reachable_objects(self, sha: str) -> Set[str]:
        """Find all objects reachable from commit"""
        # Would traverse commit graph
        return {sha}

    def _build_pack_file(self, objects: List[str]) -> bytes:
        """Build pack file containing objects"""
        # Simplified pack building
        pack = io.BytesIO()

        # Pack header
        pack.write(b"PACK")  # Signature
        pack.write(struct.pack(">I", 2))  # Version
        pack.write(struct.pack(">I", len(objects)))  # Number of objects

        # Objects (simplified - no delta compression)
        for obj_sha in objects:
            # Would read actual object and compress
            obj_data = b"mock object data"
            compressed = zlib.compress(obj_data)

            # Object header (type and size)
            obj_type = 3  # Blob
            size = len(obj_data)

            # Variable-length header
            header = (obj_type << 4) | (size & 0x0F)
            size >>= 4

            while size:
                pack.write(struct.pack("B", header | 0x80))
                header = size & 0x7F
                size >>= 7

            pack.write(struct.pack("B", header))
            pack.write(compressed)

        # Pack checksum
        pack_data = pack.getvalue()
        checksum = hashlib.sha1(pack_data).digest()

        return pack_data + checksum

    def _count_objects(self, pack_data: bytes) -> int:
        """Count objects in pack file"""
        if len(pack_data) < 12:
            return 0

        # Read header
        if pack_data[:4] != b"PACK":
            return 0

        num_objects = struct.unpack(">I", pack_data[8:12])[0]
        return num_objects


class DeltaCompression:
    """Delta compression for efficient transfer"""

    def __init__(self):
        self.window_size = 10
        self.max_delta_size = 1024 * 1024  # 1MB
        self.min_match_length = 4

    def compute_delta(self, source: bytes, target: bytes) -> bytes:
        """Compute binary delta between objects"""
        delta_ops = []

        # Build rolling hash table for source
        source_hashes = self._build_hash_table(source)

        i = 0
        while i < len(target):
            # Try to find matching block
            match = self._find_match(target[i:], source_hashes, source)

            if match and match.length >= self.min_match_length:
                # Emit copy instruction
                delta_ops.append(DeltaOp.copy(match.offset, match.length))
                i += match.length
            else:
                # Collect insert bytes
                insert_start = i
                while i < len(target) and not self._find_match(
                    target[i : i + self.min_match_length], source_hashes, source
                ):
                    i += 1

                # Emit insert instruction
                if i > insert_start:
                    delta_ops.append(DeltaOp.insert(target[insert_start:i]))

        return self._encode_delta(delta_ops, len(source), len(target))

    def apply_delta(self, source: bytes, delta: bytes) -> bytes:
        """Apply delta to reconstruct target"""
        # Parse delta header
        i = 0
        source_size, i = self._read_size(delta, i)
        target_size, i = self._read_size(delta, i)

        if len(source) != source_size:
            raise ValueError("Source size mismatch")

        # Apply delta operations
        result = bytearray()

        while i < len(delta):
            cmd = delta[i]
            i += 1

            if cmd & 0x80:
                # Copy from source
                offset = 0
                size = 0

                # Read offset
                if cmd & 0x01:
                    offset |= delta[i]
                    i += 1
                if cmd & 0x02:
                    offset |= delta[i] << 8
                    i += 1
                if cmd & 0x04:
                    offset |= delta[i] << 16
                    i += 1
                if cmd & 0x08:
                    offset |= delta[i] << 24
                    i += 1

                # Read size
                if cmd & 0x10:
                    size |= delta[i]
                    i += 1
                if cmd & 0x20:
                    size |= delta[i] << 8
                    i += 1
                if cmd & 0x40:
                    size |= delta[i] << 16
                    i += 1

                if size == 0:
                    size = 0x10000

                # Copy from source
                result.extend(source[offset : offset + size])
            else:
                # Insert new data
                size = cmd
                result.extend(delta[i : i + size])
                i += size

        if len(result) != target_size:
            raise ValueError(
                f"Target size mismatch: expected {target_size}, got {len(result)}"
            )

        return bytes(result)

    def _build_hash_table(self, data: bytes) -> Dict[int, List[int]]:
        """Build rolling hash table for source data"""
        hash_table = defaultdict(list)

        if len(data) < self.window_size:
            return hash_table

        # Initial hash
        h = 0
        for i in range(self.window_size):
            h = (h * 31 + data[i]) & 0xFFFFFFFF

        hash_table[h].append(0)

        # Rolling hash
        for i in range(self.window_size, len(data)):
            # Remove old byte
            old_byte = data[i - self.window_size]
            h = (h - old_byte * (31 ** (self.window_size - 1))) & 0xFFFFFFFF

            # Add new byte
            h = (h * 31 + data[i]) & 0xFFFFFFFF

            # Store position
            hash_table[h].append(i - self.window_size + 1)

        return hash_table

    def _find_match(
        self, data: bytes, hash_table: Dict[int, List[int]], source: bytes
    ) -> Optional["Match"]:
        """Find best match in source data"""
        if len(data) < self.window_size:
            return None

        # Compute hash of current window
        h = 0
        for i in range(self.window_size):
            h = (h * 31 + data[i]) & 0xFFFFFFFF

        # Look up in hash table
        if h not in hash_table:
            return None

        # Find best match
        best_match = None

        for pos in hash_table[h]:
            # Verify match and extend
            match_len = 0
            while (
                match_len < len(data)
                and pos + match_len < len(source)
                and data[match_len] == source[pos + match_len]
            ):
                match_len += 1

            if match_len >= self.min_match_length:
                if not best_match or match_len > best_match.length:
                    best_match = Match(pos, match_len)

        return best_match

    def _encode_delta(
        self, ops: List["DeltaOp"], source_size: int, target_size: int
    ) -> bytes:
        """Encode delta operations"""
        delta = bytearray()

        # Encode sizes
        delta.extend(self._encode_size(source_size))
        delta.extend(self._encode_size(target_size))

        # Encode operations
        for op in ops:
            if op.is_copy:
                # Copy instruction
                cmd = 0x80

                # Encode offset
                offset = op.offset
                if offset & 0xFF:
                    cmd |= 0x01
                    delta.append(offset & 0xFF)
                if offset & 0xFF00:
                    cmd |= 0x02
                    delta.append((offset >> 8) & 0xFF)
                if offset & 0xFF0000:
                    cmd |= 0x04
                    delta.append((offset >> 16) & 0xFF)
                if offset & 0xFF000000:
                    cmd |= 0x08
                    delta.append((offset >> 24) & 0xFF)

                # Encode size
                size = op.size
                if size & 0xFF:
                    cmd |= 0x10
                    delta.append(size & 0xFF)
                if size & 0xFF00:
                    cmd |= 0x20
                    delta.append((size >> 8) & 0xFF)
                if size & 0xFF0000:
                    cmd |= 0x40
                    delta.append((size >> 16) & 0xFF)

                # Write command byte at the beginning
                delta.insert(len(delta) - bin(cmd).count("1") + 1, cmd)
            else:
                # Insert instruction
                size = len(op.data)
                if size > 127:
                    raise ValueError("Insert too large")

                delta.append(size)
                delta.extend(op.data)

        return bytes(delta)

    def _encode_size(self, size: int) -> bytes:
        """Encode size in variable-length format"""
        result = bytearray()

        while True:
            byte = size & 0x7F
            size >>= 7

            if size:
                byte |= 0x80

            result.append(byte)

            if not size:
                break

        return bytes(result)

    def _read_size(self, data: bytes, start: int) -> Tuple[int, int]:
        """Read variable-length size"""
        i = start
        size = 0
        shift = 0

        while i < len(data):
            byte = data[i]
            i += 1

            size |= (byte & 0x7F) << shift
            shift += 7

            if not (byte & 0x80):
                break

        return size, i


@dataclass
class Match:
    """Represents a match in delta compression"""

    offset: int
    length: int


@dataclass
class DeltaOp:
    """Delta operation (copy or insert)"""

    is_copy: bool
    offset: Optional[int] = None
    size: Optional[int] = None
    data: Optional[bytes] = None

    @classmethod
    def copy(cls, offset: int, size: int) -> "DeltaOp":
        return cls(True, offset=offset, size=size)

    @classmethod
    def insert(cls, data: bytes) -> "DeltaOp":
        return cls(False, data=data)


# Example usage
def demo_delta_compression():
    """Demonstrate delta compression"""
    delta = DeltaCompression()

    # Original and modified versions
    source = b"The quick brown fox jumps over the lazy dog"
    target = b"The quick brown fox leaps over the lazy cat"

    print("Delta Compression Demo")
    print("-" * 40)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Source size: {len(source)} bytes")
    print(f"Target size: {len(target)} bytes")

    # Compute delta
    delta_data = delta.compute_delta(source, target)
    print(f"\nDelta size: {len(delta_data)} bytes")
    print(f"Compression ratio: {len(delta_data) / len(target):.2%}")

    # Apply delta to reconstruct
    reconstructed = delta.apply_delta(source, delta_data)
    print(f"\nReconstructed: {reconstructed}")
    print(f"Matches target: {reconstructed == target}")


if __name__ == "__main__":
    demo_delta_compression()
