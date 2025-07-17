"""
Git Repository Operations at the Protocol Level

Implementation of Git protocol operations including:
- Smart HTTP protocol
- Reference discovery
- Pack negotiation
- Clone operations
- Transport abstraction
"""

import os
import struct
import subprocess
import zlib
from io import BytesIO
from typing import BinaryIO, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


class Transport:
    """Abstract transport layer for Git protocols"""

    def get(self, path: str, params: Optional[Dict] = None) -> bytes:
        raise NotImplementedError

    def post(self, path: str, data: bytes) -> bytes:
        raise NotImplementedError


class HTTPTransport(Transport):
    """HTTP/HTTPS transport implementation"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "git/2.x.x", "Accept-Encoding": "gzip"}
        )

    def get(self, path: str, params: Optional[Dict] = None) -> bytes:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.content

    def post(self, path: str, data: bytes) -> bytes:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/x-git-upload-pack-request",
            "Accept": "application/x-git-upload-pack-result",
        }
        response = self.session.post(url, data=data, headers=headers)
        response.raise_for_status()
        return response.content


class GitProtocol:
    """Git wire protocol implementation"""

    def __init__(self, transport: Transport):
        self.transport = transport
        self.capabilities = set()

    def discover_refs(self) -> Dict[str, str]:
        """Discover references using smart HTTP protocol"""
        # Send: GET /info/refs?service=git-upload-pack
        response = self.transport.get(
            "/info/refs", params={"service": "git-upload-pack"}
        )

        # Parse response
        refs = {}
        lines = response.decode("utf-8").split("\n")

        # Skip service announcement
        if lines[0].startswith("# service="):
            lines = lines[1:]

        # Skip packet headers
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue

            # Parse pkt-line format
            if len(line) >= 4:
                try:
                    pkt_len = int(line[:4], 16)
                    if pkt_len == 0:  # Flush packet
                        break
                    line = line[4:]
                except ValueError:
                    pass

            if " " in line:
                sha, ref = line.split(" ", 1)

                # Parse capabilities from first line
                if "\0" in ref:
                    ref, caps = ref.split("\0", 1)
                    self.capabilities = set(caps.split(" "))

                refs[ref.strip()] = sha.strip()

            i += 1

        return refs

    def negotiate_pack(self, wants: List[str], haves: List[str]) -> bytes:
        """Negotiate pack file using smart protocol"""
        # Build request in pkt-line format
        request_lines = []

        # Want lines
        for i, want in enumerate(wants):
            if i == 0 and self.capabilities:
                caps = " ".join(sorted(self.capabilities))
                line = f"want {want} {caps}\n"
            else:
                line = f"want {want}\n"
            request_lines.append(self._pkt_line(line))

        # Flush packet
        request_lines.append("0000")

        # Have lines
        for have in haves:
            line = f"have {have}\n"
            request_lines.append(self._pkt_line(line))

        # Done
        request_lines.append(self._pkt_line("done\n"))

        # Send request
        request_data = "".join(request_lines).encode("utf-8")
        response = self.transport.post("/git-upload-pack", request_data)

        # Parse pack file from response
        return self._parse_pack_response(response)

    def _pkt_line(self, data: str) -> str:
        """Format data as pkt-line"""
        length = len(data) + 4
        return f"{length:04x}{data}"

    def _parse_pack_response(self, response: bytes) -> bytes:
        """Extract pack file from protocol response"""
        # Response contains pkt-lines followed by pack data
        i = 0

        # Skip NAK/ACK lines
        while i < len(response):
            if i + 4 > len(response):
                break

            pkt_len = int(response[i : i + 4], 16)
            if pkt_len == 0:  # Flush packet
                i += 4
                break

            # Skip this packet
            i += pkt_len

        # Remaining data is the pack file
        return response[i:]


class CloneOperation:
    """High-level clone implementation"""

    def __init__(self, url: str, target_dir: str):
        self.url = url
        self.target_dir = target_dir
        self.protocol = GitProtocol(HTTPTransport(url))

    def clone(self, branch: Optional[str] = None, depth: Optional[int] = None):
        """Perform repository clone"""
        # Create directory structure
        os.makedirs(self.target_dir)
        git_dir = os.path.join(self.target_dir, ".git")
        os.makedirs(git_dir)

        # Initialize repository structure
        self._init_repo_structure(git_dir)

        # Discover refs
        print("Discovering references...")
        refs = self.protocol.discover_refs()

        # Determine what to fetch
        if branch:
            want_ref = f"refs/heads/{branch}"
            if want_ref not in refs:
                raise ValueError(f"Branch {branch} not found")
            wants = [refs[want_ref]]
        else:
            # Fetch all branches
            wants = [sha for ref, sha in refs.items() if ref.startswith("refs/heads/")]

        # Handle shallow clone
        if depth:
            self.protocol.capabilities.add(f"deepen {depth}")

        # Negotiate pack
        print("Negotiating pack file...")
        pack_data = self.protocol.negotiate_pack(wants, [])

        # Process pack file
        print("Processing pack file...")
        self._process_pack(git_dir, pack_data)

        # Update references
        print("Updating references...")
        self._update_refs(git_dir, refs)

        # Checkout HEAD
        print("Checking out files...")
        self._checkout_head(git_dir)

        print("Clone completed!")

    def _init_repo_structure(self, git_dir: str):
        """Initialize basic Git repository structure"""
        # Create directories
        dirs = [
            "objects",
            "objects/info",
            "objects/pack",
            "refs",
            "refs/heads",
            "refs/tags",
            "info",
            "hooks",
            "logs",
        ]

        for dir_name in dirs:
            os.makedirs(os.path.join(git_dir, dir_name))

        # Create config file
        config_content = """[core]
\trepositoryformatversion = 0
\tfilemode = true
\tbare = false
\tlogallrefupdates = true
"""
        with open(os.path.join(git_dir, "config"), "w") as f:
            f.write(config_content)

        # Create HEAD
        with open(os.path.join(git_dir, "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")

        # Create description
        with open(os.path.join(git_dir, "description"), "w") as f:
            f.write("Unnamed repository; edit this file to name it.\n")

    def _process_pack(self, git_dir: str, pack_data: bytes):
        """Process and store pack file"""
        # Verify pack header
        if len(pack_data) < 12:
            raise ValueError("Invalid pack file")

        signature = pack_data[:4]
        if signature != b"PACK":
            raise ValueError("Invalid pack signature")

        version = struct.unpack(">I", pack_data[4:8])[0]
        num_objects = struct.unpack(">I", pack_data[8:12])[0]

        print(f"Pack version: {version}, objects: {num_objects}")

        # Calculate pack file name (simplified - should use pack checksum)
        import hashlib

        pack_hash = hashlib.sha1(pack_data).hexdigest()
        pack_name = f"pack-{pack_hash}.pack"
        pack_path = os.path.join(git_dir, "objects", "pack", pack_name)

        # Write pack file
        with open(pack_path, "wb") as f:
            f.write(pack_data)

        # In a real implementation, we would also:
        # 1. Verify pack checksum
        # 2. Generate pack index
        # 3. Optionally unpack objects

    def _update_refs(self, git_dir: str, refs: Dict[str, str]):
        """Update local references"""
        for ref_name, sha in refs.items():
            if ref_name.startswith("refs/"):
                ref_path = os.path.join(git_dir, ref_name)
                ref_dir = os.path.dirname(ref_path)
                os.makedirs(ref_dir, exist_ok=True)

                with open(ref_path, "w") as f:
                    f.write(sha + "\n")

    def _checkout_head(self, git_dir: str):
        """Checkout HEAD to working directory"""
        # Read HEAD
        head_path = os.path.join(git_dir, "HEAD")
        with open(head_path, "r") as f:
            head_content = f.read().strip()

        if head_content.startswith("ref:"):
            # Symbolic ref
            ref_name = head_content[4:].strip()
            ref_path = os.path.join(git_dir, ref_name)

            if os.path.exists(ref_path):
                with open(ref_path, "r") as f:
                    commit_sha = f.read().strip()
            else:
                # Ref doesn't exist yet
                return
        else:
            # Direct SHA
            commit_sha = head_content

        # In a real implementation, we would:
        # 1. Read the commit object
        # 2. Read the tree object
        # 3. Recursively checkout all files
        # 4. Update the index

        print(f"Would checkout commit: {commit_sha}")


class FetchOperation:
    """Fetch objects from remote repository"""

    def __init__(self, remote_url: str, local_repo: str):
        self.remote_url = remote_url
        self.local_repo = local_repo
        self.protocol = GitProtocol(HTTPTransport(remote_url))

    def fetch(self, refspecs: List[str]) -> Dict[str, str]:
        """Fetch objects and refs from remote"""
        # Discover remote refs
        remote_refs = self.protocol.discover_refs()

        # Parse refspecs
        refs_to_fetch = self._parse_refspecs(refspecs, remote_refs)

        # Find local refs
        local_refs = self._get_local_refs()

        # Determine what objects we need
        wants = [sha for ref, sha in refs_to_fetch.items()]
        haves = list(set(local_refs.values()))

        # Negotiate pack
        if wants:
            pack_data = self.protocol.negotiate_pack(wants, haves)
            self._process_pack(pack_data)

        # Update remote tracking refs
        updates = {}
        for remote_ref, sha in refs_to_fetch.items():
            local_ref = f"refs/remotes/origin/{remote_ref.split('/')[-1]}"
            self._update_ref(local_ref, sha)
            updates[local_ref] = sha

        return updates

    def _parse_refspecs(
        self, refspecs: List[str], remote_refs: Dict[str, str]
    ) -> Dict[str, str]:
        """Parse fetch refspecs"""
        refs_to_fetch = {}

        for refspec in refspecs:
            if ":" in refspec:
                src, dst = refspec.split(":", 1)
            else:
                src = refspec
                dst = refspec

            # Handle wildcards
            if "*" in src:
                prefix = src.replace("*", "")
                for ref, sha in remote_refs.items():
                    if ref.startswith(prefix):
                        refs_to_fetch[ref] = sha
            else:
                if src in remote_refs:
                    refs_to_fetch[src] = remote_refs[src]

        return refs_to_fetch

    def _get_local_refs(self) -> Dict[str, str]:
        """Get local repository refs"""
        # Simplified - would use RefManager in practice
        refs = {}
        refs_dir = os.path.join(self.local_repo, ".git", "refs")

        for root, dirs, files in os.walk(refs_dir):
            for file in files:
                ref_path = os.path.join(root, file)
                ref_name = os.path.relpath(
                    ref_path, os.path.join(self.local_repo, ".git")
                )

                with open(ref_path, "r") as f:
                    refs[ref_name] = f.read().strip()

        return refs

    def _process_pack(self, pack_data: bytes):
        """Store received pack file"""
        pack_dir = os.path.join(self.local_repo, ".git", "objects", "pack")

        # Generate pack name from checksum
        import hashlib

        pack_hash = hashlib.sha1(pack_data).hexdigest()
        pack_path = os.path.join(pack_dir, f"pack-{pack_hash}.pack")

        with open(pack_path, "wb") as f:
            f.write(pack_data)

    def _update_ref(self, ref_name: str, sha: str):
        """Update reference"""
        ref_path = os.path.join(self.local_repo, ".git", ref_name)
        ref_dir = os.path.dirname(ref_path)
        os.makedirs(ref_dir, exist_ok=True)

        with open(ref_path, "w") as f:
            f.write(sha + "\n")


# Example usage
def demo_clone():
    """Demonstrate clone operation"""
    # Note: This is a simplified demonstration
    # Real implementation would require actual Git server

    print("Git Protocol Implementation Demo")
    print("-" * 40)

    # Simulate protocol operations
    transport = HTTPTransport("https://github.com/example/repo.git")
    protocol = GitProtocol(transport)

    print("Simulating reference discovery...")
    # In practice, this would connect to real server
    refs = {
        "refs/heads/main": "a" * 40,
        "refs/heads/develop": "b" * 40,
        "refs/tags/v1.0": "c" * 40,
    }

    print("Found references:")
    for ref, sha in refs.items():
        print(f"  {ref}: {sha[:8]}...")

    print("\nCapabilities:", sorted(protocol.capabilities))


if __name__ == "__main__":
    demo_clone()
