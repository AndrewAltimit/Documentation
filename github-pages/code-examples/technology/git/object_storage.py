"""
Git Object Storage Implementation

Low-level implementation of Git's object storage including:
- Loose object storage with zlib compression
- Pack file format for efficient storage
- Pack index for fast lookups
- Object deduplication
"""

import os
import zlib
import struct
import hashlib
from typing import Dict, List, Optional, Tuple, BinaryIO
from enum import Enum
from dataclasses import dataclass


class ObjectType(Enum):
    COMMIT = 1
    TREE = 2
    BLOB = 3
    TAG = 4
    OFS_DELTA = 6
    REF_DELTA = 7


@dataclass
class GitObject:
    type: ObjectType
    content: bytes
    
    @property
    def sha1(self) -> str:
        """Compute SHA-1 hash of object"""
        type_str = {
            ObjectType.COMMIT: "commit",
            ObjectType.TREE: "tree",
            ObjectType.BLOB: "blob",
            ObjectType.TAG: "tag"
        }[self.type]
        
        header = f"{type_str} {len(self.content)}\0".encode()
        return hashlib.sha1(header + self.content).hexdigest()
    
    def compress(self) -> bytes:
        """Compress object for storage"""
        type_str = {
            ObjectType.COMMIT: "commit",
            ObjectType.TREE: "tree",
            ObjectType.BLOB: "blob",
            ObjectType.TAG: "tag"
        }[self.type]
        
        header = f"{type_str} {len(self.content)}\0".encode()
        return zlib.compress(header + self.content)


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
        
        obj_type_str, size = header.split(' ')
        obj_type = {
            'commit': ObjectType.COMMIT,
            'tree': ObjectType.TREE,
            'blob': ObjectType.BLOB,
            'tag': ObjectType.TAG
        }[obj_type_str]
        
        return GitObject(obj_type, content)
    
    def object_exists(self, sha: str) -> bool:
        """Check if object exists in store"""
        dir_name = sha[:2]
        file_name = sha[2:]
        obj_path = os.path.join(self.objects_dir, dir_name, file_name)
        return os.path.exists(obj_path)


class PackFile:
    """Git packfile format for efficient storage"""
    
    SIGNATURE = b'PACK'
    VERSION = 2
    
    def __init__(self, pack_path: str):
        self.pack_path = pack_path
        self.idx_path = pack_path.replace('.pack', '.idx')
        self.index: Optional[PackIndex] = None
        self.pack_file: Optional[BinaryIO] = None
    
    def open(self):
        """Open pack file and load index"""
        self.pack_file = open(self.pack_path, 'rb')
        self.index = PackIndex(self.idx_path)
        self.index.load()
        
        # Verify pack header
        header = self.pack_file.read(12)
        signature = header[:4]
        version = struct.unpack('>I', header[4:8])[0]
        num_objects = struct.unpack('>I', header[8:12])[0]
        
        if signature != self.SIGNATURE:
            raise ValueError("Invalid pack file signature")
        if version != self.VERSION:
            raise ValueError(f"Unsupported pack version: {version}")
        
        self.num_objects = num_objects
    
    def close(self):
        """Close pack file"""
        if self.pack_file:
            self.pack_file.close()
    
    def read_object(self, sha: str) -> GitObject:
        """Read object from pack file"""
        if not self.index:
            raise RuntimeError("Pack file not opened")
        
        offset = self.index.get_offset(sha)
        if offset is None:
            raise KeyError(f"Object {sha} not found in pack")
        
        self.pack_file.seek(offset)
        
        # Read object header
        obj_type, size = self._read_object_header()
        
        if obj_type in (ObjectType.OFS_DELTA, ObjectType.REF_DELTA):
            # Delta object - need to reconstruct
            if obj_type == ObjectType.OFS_DELTA:
                base_offset = self._read_offset_delta()
                base_obj = self._read_object_at_offset(offset - base_offset)
            else:  # REF_DELTA
                base_sha = self.pack_file.read(20).hex()
                base_obj = self.read_object(base_sha)
            
            delta_data = self._read_compressed_data()
            return self._apply_delta(base_obj, delta_data)
        else:
            # Regular object
            content = self._read_compressed_data()
            return GitObject(obj_type, content)
    
    def _read_object_header(self) -> Tuple[ObjectType, int]:
        """Read variable-length object header"""
        byte = struct.unpack('B', self.pack_file.read(1))[0]
        
        # Type is bits 4-6
        obj_type = ObjectType((byte >> 4) & 0x7)
        
        # Size is bits 0-3, with continuation bit 7
        size = byte & 0xf
        shift = 4
        
        while byte & 0x80:
            byte = struct.unpack('B', self.pack_file.read(1))[0]
            size |= (byte & 0x7f) << shift
            shift += 7
        
        return obj_type, size
    
    def _read_offset_delta(self) -> int:
        """Read offset delta value"""
        byte = struct.unpack('B', self.pack_file.read(1))[0]
        offset = byte & 0x7f
        
        while byte & 0x80:
            byte = struct.unpack('B', self.pack_file.read(1))[0]
            offset = ((offset + 1) << 7) | (byte & 0x7f)
        
        return offset
    
    def _read_compressed_data(self) -> bytes:
        """Read zlib compressed data"""
        decompressor = zlib.decompressobj()
        data = b''
        
        while True:
            chunk = self.pack_file.read(4096)
            if not chunk:
                break
            
            data += decompressor.decompress(chunk)
            
            if decompressor.eof:
                # Rewind to correct position
                unused = len(decompressor.unused_data)
                if unused:
                    self.pack_file.seek(-unused, 1)
                break
        
        return data
    
    def _apply_delta(self, base: GitObject, delta: bytes) -> GitObject:
        """Apply delta to base object"""
        # Delta format:
        # - Source size (variable length)
        # - Target size (variable length)
        # - Delta instructions
        
        i = 0
        
        # Read source size
        source_size, i = self._read_size_encoding(delta, i)
        if len(base.content) != source_size:
            raise ValueError("Delta source size mismatch")
        
        # Read target size
        target_size, i = self._read_size_encoding(delta, i)
        
        # Apply delta instructions
        result = bytearray()
        
        while i < len(delta):
            cmd = delta[i]
            i += 1
            
            if cmd & 0x80:
                # Copy from source
                offset = 0
                size = 0
                
                if cmd & 0x01:
                    offset = delta[i]
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
                
                if cmd & 0x10:
                    size = delta[i]
                    i += 1
                if cmd & 0x20:
                    size |= delta[i] << 8
                    i += 1
                if cmd & 0x40:
                    size |= delta[i] << 16
                    i += 1
                
                if size == 0:
                    size = 0x10000
                
                result.extend(base.content[offset:offset + size])
            else:
                # Insert new data
                size = cmd
                result.extend(delta[i:i + size])
                i += size
        
        if len(result) != target_size:
            raise ValueError("Delta target size mismatch")
        
        return GitObject(base.type, bytes(result))
    
    def _read_size_encoding(self, data: bytes, start: int) -> Tuple[int, int]:
        """Read variable-length size encoding"""
        i = start
        size = 0
        shift = 0
        
        while True:
            byte = data[i]
            i += 1
            size |= (byte & 0x7f) << shift
            shift += 7
            
            if not (byte & 0x80):
                break
        
        return size, i


class PackIndex:
    """Git pack index for fast object lookups"""
    
    def __init__(self, idx_path: str):
        self.idx_path = idx_path
        self.index: Dict[str, int] = {}
        self.fanout = [0] * 256
    
    def load(self):
        """Load pack index from disk"""
        with open(self.idx_path, 'rb') as f:
            # Read header
            header = f.read(8)
            if header != b'\xff\x74\x4f\x63\x00\x00\x00\x02':
                raise ValueError("Invalid pack index version")
            
            # Read fanout table
            for i in range(256):
                self.fanout[i] = struct.unpack('>I', f.read(4))[0]
            
            num_objects = self.fanout[-1]
            
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
                    # Would need to read from large offset table
                    pass
                else:
                    self.index[sha] = offset
    
    def get_offset(self, sha: str) -> Optional[int]:
        """Get offset of object in pack file"""
        return self.index.get(sha)
    
    def create(self, objects: List[Tuple[str, int]]):
        """Create pack index from list of (sha, offset) pairs"""
        # Sort by SHA1
        objects.sort(key=lambda x: x[0])
        
        # Build fanout table
        for sha, offset in objects:
            first_byte = int(sha[:2], 16)
            for i in range(first_byte, 256):
                self.fanout[i] += 1
        
        # Store in index
        for sha, offset in objects:
            self.index[sha] = offset
    
    def write(self):
        """Write pack index to disk"""
        with open(self.idx_path, 'wb') as f:
            # Write header
            f.write(b'\xff\x74\x4f\x63\x00\x00\x00\x02')
            
            # Write fanout table
            for count in self.fanout:
                f.write(struct.pack('>I', count))
            
            # Write SHA1s
            for sha in sorted(self.index.keys()):
                f.write(bytes.fromhex(sha))
            
            # Write CRC32s (simplified - all zeros)
            num_objects = len(self.index)
            f.write(b'\x00' * 4 * num_objects)
            
            # Write offsets
            for sha in sorted(self.index.keys()):
                offset = self.index[sha]
                f.write(struct.pack('>I', offset))


# Example usage
def demo_object_storage():
    """Demonstrate Git object storage"""
    # Create temporary directory structure
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = os.path.join(tmpdir, ".git")
        os.makedirs(os.path.join(git_dir, "objects"))
        
        # Initialize object store
        store = GitObjectStore(git_dir)
        
        # Create and store blob
        content = b"Hello, Git storage!"
        blob = GitObject(ObjectType.BLOB, content)
        sha = store.write_object(blob)
        print(f"Stored blob with SHA: {sha}")
        
        # Read back object
        retrieved = store.read_object(sha)
        print(f"Retrieved object type: {retrieved.type}")
        print(f"Retrieved content: {retrieved.content.decode()}")
        
        # Verify SHA matches
        assert retrieved.sha1 == sha
        print("SHA verification passed!")


if __name__ == "__main__":
    demo_object_storage()