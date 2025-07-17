"""
Database Storage Engine Implementations

Page management, buffer pool, B+ trees, and LSM trees.
"""

import hashlib
import heapq
import mmap
import os
import random
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


class Page:
    """Database page structure"""

    PAGE_SIZE = 8192  # 8KB pages
    HEADER_SIZE = 96

    def __init__(self, page_id: int):
        self.page_id = page_id
        self.page_type = "data"  # data, index, overflow, free
        self.free_space_offset = self.HEADER_SIZE
        self.slot_count = 0
        self.slots = []  # List of (offset, length) tuples
        self.data = bytearray(self.PAGE_SIZE)

        # Header fields
        self.lsn = 0  # Log sequence number
        self.checksum = 0
        self.next_page = -1
        self.prev_page = -1

        # Initialize header
        self._write_header()

    def _write_header(self):
        """Write page header"""
        # Page ID (8 bytes)
        struct.pack_into("Q", self.data, 0, self.page_id)
        # Page type (4 bytes)
        struct.pack_into("I", self.data, 8, hash(self.page_type) & 0xFFFFFFFF)
        # Free space offset (4 bytes)
        struct.pack_into("I", self.data, 12, self.free_space_offset)
        # Slot count (4 bytes)
        struct.pack_into("I", self.data, 16, self.slot_count)
        # LSN (8 bytes)
        struct.pack_into("Q", self.data, 20, self.lsn)
        # Next page (8 bytes)
        struct.pack_into("q", self.data, 28, self.next_page)
        # Prev page (8 bytes)
        struct.pack_into("q", self.data, 36, self.prev_page)

    def _read_header(self):
        """Read page header"""
        self.page_id = struct.unpack_from("Q", self.data, 0)[0]
        self.free_space_offset = struct.unpack_from("I", self.data, 12)[0]
        self.slot_count = struct.unpack_from("I", self.data, 16)[0]
        self.lsn = struct.unpack_from("Q", self.data, 20)[0]
        self.next_page = struct.unpack_from("q", self.data, 28)[0]
        self.prev_page = struct.unpack_from("q", self.data, 36)[0]

    def insert_record(self, record: bytes) -> Optional[int]:
        """Insert record into page, return slot number"""
        record_size = len(record)
        slot_size = 8  # (offset, length) pair

        # Check if enough space
        needed_space = record_size + slot_size
        available_space = (
            self.PAGE_SIZE - self.free_space_offset - (self.slot_count * slot_size)
        )

        if available_space < needed_space:
            return None  # Not enough space

        # Write record at free space offset
        self.data[self.free_space_offset : self.free_space_offset + record_size] = (
            record
        )

        # Add slot
        slot_num = self.slot_count
        self.slots.append((self.free_space_offset, record_size))

        # Update metadata
        self.free_space_offset += record_size
        self.slot_count += 1

        # Update header
        self._write_header()

        # Write slot directory at end of page
        slot_offset = self.PAGE_SIZE - ((slot_num + 1) * slot_size)
        struct.pack_into(
            "II",
            self.data,
            slot_offset,
            self.slots[slot_num][0],
            self.slots[slot_num][1],
        )

        return slot_num

    def get_record(self, slot_num: int) -> Optional[bytes]:
        """Get record by slot number"""
        if slot_num >= self.slot_count:
            return None

        # Read slot from directory
        slot_size = 8
        slot_offset = self.PAGE_SIZE - ((slot_num + 1) * slot_size)
        offset, length = struct.unpack_from("II", self.data, slot_offset)

        # Check if record was deleted
        if offset == 0 and length == 0:
            return None

        return bytes(self.data[offset : offset + length])

    def delete_record(self, slot_num: int) -> bool:
        """Mark record as deleted"""
        if slot_num >= self.slot_count:
            return False

        # Mark slot as deleted
        slot_size = 8
        slot_offset = self.PAGE_SIZE - ((slot_num + 1) * slot_size)
        struct.pack_into("II", self.data, slot_offset, 0, 0)

        return True

    def compact(self):
        """Compact page to reclaim deleted space"""
        # Read all valid records
        valid_records = []
        for i in range(self.slot_count):
            record = self.get_record(i)
            if record:
                valid_records.append((i, record))

        # Reset page
        self.free_space_offset = self.HEADER_SIZE
        self.slots = []

        # Reinsert valid records
        new_slots = {}
        for old_slot, record in valid_records:
            new_slot = self.insert_record(record)
            new_slots[old_slot] = new_slot

        return new_slots

    def get_free_space(self) -> int:
        """Get available free space"""
        slot_directory_size = self.slot_count * 8
        return self.PAGE_SIZE - self.free_space_offset - slot_directory_size

    def calculate_checksum(self) -> int:
        """Calculate page checksum"""
        # Exclude checksum field itself
        hasher = hashlib.sha256()
        hasher.update(self.data[:44])  # Before checksum
        hasher.update(self.data[48:])  # After checksum
        return int.from_bytes(hasher.digest()[:4], "big")


class BufferPool:
    """Buffer pool manager with LRU eviction"""

    def __init__(self, num_pages: int, storage_manager: "StorageManager"):
        self.num_pages = num_pages
        self.storage_manager = storage_manager
        self.pages = {}  # page_id -> Page
        self.dirty = set()  # Set of dirty page IDs
        self.pin_count = {}  # page_id -> pin count
        self.lru = OrderedDict()  # LRU tracking
        self.lock = threading.Lock()
        self.page_locks = {}  # page_id -> threading.Lock

    def fetch_page(self, page_id: int) -> Optional[Page]:
        """Fetch page with buffer management"""
        with self.lock:
            # Check if page in buffer
            if page_id in self.pages:
                # Update LRU
                self.lru.move_to_end(page_id)
                self.pin_count[page_id] = self.pin_count.get(page_id, 0) + 1
                return self.pages[page_id]

            # Need to load page
            if len(self.pages) >= self.num_pages:
                # Evict a page
                evicted = self._evict_page()
                if evicted is None:
                    return None  # No evictable pages

            # Load page from disk
            page = self.storage_manager.read_page(page_id)
            if page is None:
                return None

            # Add to buffer
            self.pages[page_id] = page
            self.pin_count[page_id] = 1
            self.lru[page_id] = True
            self.page_locks[page_id] = threading.Lock()

            return page

    def unpin_page(self, page_id: int, is_dirty: bool = False):
        """Unpin page and optionally mark as dirty"""
        with self.lock:
            if page_id in self.pin_count:
                self.pin_count[page_id] = max(0, self.pin_count[page_id] - 1)

                if is_dirty:
                    self.dirty.add(page_id)

    def flush_page(self, page_id: int) -> bool:
        """Write page to disk"""
        with self.lock:
            if page_id in self.pages and page_id in self.dirty:
                page = self.pages[page_id]
                success = self.storage_manager.write_page(page)
                if success:
                    self.dirty.discard(page_id)
                return success
        return False

    def flush_all(self):
        """Flush all dirty pages"""
        with self.lock:
            dirty_pages = list(self.dirty)

        for page_id in dirty_pages:
            self.flush_page(page_id)

    def _evict_page(self) -> Optional[int]:
        """Evict page using LRU policy"""
        # Find unpinned page using LRU order
        for page_id in self.lru:
            if self.pin_count.get(page_id, 0) == 0:
                # Flush if dirty
                if page_id in self.dirty:
                    self.flush_page(page_id)

                # Remove from buffer
                del self.pages[page_id]
                del self.lru[page_id]
                del self.pin_count[page_id]
                del self.page_locks[page_id]

                return page_id

        return None  # No evictable pages


class StorageManager:
    """Manages persistent storage of pages"""

    def __init__(self, filename: str, page_size: int = 8192):
        self.filename = filename
        self.page_size = page_size
        self.lock = threading.Lock()

        # Create file if doesn't exist
        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                # Write header page
                header = Page(0)
                header.page_type = "header"
                f.write(header.data)

    def read_page(self, page_id: int) -> Optional[Page]:
        """Read page from disk"""
        with self.lock:
            try:
                with open(self.filename, "rb") as f:
                    offset = page_id * self.page_size
                    f.seek(offset)
                    data = f.read(self.page_size)

                    if len(data) < self.page_size:
                        return None

                    page = Page(page_id)
                    page.data = bytearray(data)
                    page._read_header()

                    return page
            except IOError:
                return None

    def write_page(self, page: Page) -> bool:
        """Write page to disk"""
        with self.lock:
            try:
                # Update checksum
                page.checksum = page.calculate_checksum()
                struct.pack_into("I", page.data, 44, page.checksum)

                with open(self.filename, "r+b") as f:
                    offset = page.page_id * self.page_size
                    f.seek(offset)
                    f.write(page.data)
                    f.flush()
                    os.fsync(f.fileno())

                return True
            except IOError:
                return False

    def allocate_page(self) -> int:
        """Allocate new page"""
        with self.lock:
            # Simple allocation - find file size
            file_size = os.path.getsize(self.filename)
            new_page_id = file_size // self.page_size

            # Create empty page
            page = Page(new_page_id)

            # Extend file
            with open(self.filename, "ab") as f:
                f.write(page.data)

            return new_page_id


# B+ Tree Implementation


class BPlusTreeNode:
    """B+ tree node for indexing"""

    def __init__(self, order: int, is_leaf: bool = False):
        self.order = order  # Maximum number of keys
        self.is_leaf = is_leaf
        self.keys = []
        self.values = []  # Child pointers (internal) or data pointers (leaf)
        self.next = None  # Next leaf (for leaf nodes)
        self.parent = None

    def is_full(self) -> bool:
        """Check if node is full"""
        return len(self.keys) >= self.order

    def split(self) -> Tuple["BPlusTreeNode", Any]:
        """Split node and return new node and median key"""
        mid = self.order // 2

        if self.is_leaf:
            # Leaf split - keep all keys
            new_node = BPlusTreeNode(self.order, True)
            new_node.keys = self.keys[mid:]
            new_node.values = self.values[mid:]
            new_node.next = self.next
            new_node.parent = self.parent

            self.keys = self.keys[:mid]
            self.values = self.values[:mid]
            self.next = new_node

            return new_node, new_node.keys[0]
        else:
            # Internal node split
            new_node = BPlusTreeNode(self.order, False)
            median_key = self.keys[mid]

            new_node.keys = self.keys[mid + 1 :]
            new_node.values = self.values[mid + 1 :]
            new_node.parent = self.parent

            self.keys = self.keys[:mid]
            self.values = self.values[: mid + 1]

            # Update parent pointers
            for child in new_node.values:
                if isinstance(child, BPlusTreeNode):
                    child.parent = new_node

            return new_node, median_key

    def merge(self, sibling: "BPlusTreeNode", parent_key: Any):
        """Merge with sibling node"""
        if self.is_leaf:
            self.keys.extend(sibling.keys)
            self.values.extend(sibling.values)
            self.next = sibling.next
        else:
            self.keys.append(parent_key)
            self.keys.extend(sibling.keys)
            self.values.extend(sibling.values)

            # Update parent pointers
            for child in sibling.values:
                if isinstance(child, BPlusTreeNode):
                    child.parent = self


class BPlusTree:
    """B+ tree index structure"""

    def __init__(self, order: int = 4):
        self.order = order
        self.root = BPlusTreeNode(order, is_leaf=True)

    def search(self, key: Any) -> Optional[Any]:
        """Search for key in B+ tree"""
        node = self._find_leaf(key)

        try:
            idx = node.keys.index(key)
            return node.values[idx]
        except ValueError:
            return None

    def range_search(self, start_key: Any, end_key: Any) -> List[Tuple[Any, Any]]:
        """Range search using leaf node links"""
        results = []

        # Find starting leaf
        node = self._find_leaf(start_key)

        # Scan through leaf nodes
        while node:
            for i, key in enumerate(node.keys):
                if start_key <= key <= end_key:
                    results.append((key, node.values[i]))
                elif key > end_key:
                    return results

            node = node.next

        return results

    def insert(self, key: Any, value: Any):
        """Insert key-value pair"""
        leaf = self._find_leaf(key)

        # Check if key exists
        if key in leaf.keys:
            # Update value
            idx = leaf.keys.index(key)
            leaf.values[idx] = value
            return

        # Insert into leaf
        self._insert_into_leaf(leaf, key, value)

        # Handle overflow
        if leaf.is_full():
            self._split_and_promote(leaf)

    def delete(self, key: Any) -> bool:
        """Delete key from tree"""
        leaf = self._find_leaf(key)

        if key not in leaf.keys:
            return False

        # Remove from leaf
        idx = leaf.keys.index(key)
        leaf.keys.pop(idx)
        leaf.values.pop(idx)

        # Handle underflow
        if len(leaf.keys) < self.order // 2 and leaf != self.root:
            self._handle_underflow(leaf)

        return True

    def _find_leaf(self, key: Any) -> BPlusTreeNode:
        """Find leaf node for key"""
        node = self.root

        while not node.is_leaf:
            # Binary search for child
            idx = self._binary_search(node.keys, key)
            node = node.values[idx]

        return node

    def _binary_search(self, keys: List[Any], key: Any) -> int:
        """Binary search to find position"""
        left, right = 0, len(keys)

        while left < right:
            mid = (left + right) // 2
            if keys[mid] < key:
                left = mid + 1
            else:
                right = mid

        return left

    def _insert_into_leaf(self, leaf: BPlusTreeNode, key: Any, value: Any):
        """Insert into leaf maintaining sorted order"""
        idx = self._binary_search(leaf.keys, key)
        leaf.keys.insert(idx, key)
        leaf.values.insert(idx, value)

    def _split_and_promote(self, node: BPlusTreeNode):
        """Handle node split and key promotion"""
        new_node, median_key = node.split()

        if node == self.root:
            # Create new root
            new_root = BPlusTreeNode(self.order, is_leaf=False)
            new_root.keys = [median_key]
            new_root.values = [node, new_node]
            node.parent = new_root
            new_node.parent = new_root
            self.root = new_root
        else:
            # Insert median into parent
            parent = node.parent
            idx = self._binary_search(parent.keys, median_key)
            parent.keys.insert(idx, median_key)
            parent.values.insert(idx + 1, new_node)
            new_node.parent = parent

            # Recursively handle parent overflow
            if parent.is_full():
                self._split_and_promote(parent)

    def _handle_underflow(self, node: BPlusTreeNode):
        """Handle node underflow through redistribution or merge"""
        parent = node.parent
        node_idx = parent.values.index(node)

        # Try redistribution with siblings
        if node_idx > 0:
            left_sibling = parent.values[node_idx - 1]
            if len(left_sibling.keys) > self.order // 2:
                self._redistribute_from_left(node, left_sibling, parent, node_idx - 1)
                return

        if node_idx < len(parent.values) - 1:
            right_sibling = parent.values[node_idx + 1]
            if len(right_sibling.keys) > self.order // 2:
                self._redistribute_from_right(node, right_sibling, parent, node_idx)
                return

        # Merge with sibling
        if node_idx > 0:
            left_sibling = parent.values[node_idx - 1]
            self._merge_nodes(left_sibling, node, parent, node_idx - 1)
        else:
            right_sibling = parent.values[node_idx + 1]
            self._merge_nodes(node, right_sibling, parent, node_idx)

    def _redistribute_from_left(
        self,
        node: BPlusTreeNode,
        left_sibling: BPlusTreeNode,
        parent: BPlusTreeNode,
        parent_idx: int,
    ):
        """Redistribute keys from left sibling"""
        if node.is_leaf:
            # Move rightmost key from left sibling
            node.keys.insert(0, left_sibling.keys.pop())
            node.values.insert(0, left_sibling.values.pop())
            parent.keys[parent_idx] = node.keys[0]
        else:
            # Move key through parent
            node.keys.insert(0, parent.keys[parent_idx])
            parent.keys[parent_idx] = left_sibling.keys.pop()

            # Move rightmost child
            child = left_sibling.values.pop()
            node.values.insert(0, child)
            if isinstance(child, BPlusTreeNode):
                child.parent = node

    def _redistribute_from_right(
        self,
        node: BPlusTreeNode,
        right_sibling: BPlusTreeNode,
        parent: BPlusTreeNode,
        parent_idx: int,
    ):
        """Redistribute keys from right sibling"""
        if node.is_leaf:
            # Move leftmost key from right sibling
            node.keys.append(right_sibling.keys.pop(0))
            node.values.append(right_sibling.values.pop(0))
            parent.keys[parent_idx] = right_sibling.keys[0]
        else:
            # Move key through parent
            node.keys.append(parent.keys[parent_idx])
            parent.keys[parent_idx] = right_sibling.keys.pop(0)

            # Move leftmost child
            child = right_sibling.values.pop(0)
            node.values.append(child)
            if isinstance(child, BPlusTreeNode):
                child.parent = node

    def _merge_nodes(
        self,
        left: BPlusTreeNode,
        right: BPlusTreeNode,
        parent: BPlusTreeNode,
        parent_idx: int,
    ):
        """Merge two nodes"""
        # Merge right into left
        parent_key = parent.keys.pop(parent_idx)
        parent.values.pop(parent_idx + 1)

        left.merge(right, parent_key)

        # Handle parent underflow
        if len(parent.keys) < self.order // 2:
            if parent == self.root and len(parent.keys) == 0:
                # Root is empty, promote left
                self.root = left
                left.parent = None
            elif parent != self.root:
                self._handle_underflow(parent)


# Log-Structured Merge Tree (LSM)


@dataclass
class LSMEntry:
    """Entry in LSM tree"""

    key: str
    value: Optional[Any]  # None indicates deletion
    timestamp: int

    def __lt__(self, other):
        return self.key < other.key


class MemTable:
    """In-memory component of LSM tree"""

    def __init__(self, size_limit: int = 1024 * 1024):  # 1MB
        self.data = OrderedDict()  # Maintains insertion order
        self.size = 0
        self.size_limit = size_limit
        self.timestamp_counter = 0
        self.lock = threading.Lock()

    def put(self, key: str, value: Any):
        """Insert or update key-value pair"""
        with self.lock:
            # Calculate size change
            old_size = len(str(self.data.get(key, "")))
            new_size = len(str(value)) if value is not None else 0

            self.size += new_size - old_size

            # Create entry with timestamp
            self.timestamp_counter += 1
            entry = LSMEntry(key, value, self.timestamp_counter)
            self.data[key] = entry

    def get(self, key: str) -> Optional[LSMEntry]:
        """Get value for key"""
        with self.lock:
            return self.data.get(key)

    def delete(self, key: str):
        """Mark key as deleted"""
        self.put(key, None)

    def is_full(self) -> bool:
        """Check if memtable is full"""
        return self.size >= self.size_limit

    def to_sstable(self) -> "SSTable":
        """Convert to sorted string table"""
        with self.lock:
            entries = list(self.data.values())
            entries.sort(key=lambda e: e.key)
            return SSTable(entries)

    def clear(self):
        """Clear memtable"""
        with self.lock:
            self.data.clear()
            self.size = 0


class BloomFilter:
    """Bloom filter for SSTable"""

    def __init__(self, size: int, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [False] * size

    def _hash(self, key: str, seed: int) -> int:
        """Generate hash with seed"""
        h = hashlib.sha256()
        h.update(f"{key}{seed}".encode())
        return int.from_bytes(h.digest()[:4], "big") % self.size

    def add(self, key: str):
        """Add key to filter"""
        for i in range(self.num_hashes):
            idx = self._hash(key, i)
            self.bits[idx] = True

    def might_contain(self, key: str) -> bool:
        """Check if key might be in set"""
        for i in range(self.num_hashes):
            idx = self._hash(key, i)
            if not self.bits[idx]:
                return False
        return True


class SSTable:
    """Sorted String Table - immutable on-disk component"""

    def __init__(self, entries: List[LSMEntry]):
        self.entries = entries
        self.index = self._build_index()
        self.bloom_filter = self._build_bloom_filter()
        self.min_key = entries[0].key if entries else None
        self.max_key = entries[-1].key if entries else None

    def _build_index(self) -> Dict[str, int]:
        """Build sparse index for binary search"""
        index = {}
        for i in range(0, len(self.entries), 100):  # Index every 100th key
            index[self.entries[i].key] = i
        return index

    def _build_bloom_filter(self) -> BloomFilter:
        """Build bloom filter for existence checks"""
        bf = BloomFilter(max(1000, len(self.entries) * 10))
        for entry in self.entries:
            if entry.value is not None:  # Don't add deleted keys
                bf.add(entry.key)
        return bf

    def get(self, key: str) -> Optional[LSMEntry]:
        """Binary search for key"""
        if not self.bloom_filter.might_contain(key):
            return None

        left, right = 0, len(self.entries) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_key = self.entries[mid].key

            if mid_key == key:
                return self.entries[mid]
            elif mid_key < key:
                left = mid + 1
            else:
                right = mid - 1

        return None

    def range_scan(self, start_key: str, end_key: str) -> List[LSMEntry]:
        """Scan entries in key range"""
        results = []

        # Find starting position
        start_idx = 0
        for i, entry in enumerate(self.entries):
            if entry.key >= start_key:
                start_idx = i
                break

        # Collect entries in range
        for i in range(start_idx, len(self.entries)):
            if self.entries[i].key > end_key:
                break
            results.append(self.entries[i])

        return results

    def overlaps(self, other: "SSTable") -> bool:
        """Check if key ranges overlap"""
        if not self.entries or not other.entries:
            return False

        return not (self.max_key < other.min_key or self.min_key > other.max_key)


class LSMTree:
    """Log-Structured Merge Tree implementation"""

    def __init__(self, base_dir: str = "lsm_data"):
        self.base_dir = base_dir
        self.memtable = MemTable()
        self.immutable_memtable = None
        self.levels = [[] for _ in range(7)]  # L0 to L6
        self.level_size_ratios = [1, 10, 100, 1000, 10000, 100000, 1000000]
        self.lock = threading.Lock()

        # Create directory
        os.makedirs(base_dir, exist_ok=True)

    def put(self, key: str, value: Any):
        """Insert or update key-value pair"""
        self.memtable.put(key, value)

        if self.memtable.is_full():
            self._flush_memtable()

    def get(self, key: str) -> Optional[Any]:
        """Get value for key"""
        # Check memtable first
        entry = self.memtable.get(key)
        if entry:
            return entry.value

        # Check immutable memtable
        with self.lock:
            if self.immutable_memtable:
                entry = self.immutable_memtable.get(key)
                if entry:
                    return entry.value

        # Check SSTables from newest to oldest
        for level in range(len(self.levels)):
            for sstable in reversed(self.levels[level]):
                entry = sstable.get(key)
                if entry:
                    return entry.value

        return None

    def delete(self, key: str):
        """Delete key"""
        self.memtable.delete(key)

        if self.memtable.is_full():
            self._flush_memtable()

    def range_scan(self, start_key: str, end_key: str) -> List[Tuple[str, Any]]:
        """Scan key range"""
        # Merge results from all components
        results = {}

        # Add from memtable
        for key, entry in self.memtable.data.items():
            if start_key <= key <= end_key and entry.value is not None:
                results[key] = (entry.value, entry.timestamp)

        # Add from SSTables (older to newer to get latest values)
        for level in reversed(range(len(self.levels))):
            for sstable in self.levels[level]:
                for entry in sstable.range_scan(start_key, end_key):
                    if (
                        entry.key not in results
                        or results[entry.key][1] < entry.timestamp
                    ):
                        if entry.value is not None:
                            results[entry.key] = (entry.value, entry.timestamp)
                        elif entry.key in results:
                            del results[entry.key]  # Handle deletion

        # Return sorted results
        return [(k, v[0]) for k, v in sorted(results.items())]

    def _flush_memtable(self):
        """Flush memtable to level 0"""
        with self.lock:
            self.immutable_memtable = self.memtable
            self.memtable = MemTable()

        # Convert to SSTable
        sstable = self.immutable_memtable.to_sstable()

        with self.lock:
            self.levels[0].append(sstable)
            self.immutable_memtable = None

        # Save SSTable to disk
        self._save_sstable(sstable, 0, len(self.levels[0]) - 1)

        # Trigger compaction if needed
        self._maybe_compact(0)

    def _maybe_compact(self, level: int):
        """Compact level if it exceeds size limit"""
        max_tables = self.level_size_ratios[level]

        if len(self.levels[level]) > max_tables:
            self._compact_level(level)

    def _compact_level(self, level: int):
        """Merge SSTables from level to level+1"""
        if level >= len(self.levels) - 1:
            return

        with self.lock:
            # Select tables to compact (simple strategy: all tables)
            tables_to_compact = self.levels[level][:]

            # Find overlapping tables in next level
            overlapping_tables = []
            for table in self.levels[level + 1]:
                for compact_table in tables_to_compact:
                    if table.overlaps(compact_table):
                        overlapping_tables.append(table)
                        break

            # Remove from levels
            self.levels[level] = []
            for table in overlapping_tables:
                self.levels[level + 1].remove(table)

        # Merge all tables
        all_entries = []
        for table in tables_to_compact + overlapping_tables:
            all_entries.extend(table.entries)

        # Sort and deduplicate (keep latest version)
        all_entries.sort(key=lambda e: (e.key, -e.timestamp))

        # Remove duplicates
        merged_entries = []
        last_key = None
        for entry in all_entries:
            if entry.key != last_key:
                merged_entries.append(entry)
                last_key = entry.key

        # Create new SSTables (split if too large)
        max_size = 1000  # Max entries per SSTable
        new_sstables = []

        for i in range(0, len(merged_entries), max_size):
            entries = merged_entries[i : i + max_size]
            new_sstables.append(SSTable(entries))

        # Add to next level
        with self.lock:
            self.levels[level + 1].extend(new_sstables)

        # Save new SSTables
        for i, sstable in enumerate(new_sstables):
            self._save_sstable(
                sstable, level + 1, len(self.levels[level + 1]) - len(new_sstables) + i
            )

        # Recursively compact next level
        self._maybe_compact(level + 1)

    def _save_sstable(self, sstable: SSTable, level: int, index: int):
        """Save SSTable to disk"""
        filename = os.path.join(self.base_dir, f"L{level}_{index}.sst")
        # In real implementation, would serialize to disk
        # For demo, we keep in memory

    def _load_sstable(self, level: int, index: int) -> SSTable:
        """Load SSTable from disk"""
        filename = os.path.join(self.base_dir, f"L{level}_{index}.sst")
        # In real implementation, would deserialize from disk
        return SSTable([])


# Example demonstrations


def demonstrate_page_management():
    """Show page and buffer pool management"""
    print("Page Management Demonstration:")

    # Create storage manager
    storage = StorageManager("test.db")

    # Create buffer pool
    buffer_pool = BufferPool(10, storage)

    # Allocate and use pages
    page_id = storage.allocate_page()
    page = buffer_pool.fetch_page(page_id)

    if page:
        # Insert records
        record1 = b"John Doe,25,Engineer"
        slot1 = page.insert_record(record1)
        print(f"Inserted record at slot {slot1}")

        record2 = b"Jane Smith,30,Manager"
        slot2 = page.insert_record(record2)
        print(f"Inserted record at slot {slot2}")

        # Read records
        data = page.get_record(slot1)
        print(f"Read record: {data.decode()}")

        # Mark page dirty and unpin
        buffer_pool.unpin_page(page_id, is_dirty=True)

        # Flush to disk
        buffer_pool.flush_page(page_id)
        print("Page flushed to disk")

    # Clean up
    os.remove("test.db")


def demonstrate_btree():
    """Show B+ tree operations"""
    print("\nB+ Tree Demonstration:")

    # Create B+ tree
    btree = BPlusTree(order=4)

    # Insert data
    data = [(i, f"value_{i}") for i in range(1, 21)]
    random.shuffle(data)

    for key, value in data:
        btree.insert(key, value)
        print(f"Inserted {key}")

    # Search
    result = btree.search(10)
    print(f"\nSearch for key 10: {result}")

    # Range search
    results = btree.range_search(5, 15)
    print(f"\nRange search [5, 15]: {results}")

    # Delete
    btree.delete(10)
    print("\nDeleted key 10")

    result = btree.search(10)
    print(f"Search for key 10 after deletion: {result}")


def demonstrate_lsm():
    """Show LSM tree operations"""
    print("\nLSM Tree Demonstration:")

    # Create LSM tree
    lsm = LSMTree("test_lsm")

    # Insert data
    for i in range(100):
        lsm.put(f"key_{i:03d}", f"value_{i}")

    print("Inserted 100 key-value pairs")

    # Read data
    value = lsm.get("key_050")
    print(f"\nGet key_050: {value}")

    # Update
    lsm.put("key_050", "updated_value")
    value = lsm.get("key_050")
    print(f"After update: {value}")

    # Delete
    lsm.delete("key_025")
    value = lsm.get("key_025")
    print(f"\nAfter deleting key_025: {value}")

    # Range scan
    results = lsm.range_scan("key_010", "key_020")
    print(f"\nRange scan [key_010, key_020]:")
    for k, v in results:
        print(f"  {k}: {v}")

    # Clean up
    import shutil

    shutil.rmtree("test_lsm")


if __name__ == "__main__":
    demonstrate_page_management()
    demonstrate_btree()
    demonstrate_lsm()
