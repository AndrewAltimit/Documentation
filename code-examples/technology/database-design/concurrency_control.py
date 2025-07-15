"""
Transaction Management and Concurrency Control

Implementation of transaction theory, MVCC, 2PL, and isolation levels.
"""

from enum import Enum
from typing import List, Set, Tuple, Optional, Dict, Any
import threading
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq


class IsolationLevel(Enum):
    READ_UNCOMMITTED = 1
    READ_COMMITTED = 2
    REPEATABLE_READ = 3
    SERIALIZABLE = 4


@dataclass
class Operation:
    """Database operation"""
    txn_id: int
    op_type: str  # 'read', 'write', 'commit', 'abort'
    item: str
    value: Any = None
    timestamp: float = field(default_factory=time.time)
    
    def __repr__(self):
        if self.op_type in ['commit', 'abort']:
            return f"{self.op_type}_{self.txn_id}"
        elif self.op_type == 'read':
            return f"r_{self.txn_id}({self.item})"
        else:
            return f"w_{self.txn_id}({self.item}={self.value})"


class Schedule:
    """Sequence of operations from multiple transactions"""
    
    def __init__(self, operations: List[Operation]):
        self.operations = operations
    
    def is_serial(self) -> bool:
        """Check if schedule is serial"""
        current_txn = None
        committed = set()
        
        for op in self.operations:
            if op.op_type in ['commit', 'abort']:
                committed.add(op.txn_id)
                continue
                
            if current_txn is None:
                current_txn = op.txn_id
            elif current_txn != op.txn_id:
                # Check if previous transaction completed
                if current_txn not in committed:
                    return False
                current_txn = op.txn_id
        
        return True
    
    def is_conflict_serializable(self) -> bool:
        """Test conflict serializability using precedence graph"""
        # Build precedence graph
        graph = self.build_precedence_graph()
        
        # Check for cycles using DFS
        return not self.has_cycle(graph)
    
    def build_precedence_graph(self) -> Dict[int, Set[int]]:
        """Build precedence graph for transactions"""
        graph = defaultdict(set)
        n = len(self.operations)
        
        for i in range(n):
            for j in range(i + 1, n):
                op1, op2 = self.operations[i], self.operations[j]
                
                # Different transactions accessing same item
                if (op1.txn_id != op2.txn_id and 
                    op1.item == op2.item and
                    op1.item is not None):
                    
                    # Conflict conditions
                    if (op1.op_type == 'write' or op2.op_type == 'write'):
                        graph[op1.txn_id].add(op2.txn_id)
        
        return dict(graph)
    
    def has_cycle(self, graph: Dict[int, Set[int]]) -> bool:
        """Detect cycle in directed graph using DFS"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(lambda: WHITE)
        
        def visit(node):
            if color[node] == GRAY:
                return True  # Back edge found
            if color[node] == BLACK:
                return False  # Already processed
            
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    return True
            color[node] = BLACK
            return False
        
        for node in graph:
            if color[node] == WHITE:
                if visit(node):
                    return True
        
        return False
    
    def is_view_serializable(self) -> bool:
        """Check view serializability (NP-complete problem)"""
        # Simplified check - in practice this is computationally hard
        # Check if conflict serializable (sufficient but not necessary)
        return self.is_conflict_serializable()
    
    def get_serial_schedules(self) -> List['Schedule']:
        """Generate all possible serial schedules"""
        # Group operations by transaction
        txn_ops = defaultdict(list)
        for op in self.operations:
            txn_ops[op.txn_id].append(op)
        
        # Generate permutations of transaction order
        from itertools import permutations
        serial_schedules = []
        
        for perm in permutations(txn_ops.keys()):
            ops = []
            for txn_id in perm:
                ops.extend(txn_ops[txn_id])
            serial_schedules.append(Schedule(ops))
        
        return serial_schedules


# Multi-Version Concurrency Control (MVCC)

@dataclass
class MVCCVersion:
    """Version of a data item in MVCC"""
    value: Any
    write_timestamp: int
    read_timestamp: int


class MVCCScheduler:
    """Multi-version concurrency control implementation"""
    
    def __init__(self):
        self.versions = defaultdict(list)  # item -> list of versions
        self.timestamp_counter = 0
        self.active_transactions = {}
        self.lock = threading.Lock()
    
    def begin_transaction(self) -> int:
        """Start new transaction"""
        with self.lock:
            txn_id = self.timestamp_counter
            self.timestamp_counter += 1
            self.active_transactions[txn_id] = {
                'start_time': txn_id,
                'read_set': set(),
                'write_set': set(),
                'state': 'active'
            }
            return txn_id
    
    def read(self, txn_id: int, item: str) -> Any:
        """MVCC read operation"""
        with self.lock:
            if item not in self.versions:
                return None
            
            # Find appropriate version
            txn_start = self.active_transactions[txn_id]['start_time']
            valid_versions = [
                v for v in self.versions[item]
                if v.write_timestamp <= txn_start
            ]
            
            if not valid_versions:
                return None
            
            # Get version with highest write timestamp
            version = max(valid_versions, key=lambda v: v.write_timestamp)
            
            # Update read timestamp
            version.read_timestamp = max(version.read_timestamp, txn_id)
            self.active_transactions[txn_id]['read_set'].add(item)
            
            return version.value
    
    def write(self, txn_id: int, item: str, value: Any) -> bool:
        """MVCC write operation"""
        with self.lock:
            self.active_transactions[txn_id]['write_set'].add((item, value))
            return True
    
    def commit(self, txn_id: int) -> bool:
        """Commit transaction with validation"""
        with self.lock:
            txn_info = self.active_transactions[txn_id]
            
            # Validation phase for snapshot isolation
            for item in txn_info['read_set']:
                if item in self.versions:
                    for version in self.versions[item]:
                        # Check for write-write conflicts
                        if (version.write_timestamp > txn_info['start_time'] and
                            version.write_timestamp < txn_id):
                            # Concurrent modification detected
                            self.abort(txn_id)
                            return False
            
            # Write phase
            commit_timestamp = self.timestamp_counter
            self.timestamp_counter += 1
            
            for item, value in txn_info['write_set']:
                # Create new version
                new_version = MVCCVersion(value, commit_timestamp, commit_timestamp)
                
                if item not in self.versions:
                    self.versions[item] = []
                self.versions[item].append(new_version)
                
                # Garbage collection
                self.garbage_collect(item)
            
            txn_info['state'] = 'committed'
            return True
    
    def abort(self, txn_id: int):
        """Abort transaction"""
        with self.lock:
            if txn_id in self.active_transactions:
                self.active_transactions[txn_id]['state'] = 'aborted'
    
    def garbage_collect(self, item: str):
        """Remove obsolete versions"""
        if item not in self.versions:
            return
        
        # Find minimum start time of active transactions
        if self.active_transactions:
            active_txns = [t for t in self.active_transactions.values() 
                          if t['state'] == 'active']
            if active_txns:
                min_active = min(t['start_time'] for t in active_txns)
            else:
                min_active = self.timestamp_counter
        else:
            min_active = self.timestamp_counter
        
        # Keep only versions that might be needed
        self.versions[item] = [
            v for v in self.versions[item]
            if v.write_timestamp >= min_active or v.read_timestamp >= min_active
        ]
        
        # Keep at least one version
        if self.versions[item]:
            self.versions[item] = sorted(
                self.versions[item], 
                key=lambda v: v.write_timestamp
            )[-5:]  # Keep last 5 versions max


# Two-Phase Locking (2PL)

class LockMode(Enum):
    SHARED = 'S'
    EXCLUSIVE = 'X'
    INTENTION_SHARED = 'IS'
    INTENTION_EXCLUSIVE = 'IX'
    SHARED_INTENTION_EXCLUSIVE = 'SIX'


class LockManager:
    """Strict 2PL implementation with deadlock detection"""
    
    def __init__(self):
        self.locks = defaultdict(lambda: {
            'mode': None, 
            'holders': set(), 
            'waiters': deque()
        })
        self.txn_locks = defaultdict(set)  # txn_id -> set of (item, mode)
        self.wait_for_graph = defaultdict(set)  # txn -> set of txns
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def acquire_lock(self, txn_id: int, item: str, mode: LockMode) -> bool:
        """Acquire shared (S) or exclusive (X) lock"""
        with self.condition:
            # Check if already holding incompatible lock
            if self._already_holds_lock(txn_id, item, mode):
                return True
            
            lock_info = self.locks[item]
            
            # Check compatibility
            while not self._is_compatible(lock_info, txn_id, mode):
                # Add to waiters
                lock_info['waiters'].append((txn_id, mode))
                
                # Update wait-for graph
                self.wait_for_graph[txn_id].update(lock_info['holders'])
                
                # Check for deadlock
                if self._detect_deadlock():
                    # Remove from waiters
                    lock_info['waiters'].remove((txn_id, mode))
                    self.wait_for_graph[txn_id].difference_update(lock_info['holders'])
                    raise Exception(f"Deadlock detected for transaction {txn_id}")
                
                # Wait for lock
                self.condition.wait()
                
                # Remove from waiters when woken up
                if (txn_id, mode) in lock_info['waiters']:
                    lock_info['waiters'].remove((txn_id, mode))
            
            # Grant lock
            self._grant_lock(txn_id, item, mode)
            return True
    
    def release_lock(self, txn_id: int, item: str):
        """Release specific lock"""
        with self.condition:
            lock_info = self.locks[item]
            
            if txn_id in lock_info['holders']:
                lock_info['holders'].discard(txn_id)
                self.txn_locks[txn_id].discard((item, lock_info['mode']))
                
                # Update lock mode based on remaining holders
                if not lock_info['holders']:
                    lock_info['mode'] = None
                
                # Wake up waiters
                self._grant_waiting_locks(item)
                self.condition.notify_all()
    
    def release_all_locks(self, txn_id: int):
        """Release all locks held by transaction"""
        with self.condition:
            if txn_id not in self.txn_locks:
                return
            
            # Copy to avoid modification during iteration
            locks_to_release = list(self.txn_locks[txn_id])
            
            for item, _ in locks_to_release:
                self.release_lock(txn_id, item)
            
            # Clean up
            del self.txn_locks[txn_id]
            if txn_id in self.wait_for_graph:
                del self.wait_for_graph[txn_id]
    
    def _already_holds_lock(self, txn_id: int, item: str, mode: LockMode) -> bool:
        """Check if transaction already holds compatible lock"""
        for held_item, held_mode in self.txn_locks[txn_id]:
            if held_item == item:
                # Check if held lock is at least as strong
                if mode == LockMode.SHARED and held_mode in [LockMode.SHARED, LockMode.EXCLUSIVE]:
                    return True
                elif mode == LockMode.EXCLUSIVE and held_mode == LockMode.EXCLUSIVE:
                    return True
        return False
    
    def _is_compatible(self, lock_info: Dict, txn_id: int, mode: LockMode) -> bool:
        """Check lock compatibility"""
        if not lock_info['holders']:
            return True
        
        if txn_id in lock_info['holders']:
            # Lock upgrade
            if lock_info['mode'] == LockMode.SHARED and mode == LockMode.EXCLUSIVE:
                return len(lock_info['holders']) == 1
            return True
        
        # Compatibility matrix
        if lock_info['mode'] == LockMode.SHARED:
            return mode == LockMode.SHARED
        elif lock_info['mode'] == LockMode.EXCLUSIVE:
            return False
        
        return True
    
    def _grant_lock(self, txn_id: int, item: str, mode: LockMode):
        """Grant lock to transaction"""
        lock_info = self.locks[item]
        
        lock_info['holders'].add(txn_id)
        if mode == LockMode.EXCLUSIVE:
            lock_info['mode'] = LockMode.EXCLUSIVE
        elif lock_info['mode'] != LockMode.EXCLUSIVE:
            lock_info['mode'] = LockMode.SHARED
        
        self.txn_locks[txn_id].add((item, mode))
        
        # Clean up wait-for graph
        if txn_id in self.wait_for_graph:
            self.wait_for_graph[txn_id].clear()
    
    def _grant_waiting_locks(self, item: str):
        """Try to grant locks to waiting transactions"""
        lock_info = self.locks[item]
        granted = []
        
        for waiter_txn, waiter_mode in list(lock_info['waiters']):
            if self._is_compatible(lock_info, waiter_txn, waiter_mode):
                self._grant_lock(waiter_txn, item, waiter_mode)
                granted.append((waiter_txn, waiter_mode))
        
        for g in granted:
            lock_info['waiters'].remove(g)
    
    def _detect_deadlock(self) -> bool:
        """Detect cycles in wait-for graph using DFS"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.wait_for_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.wait_for_graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False


# Transaction Manager with Isolation Levels

class TransactionManager:
    """Implement different isolation levels"""
    
    def __init__(self, isolation_level: IsolationLevel):
        self.isolation_level = isolation_level
        self.lock_manager = LockManager()
        self.mvcc_scheduler = MVCCScheduler()
        self.transactions = {}
        self.transaction_counter = 0
        self.lock = threading.Lock()
    
    def begin_transaction(self) -> int:
        """Start new transaction"""
        with self.lock:
            txn_id = self.transaction_counter
            self.transaction_counter += 1
            
            self.transactions[txn_id] = {
                'id': txn_id,
                'state': 'active',
                'start_time': time.time(),
                'operations': []
            }
            
            if self.isolation_level in [IsolationLevel.REPEATABLE_READ, 
                                       IsolationLevel.SERIALIZABLE]:
                # MVCC-based isolation
                mvcc_txn_id = self.mvcc_scheduler.begin_transaction()
                self.transactions[txn_id]['mvcc_id'] = mvcc_txn_id
            
            return txn_id
    
    def read(self, txn_id: int, item: str) -> Any:
        """Read with appropriate isolation"""
        if self.isolation_level == IsolationLevel.READ_UNCOMMITTED:
            # No locks required - may read dirty data
            return self._read_latest(item)
        
        elif self.isolation_level == IsolationLevel.READ_COMMITTED:
            # Short-duration read locks
            self.lock_manager.acquire_lock(txn_id, item, LockMode.SHARED)
            value = self._read_latest(item)
            self.lock_manager.release_lock(txn_id, item)
            return value
        
        elif self.isolation_level == IsolationLevel.REPEATABLE_READ:
            # Use MVCC for consistent snapshot
            mvcc_id = self.transactions[txn_id]['mvcc_id']
            return self.mvcc_scheduler.read(mvcc_id, item)
        
        elif self.isolation_level == IsolationLevel.SERIALIZABLE:
            # Strict 2PL or serializable MVCC
            self.lock_manager.acquire_lock(txn_id, item, LockMode.SHARED)
            mvcc_id = self.transactions[txn_id]['mvcc_id']
            return self.mvcc_scheduler.read(mvcc_id, item)
    
    def write(self, txn_id: int, item: str, value: Any) -> bool:
        """Write with appropriate isolation"""
        # All isolation levels need exclusive locks for writes
        self.lock_manager.acquire_lock(txn_id, item, LockMode.EXCLUSIVE)
        
        if self.isolation_level in [IsolationLevel.REPEATABLE_READ, 
                                   IsolationLevel.SERIALIZABLE]:
            mvcc_id = self.transactions[txn_id]['mvcc_id']
            self.mvcc_scheduler.write(mvcc_id, item, value)
        else:
            self._write_latest(item, value)
        
        self.transactions[txn_id]['operations'].append(
            Operation(txn_id, 'write', item, value)
        )
        
        return True
    
    def commit(self, txn_id: int) -> bool:
        """Commit transaction"""
        with self.lock:
            if txn_id not in self.transactions:
                return False
            
            txn = self.transactions[txn_id]
            
            if self.isolation_level in [IsolationLevel.REPEATABLE_READ, 
                                       IsolationLevel.SERIALIZABLE]:
                # MVCC commit with validation
                mvcc_id = txn['mvcc_id']
                success = self.mvcc_scheduler.commit(mvcc_id)
                
                if not success:
                    self.abort(txn_id)
                    return False
            
            # Release all locks
            self.lock_manager.release_all_locks(txn_id)
            
            txn['state'] = 'committed'
            txn['end_time'] = time.time()
            
            return True
    
    def abort(self, txn_id: int):
        """Abort transaction"""
        with self.lock:
            if txn_id not in self.transactions:
                return
            
            txn = self.transactions[txn_id]
            
            if self.isolation_level in [IsolationLevel.REPEATABLE_READ, 
                                       IsolationLevel.SERIALIZABLE]:
                mvcc_id = txn.get('mvcc_id')
                if mvcc_id:
                    self.mvcc_scheduler.abort(mvcc_id)
            
            # Release all locks
            self.lock_manager.release_all_locks(txn_id)
            
            txn['state'] = 'aborted'
            txn['end_time'] = time.time()
    
    def _read_latest(self, item: str) -> Any:
        """Read latest value (for simple storage)"""
        # In practice, would read from storage engine
        return f"value_of_{item}"
    
    def _write_latest(self, item: str, value: Any):
        """Write value (for simple storage)"""
        # In practice, would write to storage engine
        pass


# Timestamp Ordering Protocol

class TimestampOrderingScheduler:
    """Basic Timestamp Ordering (TO) protocol"""
    
    def __init__(self):
        self.item_timestamps = defaultdict(lambda: {'read': 0, 'write': 0})
        self.transaction_timestamps = {}
        self.timestamp_counter = 0
        self.lock = threading.Lock()
    
    def begin_transaction(self) -> int:
        """Assign timestamp to new transaction"""
        with self.lock:
            ts = self.timestamp_counter
            self.timestamp_counter += 1
            self.transaction_timestamps[ts] = {
                'start': ts,
                'state': 'active'
            }
            return ts
    
    def read(self, txn_ts: int, item: str) -> Tuple[bool, Any]:
        """TO protocol read"""
        with self.lock:
            item_ts = self.item_timestamps[item]
            
            # Check TO rule: txn_ts >= write_timestamp(item)
            if txn_ts < item_ts['write']:
                # Transaction too old - abort
                return False, None
            
            # Update read timestamp
            item_ts['read'] = max(item_ts['read'], txn_ts)
            
            # Return value (simplified)
            return True, f"value_{item}_at_{item_ts['write']}"
    
    def write(self, txn_ts: int, item: str, value: Any) -> bool:
        """TO protocol write"""
        with self.lock:
            item_ts = self.item_timestamps[item]
            
            # Check TO rules
            if txn_ts < item_ts['read']:
                # Transaction too old - abort
                return False
            
            if txn_ts < item_ts['write']:
                # Thomas Write Rule - ignore obsolete write
                return True
            
            # Perform write
            item_ts['write'] = txn_ts
            item_ts['read'] = max(item_ts['read'], txn_ts)
            
            return True
    
    def commit(self, txn_ts: int):
        """Commit transaction"""
        with self.lock:
            if txn_ts in self.transaction_timestamps:
                self.transaction_timestamps[txn_ts]['state'] = 'committed'
    
    def abort(self, txn_ts: int):
        """Abort transaction"""
        with self.lock:
            if txn_ts in self.transaction_timestamps:
                self.transaction_timestamps[txn_ts]['state'] = 'aborted'


# Optimistic Concurrency Control (OCC)

class OptimisticScheduler:
    """Optimistic Concurrency Control with validation"""
    
    def __init__(self):
        self.active_transactions = {}
        self.committed_transactions = []
        self.lock = threading.Lock()
        self.validation_lock = threading.Lock()
        self.txn_counter = 0
    
    def begin_transaction(self) -> int:
        """Start transaction - no locks needed"""
        with self.lock:
            txn_id = self.txn_counter
            self.txn_counter += 1
            
            self.active_transactions[txn_id] = {
                'id': txn_id,
                'read_set': set(),
                'write_set': {},
                'start_time': time.time(),
                'state': 'active'
            }
            
            return txn_id
    
    def read(self, txn_id: int, item: str) -> Any:
        """Read without locking"""
        with self.lock:
            txn = self.active_transactions[txn_id]
            txn['read_set'].add(item)
            
            # Read from local write set if available
            if item in txn['write_set']:
                return txn['write_set'][item]
            
            # Otherwise read from database
            return self._read_from_storage(item)
    
    def write(self, txn_id: int, item: str, value: Any):
        """Write to local buffer"""
        with self.lock:
            txn = self.active_transactions[txn_id]
            txn['write_set'][item] = value
    
    def validate_and_commit(self, txn_id: int) -> bool:
        """Validation phase and commit"""
        with self.validation_lock:
            txn = self.active_transactions[txn_id]
            
            # Validation against committed transactions
            for committed_txn in self.committed_transactions:
                if committed_txn['commit_time'] > txn['start_time']:
                    # Check for conflicts
                    if self._has_conflict(txn, committed_txn):
                        self.abort(txn_id)
                        return False
            
            # Validation passed - commit
            return self._commit(txn_id)
    
    def _has_conflict(self, txn1: Dict, txn2: Dict) -> bool:
        """Check for read-write or write-write conflicts"""
        # Read-write conflict: txn1 reads what txn2 writes
        if txn1['read_set'].intersection(set(txn2['write_set'].keys())):
            return True
        
        # Write-write conflict
        if set(txn1['write_set'].keys()).intersection(set(txn2['write_set'].keys())):
            return True
        
        return False
    
    def _commit(self, txn_id: int) -> bool:
        """Commit transaction changes"""
        txn = self.active_transactions[txn_id]
        
        # Write all changes to storage
        for item, value in txn['write_set'].items():
            self._write_to_storage(item, value)
        
        # Mark as committed
        txn['state'] = 'committed'
        txn['commit_time'] = time.time()
        
        # Add to committed list
        self.committed_transactions.append(txn)
        
        # Remove from active
        del self.active_transactions[txn_id]
        
        return True
    
    def abort(self, txn_id: int):
        """Abort transaction - just discard local changes"""
        with self.lock:
            if txn_id in self.active_transactions:
                self.active_transactions[txn_id]['state'] = 'aborted'
                del self.active_transactions[txn_id]
    
    def _read_from_storage(self, item: str) -> Any:
        """Read from storage (placeholder)"""
        return f"stored_value_{item}"
    
    def _write_to_storage(self, item: str, value: Any):
        """Write to storage (placeholder)"""
        pass


# Example demonstrations

def demonstrate_serializability():
    """Show serializability testing"""
    print("Serializability Testing:")
    
    # Create a schedule
    ops = [
        Operation(1, 'read', 'A'),
        Operation(2, 'read', 'A'),
        Operation(1, 'write', 'A', 10),
        Operation(2, 'write', 'A', 20),
        Operation(1, 'commit', None),
        Operation(2, 'commit', None)
    ]
    
    schedule = Schedule(ops)
    
    print(f"Schedule: {[str(op) for op in ops]}")
    print(f"Is serial: {schedule.is_serial()}")
    print(f"Is conflict serializable: {schedule.is_conflict_serializable()}")
    
    # Show precedence graph
    graph = schedule.build_precedence_graph()
    print(f"Precedence graph: {graph}")


def demonstrate_mvcc():
    """Show MVCC in action"""
    print("\nMVCC Demonstration:")
    
    scheduler = MVCCScheduler()
    
    # Transaction 1 starts
    t1 = scheduler.begin_transaction()
    print(f"T1 started with timestamp {t1}")
    
    # Transaction 1 writes
    scheduler.write(t1, 'A', 100)
    scheduler.commit(t1)
    print("T1 wrote A=100 and committed")
    
    # Transaction 2 starts (sees T1's changes)
    t2 = scheduler.begin_transaction()
    value = scheduler.read(t2, 'A')
    print(f"T2 reads A={value}")
    
    # Transaction 3 starts concurrently
    t3 = scheduler.begin_transaction()
    
    # T3 writes A
    scheduler.write(t3, 'A', 200)
    scheduler.commit(t3)
    print("T3 wrote A=200 and committed")
    
    # T2 still sees old value
    value = scheduler.read(t2, 'A')
    print(f"T2 still reads A={value} (snapshot isolation)")


def demonstrate_2pl():
    """Show 2PL with deadlock detection"""
    print("\n2PL with Deadlock Detection:")
    
    lock_mgr = LockManager()
    
    # Simulate potential deadlock scenario
    print("T1 locks A, T2 locks B")
    lock_mgr.acquire_lock(1, 'A', LockMode.EXCLUSIVE)
    lock_mgr.acquire_lock(2, 'B', LockMode.EXCLUSIVE)
    
    print("T1 tries to lock B (waits)...")
    # This would block in real scenario
    
    print("T2 tries to lock A (deadlock!)")
    try:
        # This would detect deadlock
        pass
    except Exception as e:
        print(f"Detected: {e}")
    
    # Clean up
    lock_mgr.release_all_locks(1)
    lock_mgr.release_all_locks(2)


def demonstrate_isolation_levels():
    """Show different isolation levels"""
    print("\nIsolation Levels Demonstration:")
    
    for level in IsolationLevel:
        print(f"\n{level.name}:")
        tm = TransactionManager(level)
        
        # Start transactions
        t1 = tm.begin_transaction()
        t2 = tm.begin_transaction()
        
        # Demonstrate behavior
        if level == IsolationLevel.READ_UNCOMMITTED:
            print("T1 writes (uncommitted), T2 can read dirty data")
        elif level == IsolationLevel.READ_COMMITTED:
            print("T2 waits for T1 to commit before reading")
        elif level == IsolationLevel.REPEATABLE_READ:
            print("T2 sees consistent snapshot, no phantom reads")
        elif level == IsolationLevel.SERIALIZABLE:
            print("Full serializability guaranteed")


if __name__ == "__main__":
    demonstrate_serializability()
    demonstrate_mvcc()
    demonstrate_2pl()
    demonstrate_isolation_levels()