"""
Distributed Database Systems

Implementation of CAP theorem concepts, consensus algorithms (Raft),
distributed transactions (2PC), and vector clocks.
"""

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Set, Tuple


class ConsistencyLevel(Enum):
    EVENTUAL = 1
    STRONG = 2
    BOUNDED_STALENESS = 3
    SESSION = 4
    CONSISTENT_PREFIX = 5


class CAP_System:
    """Demonstrate CAP theorem trade-offs"""

    def __init__(
        self,
        consistency: ConsistencyLevel,
        availability: bool,
        partition_tolerance: bool,
    ):
        # CAP theorem: can only guarantee 2 of 3
        if consistency == ConsistencyLevel.STRONG:
            assert not (
                availability and partition_tolerance
            ), "Cannot have strong consistency with both availability and partition tolerance"

        self.consistency = consistency
        self.availability = availability
        self.partition_tolerance = partition_tolerance
        self.nodes = {}
        self.network_partitioned = False

    def demonstrate_trade_offs(self):
        """Show different system behaviors under CAP constraints"""
        if self.consistency == ConsistencyLevel.STRONG and self.partition_tolerance:
            # CP system - Sacrifice availability for consistency
            return (
                "CP System: May refuse writes during partition to maintain consistency"
            )

        elif self.availability and self.partition_tolerance:
            # AP system - Sacrifice consistency for availability
            return (
                "AP System: Always available but may return stale data during partition"
            )

        elif self.consistency == ConsistencyLevel.STRONG and self.availability:
            # CA system - No partition tolerance
            return (
                "CA System: Consistent and available but fails under network partition"
            )


class VectorClock:
    """Vector clock for distributed causality tracking"""

    def __init__(self, node_id: str, num_nodes: int):
        self.node_id = node_id
        self.clock = {f"node_{i}": 0 for i in range(num_nodes)}

    def increment(self):
        """Increment local counter"""
        self.clock[self.node_id] += 1

    def update(self, other_clock: Dict[str, int]):
        """Update with received clock"""
        for node, timestamp in other_clock.items():
            self.clock[node] = max(self.clock.get(node, 0), timestamp)
        self.increment()

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this event happens before other"""
        return all(self.clock[k] <= other.clock.get(k, 0) for k in self.clock.keys())

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if events are concurrent"""
        return not self.happens_before(other) and not other.happens_before(self)

    def __repr__(self):
        return f"VectorClock({dict(sorted(self.clock.items()))})"


class HybridLogicalClock:
    """Hybrid Logical Clock (HLC) - combines physical and logical time"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.physical_time = 0
        self.logical_time = 0

    def update(self, message_time: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Update HLC on send or receive"""
        current_physical = int(time.time() * 1000000)  # Microseconds

        if message_time is None:
            # Send event
            if current_physical > self.physical_time:
                self.physical_time = current_physical
                self.logical_time = 0
            else:
                self.logical_time += 1
        else:
            # Receive event
            msg_physical, msg_logical = message_time

            if current_physical > max(self.physical_time, msg_physical):
                self.physical_time = current_physical
                self.logical_time = 0
            elif max(self.physical_time, msg_physical) == self.physical_time:
                self.physical_time = max(self.physical_time, msg_physical)
                self.logical_time = self.logical_time + 1
            else:
                self.physical_time = max(self.physical_time, msg_physical)
                self.logical_time = msg_logical + 1

        return (self.physical_time, self.logical_time)

    def compare(self, other: Tuple[int, int]) -> int:
        """Compare timestamps: -1 if self < other, 0 if equal, 1 if self > other"""
        if self.physical_time < other[0]:
            return -1
        elif self.physical_time > other[0]:
            return 1
        elif self.logical_time < other[1]:
            return -1
        elif self.logical_time > other[1]:
            return 1
        else:
            return 0


# Raft Consensus Algorithm


class RaftState(Enum):
    FOLLOWER = 1
    CANDIDATE = 2
    LEADER = 3


@dataclass
class LogEntry:
    term: int
    command: Any
    index: int


class RaftNode:
    """Raft consensus algorithm node"""

    def __init__(self, node_id: int, peers: List[int]):
        self.node_id = node_id
        self.peers = peers

        # Persistent state
        self.current_term = 0
        self.voted_for = None
        self.log: List[LogEntry] = []

        # Volatile state
        self.state = RaftState.FOLLOWER
        self.commit_index = 0
        self.last_applied = 0

        # Leader state
        self.next_index = {peer: 0 for peer in peers}
        self.match_index = {peer: 0 for peer in peers}

        # Election
        self.election_timeout = self.random_timeout()
        self.last_heartbeat = time.time()

        # Communication
        self.message_queue = Queue()
        self.running = True

    def random_timeout(self) -> float:
        """Random election timeout between 150-300ms"""
        return 0.15 + 0.15 * random.random()

    def request_vote(
        self, term: int, candidate_id: int, last_log_index: int, last_log_term: int
    ) -> Tuple[int, bool]:
        """Handle RequestVote RPC"""
        # Update term if necessary
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = RaftState.FOLLOWER

        # Grant vote if conditions met
        vote_granted = False
        if term == self.current_term and (
            self.voted_for is None or self.voted_for == candidate_id
        ):

            # Check if candidate's log is at least as up-to-date
            my_last_term = self.log[-1].term if self.log else 0
            my_last_index = len(self.log) - 1

            if last_log_term > my_last_term or (
                last_log_term == my_last_term and last_log_index >= my_last_index
            ):
                self.voted_for = candidate_id
                vote_granted = True
                self.last_heartbeat = time.time()

        return self.current_term, vote_granted

    def append_entries(
        self,
        term: int,
        leader_id: int,
        prev_log_index: int,
        prev_log_term: int,
        entries: List[LogEntry],
        leader_commit: int,
    ) -> Tuple[int, bool]:
        """Handle AppendEntries RPC"""
        # Update term if necessary
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = RaftState.FOLLOWER

        # Reset election timeout
        self.last_heartbeat = time.time()

        # Reply false if term < currentTerm
        if term < self.current_term:
            return self.current_term, False

        # Reply false if log doesn't contain entry at prevLogIndex
        if prev_log_index >= len(self.log):
            return self.current_term, False

        if prev_log_index >= 0 and self.log[prev_log_index].term != prev_log_term:
            # Delete conflicting entries
            self.log = self.log[:prev_log_index]
            return self.current_term, False

        # Append new entries
        self.log = self.log[: prev_log_index + 1] + entries

        # Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)

        return self.current_term, True

    def start_election(self):
        """Become candidate and start election"""
        self.state = RaftState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()

        # Request votes from all peers
        votes = 1  # Vote for self

        last_log_term = self.log[-1].term if self.log else 0
        last_log_index = len(self.log) - 1

        for peer in self.peers:
            # In real implementation, this would be async RPC
            term, vote_granted = self.send_request_vote(
                peer, self.current_term, self.node_id, last_log_index, last_log_term
            )

            if vote_granted:
                votes += 1

            if term > self.current_term:
                self.current_term = term
                self.state = RaftState.FOLLOWER
                self.voted_for = None
                return

        # Become leader if received majority
        if votes > (len(self.peers) + 1) // 2:
            self.state = RaftState.LEADER
            self.init_leader_state()

    def init_leader_state(self):
        """Initialize leader state after election"""
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = 0

        # Send initial heartbeats
        self.send_heartbeats()

    def send_heartbeats(self):
        """Send heartbeat to all followers"""
        for peer in self.peers:
            prev_log_index = self.next_index[peer] - 1
            prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0

            # Send empty AppendEntries as heartbeat
            self.send_append_entries(
                peer,
                self.current_term,
                self.node_id,
                prev_log_index,
                prev_log_term,
                [],
                self.commit_index,
            )

    def send_request_vote(self, peer: int, *args) -> Tuple[int, bool]:
        """Send RequestVote RPC (simulated)"""
        # In real implementation, this would be network RPC
        # For simulation, return mock response
        return self.current_term, random.random() > 0.3

    def send_append_entries(self, peer: int, *args) -> Tuple[int, bool]:
        """Send AppendEntries RPC (simulated)"""
        # In real implementation, this would be network RPC
        return self.current_term, True

    def run(self):
        """Main Raft node loop"""
        while self.running:
            if self.state == RaftState.FOLLOWER:
                # Check election timeout
                if time.time() - self.last_heartbeat > self.election_timeout:
                    self.start_election()

            elif self.state == RaftState.CANDIDATE:
                # Election timeout - start new election
                if time.time() - self.last_heartbeat > self.election_timeout:
                    self.start_election()

            elif self.state == RaftState.LEADER:
                # Send periodic heartbeats
                self.send_heartbeats()
                time.sleep(0.05)  # 50ms heartbeat interval

            # Process any pending messages
            try:
                message = self.message_queue.get_nowait()
                self.process_message(message)
            except Empty:
                pass

            time.sleep(0.01)  # Small sleep to prevent busy waiting

    def process_message(self, message: Dict[str, Any]):
        """Process incoming message"""
        msg_type = message.get("type")

        if msg_type == "RequestVote":
            response = self.request_vote(**message["args"])
            # Send response back
        elif msg_type == "AppendEntries":
            response = self.append_entries(**message["args"])
            # Send response back


# Two-Phase Commit (2PC)


class TwoPhaseCommitCoordinator:
    """2PC coordinator for distributed transactions"""

    def __init__(self, participants: List[str]):
        self.participants = participants
        self.transaction_log = []
        self.pending_transactions = {}

    def execute_transaction(self, txn_id: str, operations: List[Dict]) -> bool:
        """Execute distributed transaction using 2PC"""
        # Phase 1: Prepare
        self.log_decision(txn_id, "BEGIN")

        prepare_votes = []
        for participant in self.participants:
            vote = self.send_prepare(participant, txn_id, operations)
            prepare_votes.append(vote)

        # Check if all voted to commit
        if all(prepare_votes):
            # Phase 2: Commit
            self.log_decision(txn_id, "COMMIT")

            for participant in self.participants:
                self.send_commit(participant, txn_id)

            return True
        else:
            # Phase 2: Abort
            self.log_decision(txn_id, "ABORT")

            for participant in self.participants:
                self.send_abort(participant, txn_id)

            return False

    def log_decision(self, txn_id: str, decision: str):
        """Log decision for recovery"""
        self.transaction_log.append(
            {"txn_id": txn_id, "decision": decision, "timestamp": time.time()}
        )

    def send_prepare(
        self, participant: str, txn_id: str, operations: List[Dict]
    ) -> bool:
        """Send prepare request to participant (simulated)"""
        # In real implementation, this would be RPC
        # Simulate 90% success rate
        return random.random() > 0.1

    def send_commit(self, participant: str, txn_id: str):
        """Send commit decision to participant"""
        pass

    def send_abort(self, participant: str, txn_id: str):
        """Send abort decision to participant"""
        pass

    def recover(self):
        """Recover from coordinator failure"""
        # Check transaction log for incomplete transactions
        for entry in self.transaction_log:
            if entry["decision"] == "BEGIN":
                # Transaction started but no decision logged
                # Must have failed before decision - abort
                self.log_decision(entry["txn_id"], "ABORT")
                for participant in self.participants:
                    self.send_abort(participant, entry["txn_id"])


class TwoPhaseCommitParticipant:
    """2PC participant"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.prepared_txns = {}  # txn_id -> operations
        self.undo_log = []
        self.redo_log = []
        self.lock_manager = {}  # resource -> txn_id

    def handle_prepare(self, txn_id: str, operations: List[Dict]) -> bool:
        """Handle prepare request"""
        try:
            # Acquire locks
            locks_acquired = []
            for op in operations:
                resource = op.get("resource")
                if resource in self.lock_manager:
                    # Resource already locked
                    # Release acquired locks and abort
                    for lock in locks_acquired:
                        del self.lock_manager[lock]
                    return False

                self.lock_manager[resource] = txn_id
                locks_acquired.append(resource)

            # Validate operations
            for op in operations:
                if not self.can_execute(op):
                    # Release locks
                    for resource in locks_acquired:
                        del self.lock_manager[resource]
                    return False

            # Create undo log
            for op in operations:
                self.undo_log.append(
                    {"txn_id": txn_id, "operation": self.create_undo(op)}
                )

            # Execute operations tentatively
            for op in operations:
                self.execute_tentative(op)

            # Log prepare decision
            self.prepared_txns[txn_id] = operations

            return True  # Vote commit
        except Exception:
            return False  # Vote abort

    def handle_commit(self, txn_id: str):
        """Handle commit decision"""
        if txn_id in self.prepared_txns:
            # Make changes permanent
            operations = self.prepared_txns[txn_id]

            for op in operations:
                self.redo_log.append({"txn_id": txn_id, "operation": op})

            # Release locks
            resources_to_release = []
            for resource, tid in self.lock_manager.items():
                if tid == txn_id:
                    resources_to_release.append(resource)

            for resource in resources_to_release:
                del self.lock_manager[resource]

            del self.prepared_txns[txn_id]

    def handle_abort(self, txn_id: str):
        """Handle abort decision"""
        if txn_id in self.prepared_txns:
            # Rollback changes
            undo_ops = [
                entry["operation"]
                for entry in self.undo_log
                if entry["txn_id"] == txn_id
            ]

            for op in reversed(undo_ops):
                self.execute_operation(op)

            # Release locks
            resources_to_release = []
            for resource, tid in self.lock_manager.items():
                if tid == txn_id:
                    resources_to_release.append(resource)

            for resource in resources_to_release:
                del self.lock_manager[resource]

            del self.prepared_txns[txn_id]

    def can_execute(self, operation: Dict) -> bool:
        """Check if operation can be executed"""
        # Simplified validation
        return True

    def create_undo(self, operation: Dict) -> Dict:
        """Create undo operation"""
        if operation["type"] == "write":
            return {
                "type": "write",
                "resource": operation["resource"],
                "value": self.read_current_value(operation["resource"]),
            }
        return {}

    def execute_tentative(self, operation: Dict):
        """Execute operation tentatively"""
        # Implementation depends on storage engine
        pass

    def execute_operation(self, operation: Dict):
        """Execute operation"""
        pass

    def read_current_value(self, resource: str) -> Any:
        """Read current value of resource"""
        return None


# Three-Phase Commit (3PC) - Non-blocking variant


class ThreePhaseCommitCoordinator(TwoPhaseCommitCoordinator):
    """3PC coordinator - adds prepare-to-commit phase"""

    def execute_transaction(self, txn_id: str, operations: List[Dict]) -> bool:
        """Execute distributed transaction using 3PC"""
        # Phase 1: CanCommit
        self.log_decision(txn_id, "BEGIN")

        can_commit_votes = []
        for participant in self.participants:
            vote = self.send_can_commit(participant, txn_id, operations)
            can_commit_votes.append(vote)

        if not all(can_commit_votes):
            # Abort
            self.log_decision(txn_id, "ABORT")
            for participant in self.participants:
                self.send_abort(participant, txn_id)
            return False

        # Phase 2: PreCommit
        self.log_decision(txn_id, "PRECOMMIT")

        pre_commit_acks = []
        for participant in self.participants:
            ack = self.send_pre_commit(participant, txn_id)
            pre_commit_acks.append(ack)

        if not all(pre_commit_acks):
            # Abort
            self.log_decision(txn_id, "ABORT")
            for participant in self.participants:
                self.send_abort(participant, txn_id)
            return False

        # Phase 3: DoCommit
        self.log_decision(txn_id, "COMMIT")

        for participant in self.participants:
            self.send_do_commit(participant, txn_id)

        return True

    def send_can_commit(
        self, participant: str, txn_id: str, operations: List[Dict]
    ) -> bool:
        """Send CanCommit query"""
        return random.random() > 0.1

    def send_pre_commit(self, participant: str, txn_id: str) -> bool:
        """Send PreCommit message"""
        return random.random() > 0.05

    def send_do_commit(self, participant: str, txn_id: str):
        """Send DoCommit message"""
        pass


# Saga Pattern for Long-Running Transactions


@dataclass
class SagaStep:
    """Single step in a saga"""

    name: str
    transaction: callable
    compensation: callable


class Saga:
    """Saga pattern for distributed transactions"""

    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.completed_steps: List[str] = []
        self.saga_log = []

    def add_step(self, step: SagaStep):
        """Add step to saga"""
        self.steps.append(step)

    def execute(self) -> bool:
        """Execute saga with compensations on failure"""
        for step in self.steps:
            try:
                # Log step start
                self.log_event("START", step.name)

                # Execute transaction
                result = step.transaction()

                # Log completion
                self.log_event("COMPLETE", step.name, result)
                self.completed_steps.append(step.name)

            except Exception as e:
                # Log failure
                self.log_event("FAILED", step.name, str(e))

                # Compensate completed steps in reverse order
                self.compensate()
                return False

        return True

    def compensate(self):
        """Run compensations for completed steps"""
        for step_name in reversed(self.completed_steps):
            step = next(s for s in self.steps if s.name == step_name)

            try:
                self.log_event("COMPENSATE_START", step_name)
                step.compensation()
                self.log_event("COMPENSATE_COMPLETE", step_name)
            except Exception as e:
                self.log_event("COMPENSATE_FAILED", step_name, str(e))
                # In practice, might need manual intervention

    def log_event(self, event_type: str, step_name: str, data: Any = None):
        """Log saga event"""
        self.saga_log.append(
            {
                "timestamp": time.time(),
                "saga_id": self.saga_id,
                "event_type": event_type,
                "step_name": step_name,
                "data": data,
            }
        )


# Example usage and demonstrations


def demonstrate_vector_clocks():
    """Show vector clock usage"""
    print("Vector Clock Demonstration:")

    # Three nodes
    vc1 = VectorClock("node_0", 3)
    vc2 = VectorClock("node_1", 3)
    vc3 = VectorClock("node_2", 3)

    # Node 1 performs operation
    vc1.increment()
    print(f"Node 1 after operation: {vc1}")

    # Node 1 sends message to Node 2
    vc2.update(vc1.clock)
    print(f"Node 2 after receiving from Node 1: {vc2}")

    # Node 3 performs independent operation
    vc3.increment()
    print(f"Node 3 independent operation: {vc3}")

    # Check causality
    print(f"Node 1 happens before Node 2: {vc1.happens_before(vc2)}")
    print(f"Node 1 concurrent with Node 3: {vc1.concurrent_with(vc3)}")


def demonstrate_raft():
    """Show Raft consensus"""
    print("\nRaft Consensus Demonstration:")

    # Create 5-node cluster
    nodes = []
    for i in range(5):
        peers = [j for j in range(5) if j != i]
        node = RaftNode(i, peers)
        nodes.append(node)

    # Simulate leader election
    print("Starting leader election...")
    # Node 0 times out and starts election
    nodes[0].start_election()

    if nodes[0].state == RaftState.LEADER:
        print(f"Node 0 elected as leader in term {nodes[0].current_term}")

    # Leader accepts client request
    if nodes[0].state == RaftState.LEADER:
        log_entry = LogEntry(
            term=nodes[0].current_term, command="SET x=5", index=len(nodes[0].log)
        )
        nodes[0].log.append(log_entry)
        print("Leader accepted command: SET x=5")


def demonstrate_2pc():
    """Show 2PC protocol"""
    print("\n2PC Protocol Demonstration:")

    # Create coordinator and participants
    coordinator = TwoPhaseCommitCoordinator(["node1", "node2", "node3"])

    # Execute transaction
    operations = [
        {"type": "write", "resource": "account_A", "value": -100},
        {"type": "write", "resource": "account_B", "value": 100},
    ]

    result = coordinator.execute_transaction("txn_001", operations)
    print(f"Transaction txn_001 result: {'COMMITTED' if result else 'ABORTED'}")

    # Show transaction log
    print("Transaction log:")
    for entry in coordinator.transaction_log:
        print(f"  {entry['txn_id']}: {entry['decision']}")


def demonstrate_saga():
    """Show Saga pattern"""
    print("\nSaga Pattern Demonstration:")

    # Create saga for order processing
    saga = Saga("order_123")

    # Define steps
    saga.add_step(
        SagaStep(
            name="reserve_inventory",
            transaction=lambda: print("Reserved inventory"),
            compensation=lambda: print("Released inventory reservation"),
        )
    )

    saga.add_step(
        SagaStep(
            name="charge_payment",
            transaction=lambda: print("Charged payment"),
            compensation=lambda: print("Refunded payment"),
        )
    )

    saga.add_step(
        SagaStep(
            name="ship_order",
            transaction=lambda: print("Shipped order"),
            compensation=lambda: print("Cancelled shipment"),
        )
    )

    # Execute saga
    success = saga.execute()
    print(f"Saga completed: {success}")


if __name__ == "__main__":
    demonstrate_vector_clocks()
    demonstrate_raft()
    demonstrate_2pc()
    demonstrate_saga()
