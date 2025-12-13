"""
Query Processing and Optimization

Implementation of query processing pipeline, cost-based optimization,
and join order optimization using dynamic programming.
"""

import ast
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import graphviz
import numpy as np


@dataclass
class QueryPlan:
    """Abstract syntax tree for query plans"""

    operation: str
    children: List["QueryPlan"]
    cost: float = 0.0
    cardinality: int = 0

    def __repr__(self):
        return f"QueryPlan({self.operation}, cost={self.cost:.2f}, rows={self.cardinality})"


@dataclass
class TableStats:
    """Statistics for a database table"""

    name: str
    num_rows: int
    num_pages: int
    avg_row_size: int
    columns: Dict[str, "ColumnStats"]


@dataclass
class ColumnStats:
    """Statistics for a table column"""

    name: str
    distinct_values: int
    min_value: Any
    max_value: Any
    null_count: int
    histogram: Optional[List[Tuple[Any, int]]] = None  # (value, frequency) pairs


class QueryProcessor:
    """Query processing engine implementation"""

    def __init__(self, statistics: Dict[str, TableStats]):
        self.statistics = statistics
        self.cost_model = CostModel()
        self.optimizer = QueryOptimizer(self.cost_model, statistics)

    def process_query(self, sql: str) -> Any:
        """Main query processing pipeline"""
        # 1. Parsing
        ast_tree = self.parse_sql(sql)

        # 2. Semantic analysis
        validated_ast = self.semantic_analysis(ast_tree)

        # 3. Query rewriting
        rewritten_ast = self.query_rewrite(validated_ast)

        # 4. Optimization
        logical_plan = self.generate_logical_plan(rewritten_ast)
        optimized_plan = self.optimize_plan(logical_plan)

        # 5. Physical planning
        physical_plan = self.generate_physical_plan(optimized_plan)

        # 6. Execution
        return self.execute_plan(physical_plan)

    def parse_sql(self, sql: str) -> ast.AST:
        """SQL parsing using recursive descent parser (simplified)"""
        # In practice, would use a proper SQL parser like sqlparse
        # This is a placeholder
        return ast.parse(f"# SQL: {sql}")

    def semantic_analysis(self, ast_tree: ast.AST) -> ast.AST:
        """Validate and annotate AST with semantic information"""
        # Check table existence, column references, type checking
        return ast_tree

    def query_rewrite(self, ast_tree: ast.AST) -> ast.AST:
        """Apply query rewriting rules"""
        # View expansion, subquery flattening, predicate simplification
        return ast_tree

    def generate_logical_plan(self, ast_tree: ast.AST) -> QueryPlan:
        """Convert AST to logical query plan"""
        # This would convert SQL AST to relational algebra tree
        # Placeholder implementation
        return QueryPlan("select", [])

    def optimize_plan(self, plan: QueryPlan) -> QueryPlan:
        """Apply cost-based optimization"""
        return self.optimizer.optimize(plan)

    def generate_physical_plan(self, logical_plan: QueryPlan) -> QueryPlan:
        """Convert logical plan to physical plan with specific algorithms"""
        # Choose specific implementations (hash join vs merge join, etc.)
        return self._add_physical_operators(logical_plan)

    def execute_plan(self, plan: QueryPlan) -> Any:
        """Execute the physical query plan"""
        # Actual execution engine would process operators
        return f"Executing plan: {plan}"

    def _add_physical_operators(self, plan: QueryPlan) -> QueryPlan:
        """Add physical operator choices to plan"""
        if plan.operation == "join":
            # Choose join algorithm based on cost
            if plan.cost < 1000:
                plan.operation = "nested_loop_join"
            elif plan.children[0].cardinality < plan.children[1].cardinality:
                plan.operation = "hash_join_build_left"
            else:
                plan.operation = "hash_join_build_right"

        # Recursively process children
        plan.children = [self._add_physical_operators(child) for child in plan.children]
        return plan


class CostModel:
    """Database cost model for query optimization"""

    def __init__(self):
        self.cpu_tuple_cost = 0.01
        self.cpu_operator_cost = 0.0025
        self.seq_page_cost = 1.0
        self.random_page_cost = 4.0
        self.network_cost = 10.0

    def estimate_scan_cost(
        self, table_stats: TableStats, selectivity: float = 1.0
    ) -> float:
        """Estimate cost of table scan"""
        pages = table_stats.num_pages
        tuples = table_stats.num_rows

        # Sequential scan cost
        seq_cost = (
            self.seq_page_cost * pages + self.cpu_tuple_cost * tuples * selectivity
        )

        return seq_cost

    def estimate_index_scan_cost(
        self, table_stats: TableStats, index_stats: Dict[str, Any], selectivity: float
    ) -> float:
        """Estimate cost of index scan"""
        # B-tree height
        height = index_stats.get("height", 3)

        # Index pages to read
        index_pages = height + selectivity * index_stats.get("leaf_pages", 100)

        # Data pages to read (assumes clustered index)
        data_pages = selectivity * table_stats.num_pages

        # Total cost
        cost = (
            self.random_page_cost * index_pages
            + self.random_page_cost * data_pages
            + self.cpu_tuple_cost * table_stats.num_rows * selectivity
        )

        return cost

    def estimate_join_cost(self, method: str, left_size: int, right_size: int) -> float:
        """Estimate join operation cost"""
        if method == "nested_loop":
            # O(n*m) comparisons
            return left_size * right_size * self.cpu_operator_cost

        elif method == "hash":
            # Build hash table + probe
            build_cost = right_size * self.cpu_operator_cost * 2
            probe_cost = left_size * self.cpu_operator_cost * 1.5
            return build_cost + probe_cost

        elif method == "merge":
            # Sort both + merge
            sort_cost = (
                left_size * np.log2(left_size) + right_size * np.log2(right_size)
            ) * self.cpu_operator_cost
            merge_cost = (left_size + right_size) * self.cpu_operator_cost
            return sort_cost + merge_cost

        else:
            raise ValueError(f"Unknown join method: {method}")

    def estimate_sort_cost(self, num_tuples: int, tuple_size: int) -> float:
        """Estimate cost of sorting"""
        # External merge sort cost
        pages = (num_tuples * tuple_size) / 8192  # 8KB pages

        # Number of merge passes
        num_passes = np.ceil(np.log2(pages))

        # Cost = 2 * pages * passes (read + write each pass)
        io_cost = 2 * pages * num_passes * self.seq_page_cost
        cpu_cost = num_tuples * np.log2(num_tuples) * self.cpu_operator_cost

        return io_cost + cpu_cost

    def estimate_aggregate_cost(self, num_tuples: int, group_by_columns: int) -> float:
        """Estimate aggregation cost"""
        if group_by_columns == 0:
            # Simple aggregation
            return num_tuples * self.cpu_operator_cost
        else:
            # Hash-based grouping
            return num_tuples * self.cpu_operator_cost * 2


class QueryOptimizer:
    """Cost-based query optimizer"""

    def __init__(self, cost_model: CostModel, statistics: Dict[str, TableStats]):
        self.cost_model = cost_model
        self.statistics = statistics
        self.transformation_rules = [
            self.push_selection_down,
            self.push_projection_down,
            self.combine_selections,
            self.reorder_joins,
        ]

    def optimize(self, plan: QueryPlan) -> QueryPlan:
        """Main optimization entry point"""
        # Apply transformation rules
        plan = self.apply_transformations(plan)

        # Optimize join order if applicable
        if self.has_joins(plan):
            plan = self.optimize_join_order(plan)

        # Choose physical operators
        plan = self.choose_access_methods(plan)

        # Add interesting orders
        plan = self.consider_interesting_orders(plan)

        return plan

    def apply_transformations(self, plan: QueryPlan) -> QueryPlan:
        """Apply logical transformation rules"""
        changed = True
        while changed:
            changed = False
            for rule in self.transformation_rules:
                new_plan = rule(plan)
                if new_plan != plan:
                    plan = new_plan
                    changed = True
        return plan

    def push_selection_down(self, plan: QueryPlan) -> QueryPlan:
        """Push selections as close to base relations as possible"""
        # Implementation would traverse tree and push selections down
        return plan

    def push_projection_down(self, plan: QueryPlan) -> QueryPlan:
        """Push projections down while maintaining required attributes"""
        return plan

    def combine_selections(self, plan: QueryPlan) -> QueryPlan:
        """Combine multiple selections into conjunctive predicate"""
        return plan

    def reorder_joins(self, plan: QueryPlan) -> QueryPlan:
        """Use dynamic programming to find optimal join order"""
        joins = self._extract_joins(plan)
        if len(joins) <= 2:
            return plan

        # Use separate JoinOrderOptimizer
        optimizer = JoinOrderOptimizer(self.cost_model, self.statistics)
        optimal_order = optimizer.find_optimal_order(joins)

        return self._rebuild_plan_with_order(plan, optimal_order)

    def has_joins(self, plan: QueryPlan) -> bool:
        """Check if plan contains joins"""
        if plan.operation == "join":
            return True
        return any(self.has_joins(child) for child in plan.children)

    def choose_access_methods(self, plan: QueryPlan) -> QueryPlan:
        """Choose between sequential scan, index scan, etc."""
        if plan.operation == "scan":
            table_name = plan.params.get("table")
            if table_name in self.statistics:
                stats = self.statistics[table_name]

                # Check if index is beneficial
                if "predicate" in plan.params:
                    selectivity = self._estimate_selectivity(plan.params["predicate"])

                    seq_cost = self.cost_model.estimate_scan_cost(stats, selectivity)

                    # Check available indexes
                    if "indexes" in stats:
                        for index_name, index_stats in stats["indexes"].items():
                            index_cost = self.cost_model.estimate_index_scan_cost(
                                stats, index_stats, selectivity
                            )

                            if index_cost < seq_cost:
                                plan.operation = "index_scan"
                                plan.params["index"] = index_name
                                plan.cost = index_cost
                                break

        # Recursively process children
        plan.children = [self.choose_access_methods(child) for child in plan.children]
        return plan

    def consider_interesting_orders(self, plan: QueryPlan) -> QueryPlan:
        """Consider sort orders that might benefit parent operators"""
        # If parent needs sorted input, might be worth sorting early
        return plan

    def _extract_joins(self, plan: QueryPlan) -> List[Tuple[str, str]]:
        """Extract all joins from plan"""
        joins = []
        if plan.operation == "join":
            # Extract table names from children
            left_tables = self._get_tables(plan.children[0])
            right_tables = self._get_tables(plan.children[1])
            joins.append((left_tables, right_tables))

        for child in plan.children:
            joins.extend(self._extract_joins(child))

        return joins

    def _get_tables(self, plan: QueryPlan) -> Set[str]:
        """Get all base tables in a plan subtree"""
        if plan.operation == "scan":
            return {plan.params.get("table")}

        tables = set()
        for child in plan.children:
            tables.update(self._get_tables(child))
        return tables

    def _estimate_selectivity(self, predicate: Dict[str, Any]) -> float:
        """Estimate predicate selectivity"""
        # Simplified - real optimizer would use histograms
        if predicate.get("type") == "equality":
            return 0.1
        elif predicate.get("type") == "range":
            return 0.3
        else:
            return 0.5

    def _rebuild_plan_with_order(
        self, original: QueryPlan, join_order: List[str]
    ) -> QueryPlan:
        """Rebuild plan with new join order"""
        # This would reconstruct the plan tree with optimal join order
        return original


class JoinOrderOptimizer:
    """Dynamic programming approach to join ordering"""

    def __init__(self, cost_model: CostModel, statistics: Dict[str, TableStats]):
        self.cost_model = cost_model
        self.statistics = statistics
        self.memo = {}

    def find_optimal_order(self, relations: List[str]) -> QueryPlan:
        """Find optimal join order using dynamic programming"""
        n = len(relations)

        # Initialize single relations
        for i, rel in enumerate(relations):
            self.memo[frozenset([i])] = {
                "cost": 0,
                "size": self.get_relation_size(rel),
                "plan": QueryPlan("scan", [], params={"table": rel}),
            }

        # Build up larger join sets
        for size in range(2, n + 1):
            for subset in combinations(range(n), size):
                subset_set = frozenset(subset)
                best_cost = float("inf")
                best_plan = None

                # Try all ways to split this subset
                for split_size in range(1, size):
                    for left_indices in combinations(subset, split_size):
                        left_set = frozenset(left_indices)
                        right_set = subset_set - left_set

                        if left_set not in self.memo or right_set not in self.memo:
                            continue

                        left_info = self.memo[left_set]
                        right_info = self.memo[right_set]

                        # Try different join methods
                        for method in ["nested_loop", "hash", "merge"]:
                            cost = (
                                left_info["cost"]
                                + right_info["cost"]
                                + self.cost_model.estimate_join_cost(
                                    method, left_info["size"], right_info["size"]
                                )
                            )

                            if cost < best_cost:
                                best_cost = cost
                                best_plan = QueryPlan(
                                    f"{method}_join",
                                    [left_info["plan"], right_info["plan"]],
                                    cost=cost,
                                    cardinality=self.estimate_join_size(
                                        left_info["size"], right_info["size"]
                                    ),
                                )

                self.memo[subset_set] = {
                    "cost": best_cost,
                    "size": best_plan.cardinality if best_plan else 0,
                    "plan": best_plan,
                }

        return self.memo[frozenset(range(n))]["plan"]

    def get_relation_size(self, relation: str) -> int:
        """Get size of base relation"""
        if relation in self.statistics:
            return self.statistics[relation].num_rows
        return 1000  # Default estimate

    def estimate_join_size(self, left_size: int, right_size: int) -> int:
        """Estimate join output size"""
        # Simplified - real optimizer would use join selectivity
        return int(left_size * right_size * 0.1)


class QueryPlanVisualizer:
    """Visualize query execution plans"""

    def __init__(self):
        self.node_counter = 0

    def visualize_plan(self, plan: QueryPlan, filename: str = "query_plan"):
        """Generate visual representation of query plan"""
        dot = graphviz.Digraph(comment="Query Plan")
        dot.attr(rankdir="BT")

        self._add_node(plan, dot)

        # Render to file
        dot.render(filename, format="png", cleanup=True)
        return dot

    def _add_node(self, node: QueryPlan, dot: graphviz.Digraph) -> str:
        """Add node to graph recursively"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        # Create label
        label_lines = [node.operation]
        if node.cost > 0:
            label_lines.append(f"Cost: {node.cost:.2f}")
        if node.cardinality > 0:
            label_lines.append(f"Rows: {node.cardinality}")

        # Add any parameters
        if hasattr(node, "params"):
            for key, value in node.params.items():
                label_lines.append(f"{key}: {value}")

        label = "\\n".join(label_lines)

        # Style based on operation type
        if "scan" in node.operation:
            dot.node(node_id, label, shape="box", style="filled", fillcolor="lightblue")
        elif "join" in node.operation:
            dot.node(
                node_id, label, shape="box", style="filled", fillcolor="lightgreen"
            )
        elif "sort" in node.operation:
            dot.node(
                node_id, label, shape="box", style="filled", fillcolor="lightyellow"
            )
        else:
            dot.node(node_id, label, shape="box")

        # Add children
        for child in node.children:
            child_id = self._add_node(child, dot)
            dot.edge(child_id, node_id)

        return node_id

    def to_text(self, plan: QueryPlan, indent: int = 0) -> str:
        """Convert plan to text representation"""
        lines = []
        prefix = "  " * indent

        # Current node
        line = f"{prefix}{plan.operation}"
        if plan.cost > 0:
            line += f" (cost={plan.cost:.2f}"
            if plan.cardinality > 0:
                line += f", rows={plan.cardinality}"
            line += ")"

        lines.append(line)

        # Children
        for child in plan.children:
            lines.append(self.to_text(child, indent + 1))

        return "\n".join(lines)


# Example usage
def demonstrate_query_optimization():
    """Example of query optimization process"""

    # Create sample statistics
    stats = {
        "users": TableStats(
            name="users",
            num_rows=10000,
            num_pages=100,
            avg_row_size=200,
            columns={
                "id": ColumnStats(
                    "id",
                    distinct_values=10000,
                    min_value=1,
                    max_value=10000,
                    null_count=0,
                ),
                "age": ColumnStats(
                    "age", distinct_values=80, min_value=18, max_value=100, null_count=0
                ),
            },
        ),
        "orders": TableStats(
            name="orders",
            num_rows=50000,
            num_pages=500,
            avg_row_size=100,
            columns={
                "user_id": ColumnStats(
                    "user_id",
                    distinct_values=8000,
                    min_value=1,
                    max_value=10000,
                    null_count=0,
                )
            },
        ),
        "products": TableStats(
            name="products", num_rows=1000, num_pages=10, avg_row_size=500, columns={}
        ),
    }

    # Create query processor
    processor = QueryProcessor(stats)

    # Example query plan before optimization
    scan_users = QueryPlan("scan", [], params={"table": "users"})
    scan_orders = QueryPlan("scan", [], params={"table": "orders"})
    scan_products = QueryPlan("scan", [], params={"table": "products"})

    join1 = QueryPlan("join", [scan_users, scan_orders])
    join2 = QueryPlan("join", [join1, scan_products])
    root = QueryPlan(
        "project", [join2], params={"columns": ["user_id", "product_name"]}
    )

    # Optimize
    optimizer = QueryOptimizer(CostModel(), stats)
    optimized = optimizer.optimize(root)

    # Visualize
    visualizer = QueryPlanVisualizer()
    print("Original plan:")
    print(visualizer.to_text(root))
    print("\nOptimized plan:")
    print(visualizer.to_text(optimized))


if __name__ == "__main__":
    demonstrate_query_optimization()
