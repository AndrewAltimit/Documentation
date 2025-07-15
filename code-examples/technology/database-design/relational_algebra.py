"""
Relational Algebra Implementation

Fundamental operations of relational algebra and query optimization
based on algebraic laws.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Callable, Dict, Any


class RelationalAlgebra:
    """Implementation of relational algebra operations"""
    
    @staticmethod
    def selection(relation: pd.DataFrame, predicate: Callable) -> pd.DataFrame:
        """σ(R) - Select tuples satisfying predicate"""
        return relation[relation.apply(predicate, axis=1)]
    
    @staticmethod
    def projection(relation: pd.DataFrame, attributes: List[str]) -> pd.DataFrame:
        """π(R) - Project specific attributes"""
        return relation[attributes].drop_duplicates()
    
    @staticmethod
    def union(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        """R ∪ S - Union of relations"""
        return pd.concat([R, S]).drop_duplicates()
    
    @staticmethod
    def difference(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        """R - S - Tuples in R but not in S"""
        return R.merge(S, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    
    @staticmethod
    def cartesian_product(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        """R × S - Cartesian product"""
        R['_key'] = 1
        S['_key'] = 1
        result = R.merge(S, on='_key').drop('_key', axis=1)
        return result
    
    @staticmethod
    def natural_join(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        """R ⋈ S - Natural join on common attributes"""
        common_attrs = list(set(R.columns) & set(S.columns))
        if not common_attrs:
            return RelationalAlgebra.cartesian_product(R, S)
        return R.merge(S, on=common_attrs)
    
    @staticmethod
    def theta_join(R: pd.DataFrame, S: pd.DataFrame, theta: Callable) -> pd.DataFrame:
        """R ⋈θ S - Join with arbitrary condition"""
        cp = RelationalAlgebra.cartesian_product(R, S)
        return cp[cp.apply(theta, axis=1)]
    
    @staticmethod
    def division(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        """R ÷ S - Division operation"""
        # R(A,B) ÷ S(B) = {a | ∀b ∈ S, (a,b) ∈ R}
        # Assumes R has columns that are superset of S columns
        
        s_cols = list(S.columns)
        r_only_cols = [col for col in R.columns if col not in s_cols]
        
        # Get all combinations of R-only attributes
        r_only_values = R[r_only_cols].drop_duplicates()
        
        result = []
        for _, row in r_only_values.iterrows():
            # Check if this value appears with all S tuples
            subset = R
            for col in r_only_cols:
                subset = subset[subset[col] == row[col]]
            
            if S[s_cols].merge(subset[s_cols]).equals(S[s_cols]):
                result.append(row)
        
        return pd.DataFrame(result)
    
    @staticmethod
    def rename(relation: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """ρ(R) - Rename attributes"""
        return relation.rename(columns=mapping)
    
    @staticmethod
    def aggregate(relation: pd.DataFrame, group_by: List[str], 
                  agg_funcs: Dict[str, Tuple[str, Callable]]) -> pd.DataFrame:
        """γ(R) - Aggregate functions with grouping"""
        if not group_by:
            # No grouping, aggregate entire relation
            result = {}
            for new_name, (col, func) in agg_funcs.items():
                result[new_name] = func(relation[col])
            return pd.DataFrame([result])
        else:
            # Group by specified columns
            grouped = relation.groupby(group_by)
            agg_dict = {col: func for _, (col, func) in agg_funcs.items()}
            result = grouped.agg(agg_dict)
            result.columns = list(agg_funcs.keys())
            return result.reset_index()


# Example: Query optimization using algebraic laws
class QueryOptimizer:
    """Apply algebraic optimization rules"""
    
    def __init__(self):
        self.optimization_rules = [
            self.push_selection_down,
            self.combine_selections,
            self.push_projection_down,
            self.combine_projections,
            self.reorder_joins
        ]
    
    def optimize(self, query_tree: 'QueryTree') -> 'QueryTree':
        """Apply optimization rules iteratively"""
        changed = True
        while changed:
            changed = False
            for rule in self.optimization_rules:
                new_tree = rule(query_tree)
                if new_tree != query_tree:
                    query_tree = new_tree
                    changed = True
        return query_tree
    
    @staticmethod
    def push_selection_down(query_tree: 'QueryTree') -> 'QueryTree':
        """σp(R ⋈ S) ≡ σp(R) ⋈ S if p only references R"""
        # Implementation of selection pushdown
        # This would traverse the query tree and push selections
        # as close to the base relations as possible
        return query_tree
    
    @staticmethod
    def combine_selections(query_tree: 'QueryTree') -> 'QueryTree':
        """σp(σq(R)) ≡ σp∧q(R)"""
        # Combine multiple selections into single selection
        # with conjunctive predicate
        return query_tree
    
    @staticmethod
    def push_projection_down(query_tree: 'QueryTree') -> 'QueryTree':
        """Push projections down but preserve attributes needed by operators above"""
        # Careful to maintain attributes needed by parent operators
        return query_tree
    
    @staticmethod
    def combine_projections(query_tree: 'QueryTree') -> 'QueryTree':
        """πL1(πL2(R)) ≡ πL1(R) if L1 ⊆ L2"""
        # Eliminate redundant projections
        return query_tree
    
    @staticmethod
    def reorder_joins(query_tree: 'QueryTree') -> 'QueryTree':
        """Use dynamic programming for optimal join order"""
        # Implementation would use cost model to find
        # optimal join ordering
        return query_tree


class QueryTree:
    """Abstract syntax tree for relational algebra queries"""
    
    def __init__(self, operation: str, children: List['QueryTree'] = None, 
                 params: Dict[str, Any] = None):
        self.operation = operation
        self.children = children or []
        self.params = params or {}
        self.cost = 0
        self.cardinality = 0
        self.schema = []
    
    def execute(self, database: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute query tree against database"""
        if self.operation == 'relation':
            return database[self.params['name']]
        
        elif self.operation == 'selection':
            child_result = self.children[0].execute(database)
            return RelationalAlgebra.selection(child_result, self.params['predicate'])
        
        elif self.operation == 'projection':
            child_result = self.children[0].execute(database)
            return RelationalAlgebra.projection(child_result, self.params['attributes'])
        
        elif self.operation == 'join':
            left_result = self.children[0].execute(database)
            right_result = self.children[1].execute(database)
            if 'condition' in self.params:
                return RelationalAlgebra.theta_join(left_result, right_result, 
                                                  self.params['condition'])
            else:
                return RelationalAlgebra.natural_join(left_result, right_result)
        
        elif self.operation == 'union':
            left_result = self.children[0].execute(database)
            right_result = self.children[1].execute(database)
            return RelationalAlgebra.union(left_result, right_result)
        
        elif self.operation == 'difference':
            left_result = self.children[0].execute(database)
            right_result = self.children[1].execute(database)
            return RelationalAlgebra.difference(left_result, right_result)
        
        elif self.operation == 'aggregate':
            child_result = self.children[0].execute(database)
            return RelationalAlgebra.aggregate(child_result, 
                                             self.params.get('group_by', []),
                                             self.params['agg_funcs'])
        
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def __eq__(self, other):
        """Check if two query trees are equivalent"""
        if not isinstance(other, QueryTree):
            return False
        return (self.operation == other.operation and 
                self.params == other.params and 
                self.children == other.children)


# Example usage functions
def demonstrate_relational_algebra():
    """Example of using relational algebra operations"""
    
    # Create sample relations
    students = pd.DataFrame({
        'student_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'major': ['CS', 'Math', 'CS', 'Physics']
    })
    
    enrollments = pd.DataFrame({
        'student_id': [1, 1, 2, 3, 3, 4],
        'course_id': [101, 102, 101, 102, 103, 101],
        'grade': ['A', 'B', 'A', 'B', 'A', 'C']
    })
    
    courses = pd.DataFrame({
        'course_id': [101, 102, 103],
        'title': ['Database Systems', 'Algorithms', 'Machine Learning'],
        'credits': [3, 4, 3]
    })
    
    # Selection: Find CS students
    cs_students = RelationalAlgebra.selection(
        students, 
        lambda row: row['major'] == 'CS'
    )
    
    # Projection: Get student names only
    student_names = RelationalAlgebra.projection(students, ['name'])
    
    # Natural join: Students with their enrollments
    student_enrollments = RelationalAlgebra.natural_join(students, enrollments)
    
    # Three-way join: Complete enrollment information
    complete_info = RelationalAlgebra.natural_join(student_enrollments, courses)
    
    # Aggregate: Average grade per student
    grade_map = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
    complete_info['grade_points'] = complete_info['grade'].map(grade_map)
    
    gpa_by_student = RelationalAlgebra.aggregate(
        complete_info,
        group_by=['student_id', 'name'],
        agg_funcs={
            'gpa': ('grade_points', lambda x: x.mean()),
            'total_credits': ('credits', lambda x: x.sum())
        }
    )
    
    return {
        'cs_students': cs_students,
        'student_names': student_names,
        'complete_info': complete_info,
        'gpa_by_student': gpa_by_student
    }


def sql_to_relational_algebra(sql: str) -> QueryTree:
    """Convert SQL query to relational algebra tree (simplified)"""
    # This is a simplified example - real SQL parsing is much more complex
    
    # Example: SELECT name FROM students WHERE major = 'CS'
    # Becomes: π[name](σ[major='CS'](students))
    
    # In practice, would use a proper SQL parser
    pass


def optimize_query_example():
    """Example of query optimization"""
    
    # Original query tree: π[name](σ[major='CS'](students ⋈ enrollments))
    # After optimization: π[name]((σ[major='CS'](students)) ⋈ enrollments)
    
    # Create query tree
    students_rel = QueryTree('relation', params={'name': 'students'})
    enrollments_rel = QueryTree('relation', params={'name': 'enrollments'})
    
    # Original: join first, then select
    join_tree = QueryTree('join', children=[students_rel, enrollments_rel])
    select_tree = QueryTree('selection', 
                          children=[join_tree],
                          params={'predicate': lambda r: r['major'] == 'CS'})
    project_tree = QueryTree('projection',
                           children=[select_tree],
                           params={'attributes': ['name']})
    
    # Optimize
    optimizer = QueryOptimizer()
    optimized_tree = optimizer.optimize(project_tree)
    
    return project_tree, optimized_tree