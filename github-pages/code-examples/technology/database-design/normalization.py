"""
Database Normalization and Functional Dependencies

Implementation of Armstrong's axioms, closure computation,
and BCNF decomposition algorithms.
"""

from typing import Set, List, Tuple, Dict, FrozenSet
from itertools import chain, combinations


def powerset(iterable):
    """Generate powerset of an iterable"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class FunctionalDependency:
    """Formal treatment of functional dependencies"""
    
    def __init__(self, determinant: Set[str], dependent: Set[str]):
        self.determinant = frozenset(determinant)
        self.dependent = frozenset(dependent)
    
    def __repr__(self):
        return f"{set(self.determinant)} → {set(self.dependent)}"
    
    def __eq__(self, other):
        return (self.determinant == other.determinant and 
                self.dependent == other.dependent)
    
    def __hash__(self):
        return hash((self.determinant, self.dependent))
    
    def is_trivial(self) -> bool:
        """Check if FD is trivial (Y ⊆ X for X → Y)"""
        return self.dependent.issubset(self.determinant)
    
    def applies_to(self, relation: Set[str]) -> bool:
        """Check if FD applies to given relation schema"""
        return (self.determinant.issubset(relation) and 
                self.dependent.issubset(relation))


class ArmstrongAxioms:
    """Implementation of Armstrong's axioms for functional dependencies"""
    
    @staticmethod
    def reflexivity(attributes: Set[str]) -> List[FunctionalDependency]:
        """If Y ⊆ X, then X → Y"""
        fds = []
        for subset in powerset(attributes):
            if subset:  # Non-empty subset
                fd = FunctionalDependency(attributes, set(subset))
                fds.append(fd)
        return fds
    
    @staticmethod
    def augmentation(fd: FunctionalDependency, W: Set[str]) -> FunctionalDependency:
        """If X → Y, then XW → YW"""
        return FunctionalDependency(
            fd.determinant.union(W),
            fd.dependent.union(W)
        )
    
    @staticmethod
    def transitivity(fd1: FunctionalDependency, fd2: FunctionalDependency) -> FunctionalDependency:
        """If X → Y and Y → Z, then X → Z"""
        if fd1.dependent == fd2.determinant:
            return FunctionalDependency(fd1.determinant, fd2.dependent)
        return None
    
    @staticmethod
    def union(fd1: FunctionalDependency, fd2: FunctionalDependency) -> FunctionalDependency:
        """If X → Y and X → Z, then X → YZ (derived rule)"""
        if fd1.determinant == fd2.determinant:
            return FunctionalDependency(
                fd1.determinant,
                fd1.dependent.union(fd2.dependent)
            )
        return None
    
    @staticmethod
    def decomposition(fd: FunctionalDependency) -> List[FunctionalDependency]:
        """If X → YZ, then X → Y and X → Z (derived rule)"""
        fds = []
        for attr in fd.dependent:
            fds.append(FunctionalDependency(fd.determinant, {attr}))
        return fds
    
    @staticmethod
    def compute_closure(attributes: Set[str], fds: List[FunctionalDependency]) -> Set[str]:
        """Compute attribute closure X+ under FDs"""
        closure = set(attributes)
        changed = True
        
        while changed:
            changed = False
            for fd in fds:
                if fd.determinant.issubset(closure) and not fd.dependent.issubset(closure):
                    closure.update(fd.dependent)
                    changed = True
        
        return closure
    
    @staticmethod
    def compute_fd_closure(fds: List[FunctionalDependency]) -> Set[FunctionalDependency]:
        """Compute closure of a set of functional dependencies"""
        closure = set(fds)
        changed = True
        
        while changed:
            changed = False
            new_fds = set()
            
            # Apply reflexivity
            all_attrs = set()
            for fd in closure:
                all_attrs.update(fd.determinant)
                all_attrs.update(fd.dependent)
            
            for subset in powerset(all_attrs):
                if subset:
                    for superset in powerset(all_attrs):
                        if set(subset).issubset(set(superset)):
                            new_fd = FunctionalDependency(set(superset), set(subset))
                            if new_fd not in closure and not new_fd.is_trivial():
                                new_fds.add(new_fd)
            
            # Apply augmentation
            for fd in closure:
                for attrs in powerset(all_attrs):
                    if attrs:
                        new_fd = ArmstrongAxioms.augmentation(fd, set(attrs))
                        if new_fd not in closure and not new_fd.is_trivial():
                            new_fds.add(new_fd)
            
            # Apply transitivity
            for fd1 in closure:
                for fd2 in closure:
                    new_fd = ArmstrongAxioms.transitivity(fd1, fd2)
                    if new_fd and new_fd not in closure and not new_fd.is_trivial():
                        new_fds.add(new_fd)
            
            if new_fds:
                closure.update(new_fds)
                changed = True
        
        return closure
    
    @staticmethod
    def minimal_cover(fds: List[FunctionalDependency]) -> List[FunctionalDependency]:
        """Compute minimal cover (canonical cover) of FDs"""
        # Step 1: Decompose right-hand sides
        decomposed = []
        for fd in fds:
            decomposed.extend(ArmstrongAxioms.decomposition(fd))
        
        # Step 2: Remove redundant FDs
        minimal = []
        for fd in decomposed:
            # Check if fd can be derived from others
            others = [f for f in decomposed if f != fd]
            closure = ArmstrongAxioms.compute_closure(fd.determinant, others)
            if not fd.dependent.issubset(closure):
                minimal.append(fd)
        
        # Step 3: Remove redundant attributes from left-hand sides
        final = []
        for fd in minimal:
            min_det = set(fd.determinant)
            for attr in fd.determinant:
                test_det = min_det - {attr}
                if test_det:
                    closure = ArmstrongAxioms.compute_closure(test_det, minimal)
                    if fd.dependent.issubset(closure):
                        min_det = test_det
            
            final.append(FunctionalDependency(min_det, fd.dependent))
        
        return final


class NormalFormChecker:
    """Check which normal form a relation satisfies"""
    
    @staticmethod
    def is_superkey(attrs: Set[str], relation: Set[str], fds: List[FunctionalDependency]) -> bool:
        """Check if attribute set is a superkey"""
        closure = ArmstrongAxioms.compute_closure(attrs, fds)
        return closure == relation
    
    @staticmethod
    def find_all_keys(relation: Set[str], fds: List[FunctionalDependency]) -> List[Set[str]]:
        """Find all candidate keys for a relation"""
        keys = []
        
        # Start with attributes that never appear on RHS
        essential = set(relation)
        for fd in fds:
            essential -= fd.dependent
        
        # Check all supersets of essential attributes
        for attrs in powerset(relation):
            attrs_set = essential.union(set(attrs))
            if NormalFormChecker.is_superkey(attrs_set, relation, fds):
                # Check if minimal
                is_minimal = True
                for attr in attrs_set:
                    if NormalFormChecker.is_superkey(attrs_set - {attr}, relation, fds):
                        is_minimal = False
                        break
                
                if is_minimal and attrs_set not in keys:
                    keys.append(attrs_set)
        
        return keys
    
    @staticmethod
    def is_2nf(relation: Set[str], fds: List[FunctionalDependency]) -> bool:
        """Check if relation is in 2NF"""
        keys = NormalFormChecker.find_all_keys(relation, fds)
        
        # Check for partial dependencies
        for fd in fds:
            # Skip trivial FDs
            if fd.is_trivial():
                continue
            
            # Check if LHS is a proper subset of any key
            for key in keys:
                if fd.determinant.issubset(key) and fd.determinant != key:
                    # Check if RHS contains non-key attributes
                    non_key_attrs = fd.dependent
                    for k in keys:
                        non_key_attrs -= k
                    
                    if non_key_attrs:
                        return False
        
        return True
    
    @staticmethod
    def is_3nf(relation: Set[str], fds: List[FunctionalDependency]) -> bool:
        """Check if relation is in 3NF"""
        if not NormalFormChecker.is_2nf(relation, fds):
            return False
        
        keys = NormalFormChecker.find_all_keys(relation, fds)
        key_attrs = set()
        for key in keys:
            key_attrs.update(key)
        
        # Check each FD
        for fd in fds:
            if fd.is_trivial():
                continue
            
            # Check conditions for 3NF
            # 1. X is a superkey, or
            # 2. Each attribute in Y-X is contained in a candidate key
            if not NormalFormChecker.is_superkey(fd.determinant, relation, fds):
                non_key_dependent = fd.dependent - fd.determinant
                if not non_key_dependent.issubset(key_attrs):
                    return False
        
        return True
    
    @staticmethod
    def is_bcnf(relation: Set[str], fds: List[FunctionalDependency]) -> bool:
        """Check if relation is in BCNF"""
        # For each non-trivial FD X → Y, X must be a superkey
        for fd in fds:
            if not fd.is_trivial() and fd.applies_to(relation):
                if not NormalFormChecker.is_superkey(fd.determinant, relation, fds):
                    return False
        
        return True


def bcnf_decomposition(relation: Set[str], fds: List[FunctionalDependency]) -> List[Set[str]]:
    """Decompose relation into BCNF"""
    
    def find_violating_fd(rel: Set[str], fds_subset: List[FunctionalDependency]) -> FunctionalDependency:
        """Find an FD that violates BCNF for the given relation"""
        for fd in fds_subset:
            if (fd.applies_to(rel) and 
                not fd.is_trivial() and
                not NormalFormChecker.is_superkey(fd.determinant, rel, fds_subset)):
                return fd
        return None
    
    def project_fds(relation_schema: Set[str], all_fds: List[FunctionalDependency]) -> List[FunctionalDependency]:
        """Project FDs onto a relation schema"""
        projected = []
        
        # For each subset of the relation schema
        for subset in powerset(relation_schema):
            if not subset:
                continue
            
            subset_set = set(subset)
            # Compute closure under all FDs
            closure = ArmstrongAxioms.compute_closure(subset_set, all_fds)
            
            # Keep only attributes in the relation
            closure_in_relation = closure.intersection(relation_schema)
            
            # Create FD if non-trivial
            if closure_in_relation != subset_set:
                fd = FunctionalDependency(subset_set, closure_in_relation - subset_set)
                if fd not in projected:
                    projected.append(fd)
        
        return projected
    
    # Start with the original relation
    result = [relation]
    changed = True
    
    while changed:
        changed = False
        
        for i, rel in enumerate(result):
            # Project FDs onto this relation
            rel_fds = project_fds(rel, fds)
            
            # Find violating FD
            violating_fd = find_violating_fd(rel, rel_fds)
            
            if violating_fd:
                # Decompose
                result.pop(i)
                
                # R1 = closure of violating FD's determinant
                r1 = ArmstrongAxioms.compute_closure(violating_fd.determinant, fds)
                r1 = r1.intersection(rel)  # Keep only attributes from current relation
                
                # R2 = determinant + remaining attributes
                r2 = violating_fd.determinant.union(rel - violating_fd.dependent)
                
                result.extend([r1, r2])
                changed = True
                break
    
    return result


def synthesis_algorithm_3nf(relation: Set[str], fds: List[FunctionalDependency]) -> List[Set[str]]:
    """3NF synthesis algorithm - lossless join and dependency preserving"""
    
    # Step 1: Find minimal cover
    minimal_fds = ArmstrongAxioms.minimal_cover(fds)
    
    # Step 2: Create a relation for each FD in minimal cover
    relations = []
    for fd in minimal_fds:
        rel_schema = fd.determinant.union(fd.dependent)
        # Avoid duplicate relations
        if rel_schema not in relations:
            relations.append(rel_schema)
    
    # Step 3: Ensure lossless join - add relation with candidate key if needed
    keys = NormalFormChecker.find_all_keys(relation, fds)
    if keys:
        key = min(keys, key=len)  # Choose smallest key
        key_found = False
        
        for rel in relations:
            if key.issubset(rel):
                key_found = True
                break
        
        if not key_found:
            relations.append(key)
    
    # Step 4: Remove redundant relations
    final_relations = []
    for rel in relations:
        is_redundant = False
        for other_rel in relations:
            if rel != other_rel and rel.issubset(other_rel):
                is_redundant = True
                break
        
        if not is_redundant:
            final_relations.append(rel)
    
    return final_relations


# Example usage and testing functions
def demonstrate_normalization():
    """Example of normalization process"""
    
    # Example: Order processing system
    # OrderID, CustomerID, CustomerName, ProductID, ProductName, Quantity, Price
    
    relation = {'OrderID', 'CustomerID', 'CustomerName', 'ProductID', 
                'ProductName', 'Quantity', 'Price'}
    
    fds = [
        FunctionalDependency({'OrderID', 'ProductID'}, {'Quantity'}),
        FunctionalDependency({'CustomerID'}, {'CustomerName'}),
        FunctionalDependency({'ProductID'}, {'ProductName', 'Price'}),
        FunctionalDependency({'OrderID'}, {'CustomerID'})
    ]
    
    print("Original relation:", relation)
    print("\nFunctional dependencies:")
    for fd in fds:
        print(f"  {fd}")
    
    # Check normal forms
    print(f"\nIs in 2NF: {NormalFormChecker.is_2nf(relation, fds)}")
    print(f"Is in 3NF: {NormalFormChecker.is_3nf(relation, fds)}")
    print(f"Is in BCNF: {NormalFormChecker.is_bcnf(relation, fds)}")
    
    # Find candidate keys
    keys = NormalFormChecker.find_all_keys(relation, fds)
    print(f"\nCandidate keys: {keys}")
    
    # Decompose to BCNF
    bcnf_relations = bcnf_decomposition(relation, fds)
    print("\nBCNF decomposition:")
    for i, rel in enumerate(bcnf_relations):
        print(f"  R{i+1}: {rel}")
    
    # 3NF synthesis
    tnf_relations = synthesis_algorithm_3nf(relation, fds)
    print("\n3NF synthesis:")
    for i, rel in enumerate(tnf_relations):
        print(f"  R{i+1}: {rel}")


def closure_example():
    """Example of computing attribute closure"""
    
    fds = [
        FunctionalDependency({'A'}, {'B'}),
        FunctionalDependency({'B', 'C'}, {'D'}),
        FunctionalDependency({'D'}, {'E'}),
        FunctionalDependency({'E'}, {'A'})
    ]
    
    # Compute closure of {A, C}
    attrs = {'A', 'C'}
    closure = ArmstrongAxioms.compute_closure(attrs, fds)
    print(f"Closure of {attrs}: {closure}")
    
    # Find minimal cover
    minimal = ArmstrongAxioms.minimal_cover(fds)
    print("\nMinimal cover:")
    for fd in minimal:
        print(f"  {fd}")


if __name__ == "__main__":
    demonstrate_normalization()
    print("\n" + "="*50 + "\n")
    closure_example()