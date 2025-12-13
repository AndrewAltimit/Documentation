"""
Git Workflow Modeling and Patterns

Formal modeling of Git workflows using state machines and graph theory:
- GitFlow workflow
- GitHub Flow
- GitLab Flow
- Monorepo workflows
- Custom workflow validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


class WorkflowState(Enum):
    """States in a Git workflow state machine"""

    DEVELOPMENT = "development"
    FEATURE = "feature"
    INTEGRATION = "integration"
    STAGING = "staging"
    PRODUCTION = "production"
    HOTFIX = "hotfix"
    RELEASE = "release"
    REVIEW = "review"
    TESTING = "testing"


@dataclass
class TransitionRule:
    """Rule for state transitions in workflow"""

    from_state: WorkflowState
    to_state: WorkflowState
    condition: Optional[str] = None
    action: Optional[str] = None


@dataclass
class Branch:
    """Git branch with metadata"""

    name: str
    state: WorkflowState
    source_branch: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class WorkflowViolation:
    """Represents a workflow policy violation"""

    branch: str
    violation_type: str
    message: str
    severity: str  # 'error', 'warning'


class GitWorkflow(ABC):
    """Abstract base for Git workflow implementations"""

    def __init__(self):
        self.state_graph = nx.DiGraph()
        self.branch_mapping: Dict[str, WorkflowState] = {}
        self.transition_rules: List[TransitionRule] = []
        self.branch_patterns: Dict[str, WorkflowState] = {}
        self._define_workflow()

    @abstractmethod
    def _define_workflow(self):
        """Define workflow states and transitions"""
        pass

    def validate_transition(
        self, from_branch: str, to_branch: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate if transition is allowed"""
        from_state = self._get_branch_state(from_branch)
        to_state = self._get_branch_state(to_branch)

        if not from_state or not to_state:
            return False, "Unknown branch state"

        # Check if path exists in state graph
        if not nx.has_path(self.state_graph, from_state, to_state):
            return False, f"No valid path from {from_state.value} to {to_state.value}"

        # Check transition rules
        for rule in self.transition_rules:
            if rule.from_state == from_state and rule.to_state == to_state:
                if rule.condition:
                    # Evaluate condition (simplified)
                    if not self._evaluate_condition(
                        rule.condition, from_branch, to_branch
                    ):
                        return False, f"Condition not met: {rule.condition}"

        return True, None

    def suggest_next_actions(self, current_branch: str) -> List[Dict[str, str]]:
        """Suggest valid next actions based on current state"""
        current_state = self._get_branch_state(current_branch)
        if not current_state:
            return []

        # Find reachable states
        if current_state not in self.state_graph:
            return []

        reachable = nx.descendants(self.state_graph, current_state)

        actions = []
        for state in reachable:
            # Find shortest path
            path = nx.shortest_path(self.state_graph, current_state, state)

            # Get example branches for this state
            example_branches = self._get_branches_for_state(state)

            actions.append(
                {
                    "target_state": state.value,
                    "path_length": len(path) - 1,
                    "example_branch": (
                        example_branches[0] if example_branches else f"{state.value}/*"
                    ),
                    "next_step": path[1].value if len(path) > 1 else state.value,
                }
            )

        return sorted(actions, key=lambda x: x["path_length"])

    def analyze_repository(self, branches: List[str]) -> Dict[str, any]:
        """Analyze repository compliance with workflow"""
        analysis = {
            "total_branches": len(branches),
            "state_distribution": {},
            "violations": [],
            "orphan_branches": [],
            "recommendations": [],
        }

        # Categorize branches by state
        state_counts = {state: 0 for state in WorkflowState}

        for branch in branches:
            state = self._get_branch_state(branch)
            if state:
                state_counts[state] += 1
            else:
                analysis["orphan_branches"].append(branch)

        analysis["state_distribution"] = {
            state.value: count for state, count in state_counts.items() if count > 0
        }

        # Check for violations
        analysis["violations"] = self._check_violations(branches)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _get_branch_state(self, branch: str) -> Optional[WorkflowState]:
        """Determine state of a branch based on patterns"""
        # Check exact mapping first
        if branch in self.branch_mapping:
            return self.branch_mapping[branch]

        # Check patterns
        for pattern, state in self.branch_patterns.items():
            if self._matches_pattern(branch, pattern):
                return state

        return None

    def _matches_pattern(self, branch: str, pattern: str) -> bool:
        """Check if branch matches pattern (with wildcards)"""
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            return branch.startswith(prefix)
        return branch == pattern

    def _get_branches_for_state(self, state: WorkflowState) -> List[str]:
        """Get example branches for a state"""
        branches = []

        # Check exact mappings
        for branch, branch_state in self.branch_mapping.items():
            if branch_state == state:
                branches.append(branch)

        # Check patterns
        for pattern, pattern_state in self.branch_patterns.items():
            if pattern_state == state:
                branches.append(pattern)

        return branches

    def _evaluate_condition(
        self, condition: str, from_branch: str, to_branch: str
    ) -> bool:
        """Evaluate transition condition"""
        # Simplified condition evaluation
        # In practice, would parse and evaluate complex conditions

        if condition == "tests_passing":
            return True  # Would check CI status
        elif condition == "approved":
            return True  # Would check PR approval
        elif condition == "no_conflicts":
            return True  # Would check merge conflicts

        return True

    def _check_violations(self, branches: List[str]) -> List[WorkflowViolation]:
        """Check for workflow violations"""
        violations = []

        # Override in specific workflows

        return violations

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate workflow recommendations"""
        recommendations = []

        # Check for too many feature branches
        feature_count = analysis["state_distribution"].get("feature", 0)
        if feature_count > 10:
            recommendations.append(
                f"Consider merging or cleaning up old feature branches ({feature_count} found)"
            )

        # Check for orphan branches
        if analysis["orphan_branches"]:
            recommendations.append(
                f"Found {len(analysis['orphan_branches'])} branches that don't follow naming conventions"
            )

        return recommendations


class GitFlowWorkflow(GitWorkflow):
    """GitFlow workflow implementation"""

    def _define_workflow(self):
        # Define states
        states = [
            WorkflowState.DEVELOPMENT,
            WorkflowState.FEATURE,
            WorkflowState.RELEASE,
            WorkflowState.PRODUCTION,
            WorkflowState.HOTFIX,
        ]

        self.state_graph.add_nodes_from(states)

        # Define transitions
        transitions = [
            # Feature flow
            (WorkflowState.DEVELOPMENT, WorkflowState.FEATURE),
            (WorkflowState.FEATURE, WorkflowState.DEVELOPMENT),
            # Release flow
            (WorkflowState.DEVELOPMENT, WorkflowState.RELEASE),
            (WorkflowState.RELEASE, WorkflowState.PRODUCTION),
            (WorkflowState.RELEASE, WorkflowState.DEVELOPMENT),
            # Hotfix flow
            (WorkflowState.PRODUCTION, WorkflowState.HOTFIX),
            (WorkflowState.HOTFIX, WorkflowState.PRODUCTION),
            (WorkflowState.HOTFIX, WorkflowState.DEVELOPMENT),
            # Direct release (for initial setup)
            (WorkflowState.DEVELOPMENT, WorkflowState.PRODUCTION),
        ]

        self.state_graph.add_edges_from(transitions)

        # Map branches to states
        self.branch_mapping = {
            "develop": WorkflowState.DEVELOPMENT,
            "master": WorkflowState.PRODUCTION,
            "main": WorkflowState.PRODUCTION,
        }

        # Define branch patterns
        self.branch_patterns = {
            "feature/*": WorkflowState.FEATURE,
            "release/*": WorkflowState.RELEASE,
            "hotfix/*": WorkflowState.HOTFIX,
        }

        # Define transition rules
        self.transition_rules = [
            TransitionRule(
                WorkflowState.FEATURE,
                WorkflowState.DEVELOPMENT,
                condition="tests_passing",
            ),
            TransitionRule(
                WorkflowState.RELEASE, WorkflowState.PRODUCTION, condition="approved"
            ),
            TransitionRule(
                WorkflowState.HOTFIX,
                WorkflowState.PRODUCTION,
                condition="tests_passing",
            ),
        ]

    def create_feature_branch(self, feature_name: str) -> Branch:
        """Create feature branch following GitFlow"""
        # Features branch from develop
        source = "develop"
        branch_name = f"feature/{feature_name}"

        # Validate source exists
        if not self._get_branch_state(source):
            raise ValueError(f"Source branch {source} not found")

        # Create branch
        branch = Branch(
            name=branch_name, state=WorkflowState.FEATURE, source_branch=source
        )

        return branch

    def start_release(self, version: str) -> Branch:
        """Start release branch"""
        source = "develop"
        branch_name = f"release/{version}"

        branch = Branch(
            name=branch_name, state=WorkflowState.RELEASE, source_branch=source
        )

        return branch

    def _check_violations(self, branches: List[str]) -> List[WorkflowViolation]:
        violations = []

        # Check for direct commits to master/main
        protected_branches = ["master", "main", "develop"]

        # Check for feature branches not from develop
        for branch in branches:
            if branch.startswith("feature/"):
                # Would check actual source branch
                pass

        # Check for old release branches
        release_branches = [b for b in branches if b.startswith("release/")]
        if len(release_branches) > 1:
            violations.append(
                WorkflowViolation(
                    branch="release/*",
                    violation_type="multiple_releases",
                    message=f"Multiple release branches found: {len(release_branches)}",
                    severity="warning",
                )
            )

        return violations


class GitHubFlow(GitWorkflow):
    """GitHub Flow - simplified workflow"""

    def _define_workflow(self):
        # Define states
        states = [WorkflowState.PRODUCTION, WorkflowState.FEATURE, WorkflowState.REVIEW]

        self.state_graph.add_nodes_from(states)

        # Define transitions
        transitions = [
            (WorkflowState.PRODUCTION, WorkflowState.FEATURE),
            (WorkflowState.FEATURE, WorkflowState.REVIEW),
            (WorkflowState.REVIEW, WorkflowState.PRODUCTION),
            (WorkflowState.REVIEW, WorkflowState.FEATURE),  # Changes requested
        ]

        self.state_graph.add_edges_from(transitions)

        # Map branches
        self.branch_mapping = {
            "main": WorkflowState.PRODUCTION,
            "master": WorkflowState.PRODUCTION,
        }

        # All other branches are features
        self.branch_patterns = {
            "*": WorkflowState.FEATURE,
        }


class GitLabFlow(GitWorkflow):
    """GitLab Flow with environment branches"""

    def _define_workflow(self):
        # Define states
        states = [
            WorkflowState.DEVELOPMENT,
            WorkflowState.FEATURE,
            WorkflowState.STAGING,
            WorkflowState.PRODUCTION,
        ]

        self.state_graph.add_nodes_from(states)

        # Define transitions
        transitions = [
            # Feature development
            (WorkflowState.DEVELOPMENT, WorkflowState.FEATURE),
            (WorkflowState.FEATURE, WorkflowState.DEVELOPMENT),
            # Environment promotion
            (WorkflowState.DEVELOPMENT, WorkflowState.STAGING),
            (WorkflowState.STAGING, WorkflowState.PRODUCTION),
            # Hotfix path
            (WorkflowState.PRODUCTION, WorkflowState.DEVELOPMENT),
        ]

        self.state_graph.add_edges_from(transitions)

        # Map branches
        self.branch_mapping = {
            "main": WorkflowState.DEVELOPMENT,
            "staging": WorkflowState.STAGING,
            "production": WorkflowState.PRODUCTION,
        }

        self.branch_patterns = {
            "feature/*": WorkflowState.FEATURE,
            "bugfix/*": WorkflowState.FEATURE,
        }


class MonorepoWorkflow(GitWorkflow):
    """Workflow for monorepo management"""

    def __init__(self, projects: List[str]):
        self.projects = projects
        self.project_dependencies = self._analyze_dependencies()
        super().__init__()

    def _define_workflow(self):
        # Similar to GitHub flow but with project awareness
        states = [
            WorkflowState.PRODUCTION,
            WorkflowState.FEATURE,
            WorkflowState.INTEGRATION,
            WorkflowState.TESTING,
        ]

        self.state_graph.add_nodes_from(states)

        transitions = [
            (WorkflowState.PRODUCTION, WorkflowState.FEATURE),
            (WorkflowState.FEATURE, WorkflowState.INTEGRATION),
            (WorkflowState.INTEGRATION, WorkflowState.TESTING),
            (WorkflowState.TESTING, WorkflowState.PRODUCTION),
            (WorkflowState.TESTING, WorkflowState.FEATURE),  # Failed tests
        ]

        self.state_graph.add_edges_from(transitions)

        self.branch_mapping = {
            "main": WorkflowState.PRODUCTION,
        }

        # Project-specific branch patterns
        for project in self.projects:
            self.branch_patterns[f"{project}/*"] = WorkflowState.FEATURE

        self.branch_patterns["integration/*"] = WorkflowState.INTEGRATION

    def _analyze_dependencies(self) -> nx.DiGraph:
        """Analyze inter-project dependencies"""
        dep_graph = nx.DiGraph()

        # Simplified - would parse actual dependencies
        for project in self.projects:
            dep_graph.add_node(project)

        # Example dependencies
        if "frontend" in self.projects and "api" in self.projects:
            dep_graph.add_edge("frontend", "api")

        if "api" in self.projects and "database" in self.projects:
            dep_graph.add_edge("api", "database")

        return dep_graph

    def affected_projects(self, changed_files: List[str]) -> Set[str]:
        """Determine projects affected by changes"""
        directly_affected = set()

        # Find directly affected projects
        for file in changed_files:
            project = self._file_to_project(file)
            if project:
                directly_affected.add(project)

        # Find transitively affected projects
        all_affected = set(directly_affected)
        for project in directly_affected:
            # All projects that depend on this one
            if project in self.project_dependencies:
                dependents = nx.ancestors(self.project_dependencies, project)
                all_affected.update(dependents)

        return all_affected

    def _file_to_project(self, file_path: str) -> Optional[str]:
        """Map file path to project"""
        for project in self.projects:
            if file_path.startswith(f"{project}/"):
                return project
        return None

    def validate_changes(
        self, branch: str, changed_files: List[str]
    ) -> List[WorkflowViolation]:
        """Validate changes follow monorepo policies"""
        violations = []

        affected = self.affected_projects(changed_files)

        # Check if branch name reflects affected projects
        if len(affected) == 1:
            project = list(affected)[0]
            if not branch.startswith(f"{project}/"):
                violations.append(
                    WorkflowViolation(
                        branch=branch,
                        violation_type="branch_naming",
                        message=f"Branch should start with {project}/ for single-project changes",
                        severity="warning",
                    )
                )
        elif len(affected) > 1:
            if not branch.startswith("integration/"):
                violations.append(
                    WorkflowViolation(
                        branch=branch,
                        violation_type="multi_project_change",
                        message=f"Multi-project changes should use integration/* branch",
                        severity="warning",
                    )
                )

        return violations


class WorkflowAnalyzer:
    """Analyze and visualize Git workflows"""

    def __init__(self, workflow: GitWorkflow):
        self.workflow = workflow

    def visualize_workflow(self) -> str:
        """Generate workflow visualization (Mermaid format)"""
        lines = ["graph TD"]

        # Add nodes
        for state in self.workflow.state_graph.nodes():
            lines.append(f"    {state.value}[{state.value.title()}]")

        # Add edges
        for source, target in self.workflow.state_graph.edges():
            lines.append(f"    {source.value} --> {target.value}")

        return "\n".join(lines)

    def find_bottlenecks(self) -> List[WorkflowState]:
        """Find potential bottlenecks in workflow"""
        bottlenecks = []

        # Find nodes with high in-degree and low out-degree
        for node in self.workflow.state_graph.nodes():
            in_degree = self.workflow.state_graph.in_degree(node)
            out_degree = self.workflow.state_graph.out_degree(node)

            if in_degree > 2 and out_degree <= 1:
                bottlenecks.append(node)

        return bottlenecks

    def calculate_metrics(
        self, branch_history: List[Tuple[str, str, str]]
    ) -> Dict[str, any]:
        """Calculate workflow metrics from branch history"""
        metrics = {
            "avg_feature_lifetime": 0,
            "bottleneck_states": [],
            "transition_frequency": {},
            "violation_rate": 0,
        }

        # Analyze transitions
        transition_counts = {}
        for from_branch, to_branch, timestamp in branch_history:
            from_state = self.workflow._get_branch_state(from_branch)
            to_state = self.workflow._get_branch_state(to_branch)

            if from_state and to_state:
                key = f"{from_state.value}->{to_state.value}"
                transition_counts[key] = transition_counts.get(key, 0) + 1

        metrics["transition_frequency"] = transition_counts

        # Find bottlenecks
        metrics["bottleneck_states"] = [s.value for s in self.find_bottlenecks()]

        return metrics


# Example usage
def demo_workflows():
    """Demonstrate workflow modeling"""
    print("Git Workflow Modeling Demo")
    print("=" * 50)

    # GitFlow example
    print("\n1. GitFlow Workflow:")
    gitflow = GitFlowWorkflow()

    # Create feature branch
    feature = gitflow.create_feature_branch("user-authentication")
    print(f"Created branch: {feature.name} (state: {feature.state.value})")

    # Validate transition
    valid, reason = gitflow.validate_transition(
        "feature/user-authentication", "develop"
    )
    print(f"Can merge to develop: {valid}")

    # Get suggestions
    suggestions = gitflow.suggest_next_actions("develop")
    print("\nNext actions from develop:")
    for suggestion in suggestions[:3]:
        print(f"  - {suggestion['target_state']}: {suggestion['example_branch']}")

    # Analyze repository
    branches = [
        "main",
        "develop",
        "feature/auth",
        "feature/ui",
        "feature/api",
        "release/1.0",
        "release/1.1",
        "hotfix/security",
    ]

    analysis = gitflow.analyze_repository(branches)
    print(f"\nRepository analysis:")
    print(f"  Total branches: {analysis['total_branches']}")
    print(f"  State distribution: {analysis['state_distribution']}")
    print(f"  Violations: {len(analysis['violations'])}")

    # Monorepo example
    print("\n\n2. Monorepo Workflow:")
    projects = ["frontend", "api", "database", "shared"]
    monorepo = MonorepoWorkflow(projects)

    # Check affected projects
    changed_files = ["api/src/users.js", "shared/utils.js", "frontend/package.json"]

    affected = monorepo.affected_projects(changed_files)
    print(f"Changed files affect projects: {affected}")

    # Validate changes
    violations = monorepo.validate_changes("feature/update-api", changed_files)
    if violations:
        print("Violations found:")
        for v in violations:
            print(f"  - {v.violation_type}: {v.message}")

    # Visualize workflow
    print("\n\n3. Workflow Visualization (Mermaid):")
    analyzer = WorkflowAnalyzer(gitflow)
    print(analyzer.visualize_workflow())


if __name__ == "__main__":
    demo_workflows()
