"""
Git Performance Analysis and Optimization

Advanced techniques for analyzing and optimizing Git repository performance:
- Object database analysis
- Pack file optimization 
- Bitmap indexes for reachability
- Geometric repacking
- Performance metrics and profiling
"""

import os
import hashlib
import struct
import time
from typing import Dict, List, Optional, Set, Tuple, BinaryIO
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class ObjectStats:
    """Statistics about objects in repository"""
    count_by_type: Dict[str, int]
    total_size: int
    large_objects: List['LargeObject']
    unreachable_objects: Set[str]
    
    def __init__(self):
        self.count_by_type = defaultdict(int)
        self.total_size = 0
        self.large_objects = []
        self.unreachable_objects = set()


@dataclass
class LargeObject:
    """Large object information"""
    sha: str
    type: str
    size: int
    path: Optional[str] = None


@dataclass
class PackStats:
    """Pack file statistics"""
    num_packs: int
    total_size: int
    num_objects: int
    delta_objects: int
    efficiency: float  # Compression ratio


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report"""
    object_stats: ObjectStats
    pack_stats: PackStats
    ref_stats: Dict[str, any]
    worktree_stats: Dict[str, any]
    recommendations: List[str]


@dataclass
class OptimizationResult:
    """Result of optimization operation"""
    success: bool
    space_saved: int
    time_taken: float
    new_packs: List[str]
    message: str


class PerformanceAnalyzer:
    """Analyze and optimize Git repository performance"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.git_dir = os.path.join(repo_path, ".git")
        self.metrics = {}
    
    def analyze_repository(self) -> PerformanceReport:
        """Comprehensive performance analysis"""
        report = PerformanceReport(
            object_stats=ObjectStats(),
            pack_stats=self.analyze_pack_files(),
            ref_stats={},
            worktree_stats={},
            recommendations=[]
        )
        
        # Object database metrics
        report.object_stats = self.analyze_object_database()
        
        # Reference performance
        report.ref_stats = self.analyze_references()
        
        # Working tree analysis
        report.worktree_stats = self.analyze_working_tree()
        
        # Generate recommendations
        report.recommendations = self.generate_recommendations(report)
        
        return report
    
    def analyze_object_database(self) -> ObjectStats:
        """Analyze object database performance"""
        stats = ObjectStats()
        
        # Analyze loose objects
        objects_dir = os.path.join(self.git_dir, "objects")
        
        for subdir in os.listdir(objects_dir):
            if len(subdir) == 2:  # Object subdirectory
                subdir_path = os.path.join(objects_dir, subdir)
                if os.path.isdir(subdir_path):
                    for obj_file in os.listdir(subdir_path):
                        sha = subdir + obj_file
                        obj_path = os.path.join(subdir_path, obj_file)
                        
                        # Get object info
                        obj_type, obj_size = self._get_object_info(sha)
                        
                        stats.count_by_type[obj_type] += 1
                        stats.total_size += obj_size
                        
                        # Track large objects
                        if obj_size > 10 * 1024 * 1024:  # 10MB
                            stats.large_objects.append(
                                LargeObject(sha, obj_type, obj_size)
                            )
        
        # Analyze pack files
        pack_dir = os.path.join(objects_dir, "pack")
        if os.path.exists(pack_dir):
            for pack_file in os.listdir(pack_dir):
                if pack_file.endswith(".pack"):
                    pack_path = os.path.join(pack_dir, pack_file)
                    self._analyze_pack_objects(pack_path, stats)
        
        # Find unreachable objects
        stats.unreachable_objects = self.find_unreachable_objects()
        
        return stats
    
    def analyze_pack_files(self) -> PackStats:
        """Analyze pack file efficiency"""
        pack_dir = os.path.join(self.git_dir, "objects", "pack")
        
        stats = PackStats(
            num_packs=0,
            total_size=0,
            num_objects=0,
            delta_objects=0,
            efficiency=0.0
        )
        
        if not os.path.exists(pack_dir):
            return stats
        
        for pack_file in os.listdir(pack_dir):
            if pack_file.endswith(".pack"):
                pack_path = os.path.join(pack_dir, pack_file)
                stats.num_packs += 1
                stats.total_size += os.path.getsize(pack_path)
                
                # Read pack header
                with open(pack_path, 'rb') as f:
                    # Skip signature and version
                    f.seek(8)
                    num_objects = struct.unpack('>I', f.read(4))[0]
                    stats.num_objects += num_objects
        
        # Calculate efficiency
        if stats.total_size > 0:
            # Estimate uncompressed size
            avg_object_size = 1024  # 1KB average
            uncompressed = stats.num_objects * avg_object_size
            stats.efficiency = 1.0 - (stats.total_size / uncompressed)
        
        return stats
    
    def analyze_references(self) -> Dict[str, any]:
        """Analyze reference performance"""
        refs_dir = os.path.join(self.git_dir, "refs")
        
        stats = {
            'num_refs': 0,
            'num_loose_refs': 0,
            'num_packed_refs': 0,
            'ref_depth': {},
            'stale_refs': []
        }
        
        # Count loose refs
        for root, dirs, files in os.walk(refs_dir):
            for file in files:
                stats['num_refs'] += 1
                stats['num_loose_refs'] += 1
                
                # Calculate directory depth
                depth = root.replace(refs_dir, '').count(os.sep)
                stats['ref_depth'][depth] = stats['ref_depth'].get(depth, 0) + 1
        
        # Check packed refs
        packed_refs_path = os.path.join(self.git_dir, "packed-refs")
        if os.path.exists(packed_refs_path):
            with open(packed_refs_path, 'r') as f:
                for line in f:
                    if not line.startswith('#') and not line.startswith('^'):
                        stats['num_packed_refs'] += 1
        
        # Find stale remote tracking branches
        # Simplified - would check against remote
        
        return stats
    
    def analyze_working_tree(self) -> Dict[str, any]:
        """Analyze working tree performance"""
        stats = {
            'num_files': 0,
            'total_size': 0,
            'large_files': [],
            'untracked_files': 0,
            'ignored_size': 0
        }
        
        # Walk working tree
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if '.git' in root:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    stats['num_files'] += 1
                    stats['total_size'] += file_size
                    
                    # Track large files
                    if file_size > 50 * 1024 * 1024:  # 50MB
                        stats['large_files'].append({
                            'path': os.path.relpath(file_path, self.repo_path),
                            'size': file_size
                        })
                except OSError:
                    pass
        
        return stats
    
    def optimize_pack_files(self) -> OptimizationResult:
        """Optimize pack file organization"""
        start_time = time.time()
        
        # Get current pack statistics
        before_stats = self.analyze_pack_files()
        
        # Determine optimization strategy
        if self.is_large_repository():
            result = self.geometric_repack()
        else:
            result = self.standard_repack()
        
        # Get after statistics
        after_stats = self.analyze_pack_files()
        
        space_saved = before_stats.total_size - after_stats.total_size
        time_taken = time.time() - start_time
        
        return OptimizationResult(
            success=True,
            space_saved=space_saved,
            time_taken=time_taken,
            new_packs=result.new_packs,
            message=f"Optimized from {before_stats.num_packs} to {after_stats.num_packs} packs"
        )
    
    def geometric_repack(self) -> OptimizationResult:
        """Geometric repacking for better performance"""
        # Get existing packs
        packs = self.get_pack_files()
        
        # Sort by size
        packs.sort(key=lambda p: p['size'])
        
        # Create geometric series of pack sizes
        # Each pack is ~2x the size of previous
        new_packs = []
        current_group = []
        target_size = 1024 * 1024  # Start at 1MB
        
        for pack in packs:
            current_group.append(pack)
            
            group_size = sum(p['size'] for p in current_group)
            if group_size >= target_size:
                # Repack this group
                new_pack = self._repack_group(current_group)
                new_packs.append(new_pack)
                
                # Reset for next group
                current_group = []
                target_size *= 2  # Geometric progression
        
        # Handle remaining packs
        if current_group:
            new_pack = self._repack_group(current_group)
            new_packs.append(new_pack)
        
        return OptimizationResult(
            success=True,
            space_saved=0,
            time_taken=0,
            new_packs=[p['name'] for p in new_packs],
            message="Geometric repacking completed"
        )
    
    def standard_repack(self) -> OptimizationResult:
        """Standard repacking with optimal parameters"""
        # Repack configuration
        repack_config = {
            'window': 250,          # Larger window for better compression
            'depth': 50,            # Deeper delta chains
            'threads': os.cpu_count(),
            'compression': 9,       # Maximum compression
            'delta_base_offset': True
        }
        
        # In practice, would call git repack
        # git repack -a -d -f --window=250 --depth=50
        
        return OptimizationResult(
            success=True,
            space_saved=0,
            time_taken=0,
            new_packs=["pack-optimized.pack"],
            message="Standard repacking completed"
        )
    
    def find_unreachable_objects(self) -> Set[str]:
        """Find objects not reachable from any ref"""
        # Simplified - would use git fsck
        unreachable = set()
        
        # In practice:
        # git fsck --unreachable --no-reflogs
        
        return unreachable
    
    def generate_recommendations(self, report: PerformanceReport) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check loose objects
        loose_count = sum(v for k, v in report.object_stats.count_by_type.items())
        if loose_count > 1000:
            recommendations.append(
                f"Run 'git gc' to pack {loose_count} loose objects"
            )
        
        # Check pack file count
        if report.pack_stats.num_packs > 50:
            recommendations.append(
                f"Too many pack files ({report.pack_stats.num_packs}). "
                "Run 'git repack -ad' to consolidate"
            )
        
        # Check for large objects
        if report.object_stats.large_objects:
            recommendations.append(
                f"Found {len(report.object_stats.large_objects)} large objects. "
                "Consider using Git LFS"
            )
        
        # Check unreachable objects
        unreachable_count = len(report.object_stats.unreachable_objects)
        if unreachable_count > 100:
            recommendations.append(
                f"Found {unreachable_count} unreachable objects. "
                "Run 'git prune' to remove"
            )
        
        # Check ref organization
        if report.ref_stats['num_loose_refs'] > 100:
            recommendations.append(
                "Many loose refs. Run 'git pack-refs --all' to optimize"
            )
        
        # Check working tree
        large_files = report.worktree_stats.get('large_files', [])
        if large_files:
            recommendations.append(
                f"Found {len(large_files)} large files in working tree. "
                "Consider .gitignore or Git LFS"
            )
        
        return recommendations
    
    def is_large_repository(self) -> bool:
        """Check if repository is considered large"""
        stats = self.analyze_pack_files()
        
        # Large if > 1GB or > 100k objects
        return (stats.total_size > 1024 * 1024 * 1024 or 
                stats.num_objects > 100000)
    
    def get_pack_files(self) -> List[Dict[str, any]]:
        """Get list of pack files with metadata"""
        pack_dir = os.path.join(self.git_dir, "objects", "pack")
        packs = []
        
        if os.path.exists(pack_dir):
            for pack_file in os.listdir(pack_dir):
                if pack_file.endswith(".pack"):
                    pack_path = os.path.join(pack_dir, pack_file)
                    packs.append({
                        'name': pack_file,
                        'path': pack_path,
                        'size': os.path.getsize(pack_path)
                    })
        
        return packs
    
    def _get_object_info(self, sha: str) -> Tuple[str, int]:
        """Get object type and size"""
        # Simplified - would use git cat-file
        return "blob", 1024  # Mock data
    
    def _analyze_pack_objects(self, pack_path: str, stats: ObjectStats):
        """Analyze objects in pack file"""
        # Would parse pack file and update stats
        pass
    
    def _repack_group(self, packs: List[Dict]) -> Dict[str, any]:
        """Repack a group of pack files"""
        # Would call git repack on specific packs
        total_size = sum(p['size'] for p in packs)
        
        return {
            'name': f'pack-{hashlib.sha1(str(packs).encode()).hexdigest()[:8]}.pack',
            'size': int(total_size * 0.8)  # Assume 20% compression
        }


class BitmapIndex:
    """Reachability bitmap implementation for fast operations"""
    
    def __init__(self, pack_file: str):
        self.pack_file = pack_file
        self.bitmap_file = pack_file.replace('.pack', '.bitmap')
        self.bitmaps: Dict[str, 'EWAHBitmap'] = {}
        self.commit_positions: Dict[str, int] = {}
    
    def build_bitmaps(self, commits: List[str]):
        """Build reachability bitmaps for commits"""
        # Select commits for bitmap coverage
        selected_commits = self.select_bitmap_commits(commits)
        
        for i, commit in enumerate(selected_commits):
            # Build reachability bitmap
            bitmap = EWAHBitmap()
            reachable = self.find_reachable_objects(commit)
            
            for obj_idx in reachable:
                bitmap.set(obj_idx)
            
            # Compress bitmap
            bitmap.compress()
            self.bitmaps[commit] = bitmap
            self.commit_positions[commit] = i
    
    def select_bitmap_commits(self, commits: List[str]) -> List[str]:
        """Select optimal commits for bitmap coverage"""
        selected = []
        covered_objects = set()
        total_objects = self._count_total_objects()
        
        # Sort commits by recency and importance
        sorted_commits = self._sort_commits_by_importance(commits)
        
        for commit in sorted_commits:
            # Check if this commit adds significant coverage
            reachable = self.find_reachable_objects(commit)
            new_coverage = len(reachable - covered_objects)
            
            if new_coverage > total_objects * 0.01:  # At least 1% new coverage
                selected.append(commit)
                covered_objects.update(reachable)
                
                # Stop when we have good coverage
                if len(covered_objects) > total_objects * 0.95:
                    break
        
        return selected
    
    def query_reachability(self, commits: List[str]) -> Set[int]:
        """Fast reachability query using bitmaps"""
        if not commits:
            return set()
        
        # Start with first commit's bitmap
        if commits[0] in self.bitmaps:
            result = self.bitmaps[commits[0]].copy()
        else:
            # Compute bitmap for commit
            result = self._compute_bitmap(commits[0])
        
        # OR with remaining commits
        for commit in commits[1:]:
            if commit in self.bitmaps:
                result.or_with(self.bitmaps[commit])
            else:
                commit_bitmap = self._compute_bitmap(commit)
                result.or_with(commit_bitmap)
        
        return result.to_set()
    
    def write_bitmap_index(self):
        """Write bitmap index to disk"""
        with open(self.bitmap_file, 'wb') as f:
            # Write header
            f.write(b'BITM')  # Magic
            f.write(struct.pack('>I', 1))  # Version
            
            # Write bitmap entries
            f.write(struct.pack('>I', len(self.bitmaps)))
            
            for commit, bitmap in self.bitmaps.items():
                # Write commit SHA
                f.write(bytes.fromhex(commit))
                
                # Write bitmap data
                bitmap_data = bitmap.serialize()
                f.write(struct.pack('>I', len(bitmap_data)))
                f.write(bitmap_data)
    
    def find_reachable_objects(self, commit: str) -> Set[int]:
        """Find all objects reachable from commit"""
        # Simplified - would traverse commit graph
        return {i for i in range(100)}  # Mock data
    
    def _count_total_objects(self) -> int:
        """Count total objects in pack"""
        # Would read from pack index
        return 10000
    
    def _sort_commits_by_importance(self, commits: List[str]) -> List[str]:
        """Sort commits by importance for bitmap selection"""
        # Would consider:
        # - Commit date (recent first)
        # - Number of parents (merge commits)
        # - Branch tips
        # - Tags
        return sorted(commits)
    
    def _compute_bitmap(self, commit: str) -> 'EWAHBitmap':
        """Compute bitmap for commit not in index"""
        bitmap = EWAHBitmap()
        reachable = self.find_reachable_objects(commit)
        
        for obj_idx in reachable:
            bitmap.set(obj_idx)
        
        bitmap.compress()
        return bitmap


class EWAHBitmap:
    """Compressed bitmap using EWAH (Enhanced Word-Aligned Hybrid) encoding"""
    
    def __init__(self):
        self.words = []
        self.bits = set()
    
    def set(self, bit: int):
        """Set a bit"""
        self.bits.add(bit)
    
    def compress(self):
        """Compress bitmap using EWAH encoding"""
        # Simplified - real EWAH uses run-length encoding
        # for sequences of 0s and 1s
        pass
    
    def or_with(self, other: 'EWAHBitmap'):
        """OR operation with another bitmap"""
        self.bits |= other.bits
    
    def and_with(self, other: 'EWAHBitmap'):
        """AND operation with another bitmap"""
        self.bits &= other.bits
    
    def copy(self) -> 'EWAHBitmap':
        """Create copy of bitmap"""
        new_bitmap = EWAHBitmap()
        new_bitmap.bits = self.bits.copy()
        return new_bitmap
    
    def to_set(self) -> Set[int]:
        """Convert to set of integers"""
        return self.bits.copy()
    
    def serialize(self) -> bytes:
        """Serialize bitmap for storage"""
        # Simplified serialization
        data = struct.pack('>I', len(self.bits))
        for bit in sorted(self.bits):
            data += struct.pack('>I', bit)
        return data


# Example usage
def demo_performance_analysis():
    """Demonstrate performance analysis"""
    print("Git Performance Analysis Demo")
    print("=" * 50)
    
    # Create analyzer (would use real repo path)
    analyzer = PerformanceAnalyzer("/path/to/repo")
    
    # Simulated analysis report
    report = PerformanceReport(
        object_stats=ObjectStats(),
        pack_stats=PackStats(
            num_packs=15,
            total_size=500 * 1024 * 1024,  # 500MB
            num_objects=50000,
            delta_objects=35000,
            efficiency=0.7
        ),
        ref_stats={
            'num_refs': 150,
            'num_loose_refs': 120,
            'num_packed_refs': 30
        },
        worktree_stats={
            'num_files': 5000,
            'total_size': 100 * 1024 * 1024,  # 100MB
            'large_files': [
                {'path': 'data/large.bin', 'size': 25 * 1024 * 1024}
            ]
        },
        recommendations=[]
    )
    
    # Add some large objects
    report.object_stats.large_objects = [
        LargeObject("abc123", "blob", 15 * 1024 * 1024, "images/photo.jpg"),
        LargeObject("def456", "blob", 20 * 1024 * 1024, "videos/demo.mp4")
    ]
    
    # Generate recommendations
    report.recommendations = analyzer.generate_recommendations(report)
    
    # Display report
    print("\nRepository Performance Report:")
    print(f"  Pack files: {report.pack_stats.num_packs}")
    print(f"  Total size: {report.pack_stats.total_size / 1024 / 1024:.1f}MB")
    print(f"  Objects: {report.pack_stats.num_objects:,}")
    print(f"  Compression: {report.pack_stats.efficiency:.1%}")
    
    print(f"\n  Large objects: {len(report.object_stats.large_objects)}")
    for obj in report.object_stats.large_objects[:3]:
        print(f"    - {obj.path or obj.sha[:8]}: {obj.size / 1024 / 1024:.1f}MB")
    
    print(f"\n  References: {report.ref_stats['num_refs']}")
    print(f"    - Loose: {report.ref_stats['num_loose_refs']}")
    print(f"    - Packed: {report.ref_stats['num_packed_refs']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Demonstrate bitmap index
    print("\n\nBitmap Index Demo:")
    bitmap_idx = BitmapIndex("pack-abc123.pack")
    
    # Build bitmaps for some commits
    commits = [f"commit{i}" for i in range(10)]
    selected = bitmap_idx.select_bitmap_commits(commits)
    print(f"Selected {len(selected)} commits for bitmap coverage")
    
    # Query reachability
    query_commits = ["commit5", "commit7"]
    reachable = bitmap_idx.query_reachability(query_commits)
    print(f"Objects reachable from {query_commits}: {len(reachable)}")


if __name__ == "__main__":
    demo_performance_analysis()