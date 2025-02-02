import networkx as nx
import time
from collections import defaultdict
import community.community_louvain as community
from typing import List, Set, Dict
from itertools import combinations


class CommunityDetector:
    def __init__(self, network: nx.DiGraph, dept_labels: Dict[int, int]):
        """Initialize with network and department labels"""
        self.DG = network
        self.G = self.DG.to_undirected()
        self.n = self.DG.number_of_nodes()
        self.dept_labels = dept_labels
        self.shortest_paths = None
        if nx.is_connected(self.G):
            self.diameter = nx.diameter(self.G)
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
            self.diameter = nx.diameter(subgraph)

    @classmethod
    def from_files(cls, network_file: str, dept_file: str):
        """Load network and department data"""
        print(f"Loading network from {network_file}")
        network = nx.read_edgelist(network_file, create_using=nx.DiGraph(), nodetype=int)
        print(f"Loading department labels from {dept_file}")
        dept_labels = {}
        with open(dept_file, 'r') as f:
            for line in f:
                node, dept = map(int, line.strip().split())
                dept_labels[node] = dept

        return cls(network, dept_labels)

    def _compute_shortest_paths(self):
        """Compute shortest paths for a larger sample of nodes"""
        print("\nComputing shortest paths (sampling approach)...")
        nodes = list(self.DG.nodes())
        sample_size = min(300, len(nodes))  # Increase sample size to 300
        sample_nodes = set(nodes[:sample_size])
        paths = {}
        for node in sample_nodes:
            paths[node] = dict(nx.single_source_shortest_path_length(self.DG, node))

        return paths

    def distance_quality(self, communities: List[Set[int]], sample_paths: Dict) -> float:
        """Calculate more lenient distance quality for communities"""
        if not communities:
            return float('-inf')

        quality = 0.0
        total_pairs = 0

        sample_nodes = set(sample_paths.keys())

        for cluster in communities:
            cluster_sample = cluster.intersection(sample_nodes)
            if len(cluster_sample) < 2:
                continue

            # Calculate internal density
            internal_edges = sum(1 for i, j in combinations(cluster, 2)
                                 if j in self.DG[i] or i in self.DG[j])
            possible_edges = len(cluster) * (len(cluster) - 1) / 2
            density = internal_edges / possible_edges if possible_edges > 0 else 0

            for i, j in combinations(cluster_sample, 2):
                if j not in sample_paths[i]:
                    continue
                # Actual distance Dᵥ(i,j)
                actual_dist = sample_paths[i][j]
                # Reward shorter distances more in dense communities
                quality += (self.diameter - actual_dist) * (1 + density)
                total_pairs += 1

        return quality / max(1, total_pairs)

    def find_communities(self) -> List[Set[int]]:
        """Find communities with more aggressive merging strategy"""
        if self.shortest_paths is None:
            self.shortest_paths = self._compute_shortest_paths()

        print("\nFinding communities...")
        nodes = list(self.DG.nodes())
        current_communities = [{node} for node in nodes]
        best_quality = self.distance_quality(current_communities, self.shortest_paths)
        print(f"Initial quality: {best_quality: .4f}")

        iteration = 0
        no_improvement_count = 0
        min_communities = 25  # Target number similar to modularity

        while len(current_communities) > min_communities:
            iteration += 1
            improved = False
            current_quality = best_quality

            # Get all pairs of communities and their connection strengths
            community_pairs = []
            for i in range(len(current_communities)):
                for j in range(i + 1, len(current_communities)):
                    if len(current_communities[i]) + len(current_communities[j]) > self.n / 3:
                        continue

                    # Count connections between communities
                    connections = sum(1 for n1 in current_communities[i]
                                      for n2 in current_communities[j] if n2 in self.DG[n1])
                    if connections > 0:
                        # Score based on connections and sizes
                        score = connections / (len(current_communities[i]) + len(current_communities[j]))
                        community_pairs.append((i, j, score))

            if not community_pairs:
                break

            # Sort pairs by score
            community_pairs.sort(key=lambda x: x[2], reverse=True)

            # Try top pairs
            for i, j, score in community_pairs[:10]:  # Try top 10 pairs
                merged = current_communities[i] | current_communities[j]
                temp_communities = [c for idx, c in enumerate(current_communities)
                                    if idx not in (i, j)] + [merged]

                quality = self.distance_quality(temp_communities, self.shortest_paths)

                # Accept merge if quality doesn't decrease too much
                if quality >= current_quality * 0.99:  # Allow up to 1% quality decrease
                    best_quality = quality
                    current_communities = temp_communities
                    improved = True
                    print(f"Iteration {iteration}: Quality: {best_quality: .4f}, "
                          f"Communities: {len(current_communities)}")
                    break

            if not improved:
                # If no good pairs found, try merging smallest communities
                if len(current_communities) > min_communities:
                    sizes = [(i, len(c)) for i, c in enumerate(current_communities)]
                    sizes.sort(key=lambda x: x[1])
                    i, j = sizes[0][0], sizes[1][0]
                    merged = current_communities[i] | current_communities[j]
                    current_communities = [c for idx, c in enumerate(current_communities)
                                           if idx not in (i, j)] + [merged]
                    best_quality = self.distance_quality(current_communities, self.shortest_paths)
                    print(f"Forced merge of smallest communities: {len(current_communities)} remaining")

            # Print progress periodically
            if iteration % 10 == 0:
                print(f"Progress: {len(current_communities)} communities remaining")

        print(f"\nCommunity detection completed after {iteration} iterations")
        print(f"Final quality score: {best_quality: .4f}")
        print(f"Final number of communities: {len(current_communities)}")
        return current_communities

    def detect_communities_distance_quality(self) -> List[Set[int]]:
        """Main method for distance quality community detection"""
        communities = self.find_communities()
        quality = self.distance_quality(communities, self.shortest_paths)
        print(f"\nDistance Quality Results: ")
        print(f"Found {len(communities)} communities")
        print(f"Quality score: {quality: .4f}")
        self.create_gexf(self.DG, communities, "distance_communities.gexf")
        return communities

    def detect_communities_modularity(self) -> List[Set[int]]:
        """Detect communities using modularity for comparison"""
        print("\nDetecting communities using Modularity...")
        partition = community.best_partition(self.G)
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)
        communities = list(communities.values())
        print(f"Modularity Results: ")
        print(f"Found {len(communities)} communities")
        print(f"Modularity score: {community.modularity(partition, self.G): .4f}")
        self.create_gexf(self.DG, communities, "modularity_communities.gexf")
        return communities

    def create_gexf(self, G: nx.Graph, communities: List[Set[int]], filename: str):
        """Export network with community and department information"""
        H = G.copy()
        for idx, community in enumerate(communities):
            for node in community:
                H.nodes[node]['community_id'] = idx
                H.nodes[node]['department_id'] = self.dept_labels.get(node, -1)
        print(f"\nExporting to {filename}")
        nx.write_gexf(H, filename)


def main():
    start_time = time.time()

    detector = CommunityDetector.from_files(
        r'C:\Users\HP\Desktop\NetworkTheoryProject\data\email-Eu-core.txt',
        r'C:\Users\HP\Desktop\NetworkTheoryProject\data\email-Eu-core-department-labels.txt'
    )

    print("\nStarting community detection analysis....")
    detector.detect_communities_modularity()
    detector.detect_communities_distance_quality()

    execution_time = time.time() - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print(f"\nAnalysis complete!")
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
    print("Results exported to:")
    print("- modularity_communities.gexf")
    print("- distance_communities.gexf")


if __name__ == "__main__":
    main()
