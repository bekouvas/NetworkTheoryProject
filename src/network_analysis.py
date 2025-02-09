import networkx as nx
import time
from collections import defaultdict
import community.community_louvain as community
from typing import List, Set, Dict
from itertools import combinations
import math
import numpy as np


class CommunityDetector:
    def __init__(self, network: nx.DiGraph, dept_labels: Dict[int, int]):
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
        print(f"Loading network from {network_file}")
        network = nx.read_edgelist(network_file, create_using=nx.DiGraph(), nodetype=int)
        print(f"Loading department labels from {dept_file}")
        dept_labels = {}
        with open(dept_file, 'r') as f:
            for line in f:
                node, dept = map(int, line.strip().split())
                dept_labels[node] = dept
        return cls(network, dept_labels)

    def compute_shortest_paths(self):
        print("\nComputing shortest paths (sampling approach)...")
        nodes = list(self.DG.nodes())
        sample_size = min(1005, len(nodes))
        sample_nodes = set(nodes[:sample_size])
        paths = {}
        for node in sample_nodes:
            paths[node] = dict(nx.single_source_shortest_path_length(self.DG, node))
        return paths

    def calculate_community_conductance(self, community_nodes: Set[int]) -> float:
        """Calculate conductance for a community"""
        internal_edges = 0
        external_edges = 0
        for node in community_nodes:
            for neighbor in self.DG[node]:
                if neighbor in community_nodes:
                    internal_edges += 1
                else:
                    external_edges += 1
        total_edges = internal_edges + external_edges
        return external_edges / total_edges if total_edges > 0 else 1.0

    def distance_quality(self, communities: List[Set[int]], sample_paths: Dict) -> float:
        if not communities:
            return float('-inf')

        quality = 0.0
        total_pairs = 0
        sample_nodes = set(sample_paths.keys())
        avg_community_size = self.n / len(communities)

        for cluster in communities:
            cluster_sample = cluster.intersection(sample_nodes)
            if len(cluster_sample) < 2:
                continue

            # Calculate internal density
            internal_edges = sum(1 for i, j in combinations(cluster, 2)
                                 if j in self.DG[i] or i in self.DG[j])
            possible_edges = len(cluster) * (len(cluster) - 1) / 2
            density = internal_edges / possible_edges if possible_edges > 0 else 0

            # Calculate conductance
            conductance = self.calculate_community_conductance(cluster)

            # More lenient size penalty
            size_ratio = len(cluster) / avg_community_size
            size_penalty = math.log(size_ratio + 1) + 1  # More lenient size penalty

            # Calculate quality contribution for this community
            cluster_quality = 0
            cluster_pairs = 0

            for i, j in combinations(cluster_sample, 2):
                if j not in sample_paths[i]:
                    continue
                actual_dist = sample_paths[i][j]

                # Modified quality calculation to encourage merging
                path_quality = (self.diameter - actual_dist) / self.diameter  # Normalized path quality
                density_factor = (density + 0.1) * (1.2 - conductance)  # More lenient density factor
                size_factor = 1 / size_penalty

                cluster_quality += path_quality * density_factor * size_factor
                cluster_pairs += 1

            if cluster_pairs > 0:
                quality += cluster_quality
                total_pairs += cluster_pairs

        # Add bonus for having fewer communities (encourage merging)
        community_factor = math.exp(-len(communities) / 100)  # Bonus for fewer communities
        return (quality / max(1, total_pairs)) * (1 + community_factor)

    def find_communities(self) -> List[Set[int]]:
        if self.shortest_paths is None:
            self.shortest_paths = self.compute_shortest_paths()

        print("\nFinding communities...")
        nodes = list(self.DG.nodes())
        current_communities = [{node} for node in nodes]
        best_quality = self.distance_quality(current_communities, self.shortest_paths)
        print(f"Initial quality: {best_quality: .4f}")

        iteration = 0
        no_improvement_count = 0
        quality_history = [best_quality]

        while len(current_communities) > 2:
            iteration += 1
            improved = False
            current_quality = best_quality
            # Get all pairs of communities and their connection strengths
            community_pairs = []
            for i in range(len(current_communities)):
                for j in range(i + 1, len(current_communities)):
                    # Count connections between communities
                    connections = sum(1 for n1 in current_communities[i]
                                      for n2 in current_communities[j] if n2 in self.DG[n1])
                    if connections > 0:
                        size_i = len(current_communities[i])
                        size_j = len(current_communities[j])
                        density_i = self.calculate_density(current_communities[i])
                        density_j = self.calculate_density(current_communities[j])

                        # Modified scoring to encourage merging
                        connection_density = connections / (size_i * size_j)
                        size_balance = min(size_i, size_j) / max(size_i, size_j)
                        density_similarity = 1 - abs(density_i - density_j)

                        score = connection_density * (size_balance + 0.2) * (density_similarity + 0.1)
                        community_pairs.append((i, j, score))

            if not community_pairs:
                break

            community_pairs.sort(key=lambda x: x[2], reverse=True)
            # Try more pairs each iteration
            for i, j, score in community_pairs[:10]:  # Try more pairs
                merged = current_communities[i] | current_communities[j]

                # More lenient size limit
                if len(merged) > self.n / 2:  # Changed from n/3 to n/2
                    continue

                temp_communities = [c for idx, c in enumerate(current_communities)
                                    if idx not in (i, j)] + [merged]

                quality = self.distance_quality(temp_communities, self.shortest_paths)

                # More lenient threshold for accepting merges
                size_ratio = len(merged) / self.n
                quality_std = np.std(quality_history[-10:]) if len(quality_history) >= 10 else 0
                threshold = 0.95 - (0.05 * size_ratio) - (0.01 * quality_std)  # More lenient threshold

                if quality >= current_quality * threshold:
                    best_quality = quality
                    current_communities = temp_communities
                    quality_history.append(quality)
                    improved = True
                    print(f"Iteration {iteration}: Quality: {best_quality: .4f}, "
                          f"Communities: {len(current_communities)}")
                    break

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 10:  # More iterations before stopping
                    # Try forced merge of smallest communities before stopping
                    if len(current_communities) > 50:  # Don't force merges if communities are too few
                        sizes = [(i, len(c)) for i, c in enumerate(current_communities)]
                        sizes.sort(key=lambda x: x[1])
                        i, j = sizes[0][0], sizes[1][0]
                        merged = current_communities[i] | current_communities[j]
                        current_communities = [c for idx, c in enumerate(current_communities)
                                               if idx not in (i, j)] + [merged]
                        best_quality = self.distance_quality(current_communities, self.shortest_paths)
                        quality_history.append(best_quality)
                        no_improvement_count = 0
                        print(f"Forced merge of smallest communities: {len(current_communities)} remaining")
                    else:
                        break

            if iteration % 10 == 0:
                print(f"Progress: {len(current_communities)} communities remaining")

        print(f"\nCommunity detection completed after {iteration} iterations")
        print(f"Final quality score: {best_quality: .4f}")
        print(f"Final number of communities: {len(current_communities)}")
        return current_communities

    def calculate_density(self, nodes: Set[int]) -> float:
        if len(nodes) < 2:
            return 0.0
        subgraph = self.DG.subgraph(nodes)
        possible_edges = len(nodes) * (len(nodes) - 1)
        return subgraph.number_of_edges() / possible_edges if possible_edges > 0 else 0.0

    def detect_communities_distance_quality(self) -> List[Set[int]]:
        communities = self.find_communities()
        quality = self.distance_quality(communities, self.shortest_paths)
        print(f"\nDistance Quality Results: ")
        print(f"Found {len(communities)} communities")
        print(f"Quality score: {quality: .4f}")
        self.create_gexf(self.DG, communities, "distance_communities.gexf")
        return communities

    def detect_communities_modularity(self) -> List[Set[int]]:
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
