class UnionFind:
    parent: list[int]
    rank: list[int]

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size  # tracks the "depth" of each tree

    def find(self, index: int) -> int:
        """Find the root (with path compression)."""
        if self.parent[index] != index:
            # Recursively find the root and relink directly (path compression)
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, a: int, b: int) -> bool:
        """Merge two sets (with union by rank)."""
        a_root = self.find(a)
        b_root = self.find(b)
        if a_root == b_root:
            return False

        # Attach the lower-rank root under the higher-rank root
        if self.rank[a_root] < self.rank[b_root]:
            self.parent[a_root] = b_root
        elif self.rank[a_root] > self.rank[b_root]:
            self.parent[b_root] = a_root
        else:
            self.parent[b_root] = a_root
            self.rank[a_root] += 1  # same rank -> the merged tree gets one level deeper

        return True
