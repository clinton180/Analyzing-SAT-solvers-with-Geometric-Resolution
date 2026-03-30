""" Multilevel CDS for gap boxes. """
from __future__ import annotations
import os, sys

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Tuple, Dict, Optional, Iterable, Set, Sequence
import sys
sys.setrecursionlimit(200000) # for deep recursion in CDS queries


# Dyadic Trie (1D)
class DyadicTrieNode:
    """
    A node in a 1D dyadic trie (binary prefix tree).
    Used as a building block for the multilevel CDS.
    """
    __slots__ = ("child0", "child1", "is_storage", "next_dim_root", "box_terminal", "payloads")
    def __init__(self):
        self.child0: Optional[DyadicTrieNode] = None
        self.child1: Optional[DyadicTrieNode] = None
        # is_storage: this prefix corresponds to a stored dyadic segment on this dimension
        self.is_storage: bool = False
        # next_dim_root: root of the trie for the next dimension (for CDS)
        self.next_dim_root: Optional[DyadicTrie] = None
        # box_terminal: True if this node is the last dimension node of a stored box
        self.box_terminal: bool = False
        self.payloads: list[Tuple[str, ...]] = []  # store full boxes ending here


class DyadicTrie:
    """
    1D dyadic trie for binary prefixes.
    """
    def __init__(self):
        self.root = DyadicTrieNode()

    def insert_prefix(self, prefix: str) -> DyadicTrieNode:
        """
        Insert a binary prefix (e.g. "10", "0", "") into the trie.

        Returns:
            The node corresponding to that prefix.
        """
        node = self.root
        for bit in prefix:
            if bit == '0':
                if node.child0 is None:
                    node.child0 = DyadicTrieNode()
                node = node.child0
            else:
                if node.child1 is None:
                    node.child1 = DyadicTrieNode()
                node = node.child1
        node.is_storage = True
        return node

    def ancestors_of(self, prefix: str) -> Iterable[DyadicTrieNode]:
        """
        Yield all storage nodes on the path describing 'prefix'.
        """
        node = self.root
        if node.is_storage:
            yield node
        for bit in prefix:
            if bit == '0':
                node = node.child0
            else:
                node = node.child1
            if node is None:
                break
            if node.is_storage:
                yield node

    def ancestors_of_with_depth(self, prefix: str) -> Iterable[Tuple[DyadicTrieNode, int]]:
        """
        Yield (storage_node, depth) along the path describing 'prefix'.
        depth = length of the prefix represented by the yielded node.
        """
        node = self.root
        depth = 0
        if node.is_storage:
            yield node, depth
        for bit in prefix:
            depth += 1
            if bit == '0':
                node = node.child0
            else:
                node = node.child1
            if node is None:
                break
            if node.is_storage:
                yield node, depth


    def one_step_descendant_storage(self, prefix: str) -> Iterable[DyadicTrieNode]:
        """
        Yield storage nodes on paths (prefix+'0') and (prefix+'1') that are
        STRICTLY deeper than len(prefix). These correspond to near-miss
        candidates that would become ancestors after one split on this dim.
        """
        L = len(prefix)
        seen_ids: Set[int] = set()
        for child in (prefix + "0", prefix + "1"):
            for node, depth in self.ancestors_of_with_depth(child):
                if depth <= L:
                    continue
                nid = id(node)
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                yield node

# Multilevel CDS (stores GAP boxes)
class MultilevelCDS:
    """
    Multilevel dyadic tree (Constraint Data Structure) that stores dyadic GAP boxes.
    """

    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        # root trie for dimension 0
        self._root_trie = DyadicTrie()

    # Insertion
    def insert_box_prefixes(self, prefix_tuple: List[str]) -> None:
        """
        Insert a dyadic GAP box specified by prefixes per dimension.

        Args:
            prefix_tuple: list[str] of length num_dims, e.g. ["0", "10", ""].
        """
        assert len(prefix_tuple) == self.num_dims, "prefix_tuple dim mismatch"

        # Dimension 0
        node = self._root_trie.insert_prefix(prefix_tuple[0])

        # Dimensions 1..n-1
        for dim in range(1, self.num_dims):
            if node.next_dim_root is None:
                node.next_dim_root = DyadicTrie()
            trie = node.next_dim_root
            node = trie.insert_prefix(prefix_tuple[dim])

        # Mark the last node as a terminal box node.
        node.box_terminal = True
        node.payloads.append(tuple(prefix_tuple))  # <--- critical

    # ----------------------------
    # Query: boxes containing a given prefix-box
    # ----------------------------
    def boxes_containing_prefix_box(self, prefix_tuple: List[str]) -> List[Tuple[str, ...]]:
        """
        Given a dyadic search box represented by prefixes (one per dim),
        return all stored GAP boxes whose prefixes are ancestors (containment)
        in every dimension.

        Returns:
            A list of "chains" of DyadicTrieNodes, one per dimension.
            For many use cases, you only need to check whether the list
            is non-empty (i.e., there exists at least one containing box).
        """
        assert len(prefix_tuple) == self.num_dims
        out: List[Tuple[str, ...]] = []

        for node0 in self._root_trie.ancestors_of(prefix_tuple[0]):
            self._boxes_containing_suffix(prefix_tuple, dim=1, node=node0, out=out)
        return out

    def _boxes_containing_suffix(self,
                                 prefix_tuple: List[str],
                                 dim: int,
                                 node: DyadicTrieNode,
                                 out: List[Tuple[str, ...]]) -> None:
        """
        Recursive helper: for dimension 'dim', having chosen ancestor nodes
        for dimensions [0..dim-1] in acc.

        At dim == num_dims, we record the chain if the last node is terminal.
        """
        if dim == self.num_dims:
            # We have one storage node per dimension.
            if node.box_terminal:
                out.extend(node.payloads)  # <--- return actual boxes
            return

        trie = node.next_dim_root
        if trie is None:
            return

        for n_k in trie.ancestors_of(prefix_tuple[dim]):
            self._boxes_containing_suffix(prefix_tuple, dim + 1, n_k, out)

    def _any_box_containing_suffix_iterative(
        self,
        prefix_tuple: List[str],
        start_dim: int,
        start_node: DyadicTrieNode,
    ) -> Optional[Tuple[str, ...]]:
        stack = [(start_dim, start_node)]

        while stack:
            dim, node = stack.pop()

            if dim == self.num_dims:
                if node.box_terminal:
                    return node.payloads[0]
                continue

            trie = node.next_dim_root
            if trie is None:
                continue

            candidates = list(trie.ancestors_of(prefix_tuple[dim]))

            # reverse so stack preserves same left-to-right traversal order
            for n_k in reversed(candidates):
                stack.append((dim + 1, n_k))

        return None

    def any_box_containing_prefix_box(
        self,
        prefix_tuple: List[str],
        collect_active: bool = False,
    ):
        """
        If collect_active=False:
            Return ANY stored GAP box whose prefixes are ancestors
            of prefix_tuple in every dimension. Stops at first witness.

        If collect_active=True:
            Return (active_boxes, witness_or_None)
            where active_boxes are all terminal boxes visited
            before termination.
        """
        assert len(prefix_tuple) == self.num_dims

        if not collect_active:
            for node0 in self._root_trie.ancestors_of(prefix_tuple[0]):
                hit = self._any_box_containing_suffix_iterative(prefix_tuple, start_dim=1, start_node=node0)
                if hit is not None:
                    return hit, None
            return None, None

        active: List[Tuple[str, ...]] = []
        seen: Set[Tuple[str, ...]] = set()

        for node0 in self._root_trie.ancestors_of(prefix_tuple[0]):
            hit = self._any_box_containing_suffix_active(
                prefix_tuple,
                dim=1,
                node=node0,
                active=active,
                seen=seen,
            )
            if hit is not None:
                return hit, active
        return None, active
    
    
    def _any_box_containing_suffix_active(
        self,
        prefix_tuple: List[str],
        dim: int,
        node: DyadicTrieNode,
        active: List[Tuple[str, ...]],
        seen: Set[Tuple[str, ...]],
    ) -> Optional[Tuple[str, ...]]:
        """
        Short-circuiting recursive helper that:
        - returns first terminal box found
        - collects all terminal boxes visited before termination
        """
        if dim == self.num_dims:
            if node.box_terminal:
                for box in node.payloads:
                    if box not in seen:
                        active.append(box)
                        seen.add(box)
                # return exactly one witness (preserves original semantics)
                return node.payloads[0]
            return None

        trie = node.next_dim_root
        if trie is None:
            return None

        for n_k in trie.ancestors_of(prefix_tuple[dim]):
            hit = self._any_box_containing_suffix_active(
                prefix_tuple,
                dim + 1,
                n_k,
                active,
                seen,
            )
            if hit is not None:
                return hit
        return None
    
    def any_cover_or_nearmiss_prefix_box(
        self,
        prefix_tuple: List[str],
        widths: Sequence[int],
        collect_active: bool = False,
        cap: int = 512,
    ):
        assert len(prefix_tuple) == self.num_dims
        assert len(widths) == self.num_dims

        if not collect_active:
            hit = self._any_cover_or_nearmiss_suffix(
                prefix_tuple, widths, dim=0, trie=self._root_trie,
                mismatch_used=False, active=None, seen=None, cap=0
            )
            return hit, None

        active: List[Tuple[str, ...]] = []
        seen: Set[Tuple[str, ...]] = set()

        hit = self._any_cover_or_nearmiss_suffix(
            prefix_tuple, widths, dim=0, trie=self._root_trie,
            mismatch_used=False, active=active, seen=seen, cap=cap
        )
        return hit, active


    def _any_cover_or_nearmiss_suffix(
        self,
        prefix_tuple: List[str],
        widths: Sequence[int],
        dim: int,
        trie: DyadicTrie,
        mismatch_used: bool,
        active: Optional[List[Tuple[str, ...]]],
        seen: Optional[Set[Tuple[str, ...]]],
        cap: int,
    ) -> Optional[Tuple[str, ...]]:
        """
        Recursive traversal over dimensions.
        - dim is the current dimension index.
        - trie is the DyadicTrie for this dimension at the current multilevel node.
        - mismatch_used indicates whether we've already spent the one allowed mismatch.
        """
        p = prefix_tuple[dim]

        # 1) exact candidates: ancestors of p
        for node in trie.ancestors_of(p):
            hit = self._recurse_next_dim(
                prefix_tuple, widths, dim, node,
                mismatch_used, active, seen, cap
            )
            if hit is not None:
                return hit

        # 2) near-miss candidates: spend mismatch on this dim (if not used yet and dim is thick)
        if (not mismatch_used) and (len(p) < widths[dim]):
            for node in trie.one_step_descendant_storage(p):
                hit = self._recurse_next_dim(
                    prefix_tuple, widths, dim, node,
                    True, active, seen, cap
                )
                if hit is not None:
                    return hit

        return None


    def _recurse_next_dim(
        self,
        prefix_tuple: List[str],
        widths: Sequence[int],
        dim: int,
        node: DyadicTrieNode,
        mismatch_used: bool,
        active: Optional[List[Tuple[str, ...]]],
        seen: Optional[Set[Tuple[str, ...]]],
        cap: int,
    ) -> Optional[Tuple[str, ...]]:
        """
        Having selected a storage node at dimension `dim`,
        either:
        - record terminals if dim is last
        - recurse to next dimension's trie otherwise
        """
        if dim + 1 == self.num_dims:
            if node.box_terminal:
                # collect visited terminals
                if active is not None and seen is not None and len(active) < cap:
                    for box in node.payloads:
                        if box not in seen:
                            if len(active) >= cap:
                                break
                            active.append(box)
                            seen.add(box)

                # witness only if NO mismatch was used anywhere (bad_count=0)
                if not mismatch_used:
                    return node.payloads[0]
            return None

        next_trie = node.next_dim_root
        if next_trie is None:
            return None

        return self._any_cover_or_nearmiss_suffix(
            prefix_tuple, widths, dim + 1, next_trie,
            mismatch_used, active, seen, cap
        )