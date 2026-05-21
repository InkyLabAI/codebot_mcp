"""
Tests for the call graph ordering utility.

Tests cover:
- Local-only algorithm (no repository_id)
  - Entry point based chaining (each entry point creates its own chain)
  - Shared callees (functions can appear in multiple chains)
  - Direct call chaining (A calls B, both in results)
  - Missing intermediate functions (A calls B calls C, B not in results)
  - Multiple call branches
  - Isolated functions
- PCST algorithm (with repository_id)
  - Bridge node discovery
  - Weight-based path selection
  - Cutoff behavior
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
import uuid

from codebot_mcp.utils.call_graph import (
    order_functions_by_call_graph,
    order_functions_local,
    find_steiner_bridges,
    get_call_chain_stats,
    _merge_overlapping_chains,
    _topological_sort_chain,
    _chain_score,
    _dfs_preorder_sort,
)


def make_func(func_id: str, calls: list = None, similarity: float = 0.5):
    """Helper to create mock FunctionSearchResult."""
    mock = MagicMock()
    mock.function = MagicMock()
    mock.function.function_id = func_id
    mock.function.calls = calls or []
    mock.similarity = similarity
    mock.is_bridge = False  # Default to not a bridge
    return mock


# =============================================================================
# Tests for Local-only Algorithm (order_functions_local)
# =============================================================================

class TestEntryPointChaining:
    """Tests for entry point based chaining - each entry point creates its own chain."""

    def test_two_entry_points_shared_callees(self):
        """Test A calls [B, C] and D calls [B, C] -> two separate chains.

        Without merging, each entry point keeps its own chain with shared callees.
        """
        A = make_func('A', calls=['B', 'C'])
        B = make_func('B', calls=[])
        C = make_func('C', calls=[])
        D = make_func('D', calls=['B', 'C'])

        result = order_functions_local([A, B, C, D])

        # Each entry point creates its own chain containing shared callees
        chain_id_sets = [{f.function.function_id for f in c} for c in result]
        assert {'A', 'B', 'C'} in chain_id_sets
        assert {'D', 'B', 'C'} in chain_id_sets

    def test_three_entry_points_shared_callee(self):
        """Test A, B, C all call D -> 3 chains [A, D], [B, D], [C, D]."""
        A = make_func('A', calls=['D'])
        B = make_func('B', calls=['D'])
        C = make_func('C', calls=['D'])
        D = make_func('D', calls=[])

        result = order_functions_local([A, B, C, D])

        # Should have 3 chains
        assert len(result) == 3

        chain_ids = [[f.function.function_id for f in chain] for chain in result]

        # Each chain should have 2 elements
        for chain in chain_ids:
            assert len(chain) == 2
            assert chain[1] == 'D'  # D should be second in each chain

        # Verify each entry point has its chain
        entry_points = {chain[0] for chain in chain_ids}
        assert entry_points == {'A', 'B', 'C'}

    def test_single_entry_point_branching(self):
        """Test A calls B, B calls [C, D], C and D call E -> 1 chain with all."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=['C', 'D'])
        C = make_func('C', calls=['E'])
        D = make_func('D', calls=['E'])
        E = make_func('E', calls=[])

        result = order_functions_local([A, B, C, D, E])

        # Should have 1 chain (only A is entry point)
        assert len(result) == 1

        chain_ids = [f.function.function_id for f in result[0]]

        # All 5 functions should be in the chain
        assert set(chain_ids) == {'A', 'B', 'C', 'D', 'E'}
        assert chain_ids[0] == 'A'  # A should be first


class TestDirectCallChaining:
    """Tests for direct call relationships."""

    def test_simple_chain(self):
        """Test A calls B, both in results -> chain [A, B]."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=[])

        result = order_functions_local([A, B])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A', 'B']

    def test_three_function_chain(self):
        """Test A calls B, B calls C, all in results -> chain [A, B, C]."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=['C'])
        C = make_func('C', calls=[])

        result = order_functions_local([A, B, C])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A', 'B', 'C']

    def test_multiple_direct_calls(self):
        """Test A calls B and C directly, all in results."""
        A = make_func('A', calls=['B', 'C'])
        B = make_func('B', calls=[])
        C = make_func('C', calls=[])

        result = order_functions_local([A, B, C])

        assert len(result) == 1
        chain_ids = [f.function.function_id for f in result[0]]
        assert chain_ids[0] == 'A'  # A is the caller, should be first
        assert set(chain_ids) == {'A', 'B', 'C'}


class TestMissingIntermediateFunctions:
    """Tests for chains with missing intermediate functions."""

    def test_missing_middle_function(self):
        """Test A calls B, B calls C, only A and C in results -> SEPARATE chains."""
        A = make_func('A', calls=['B'])  # A calls B, but B is not in results
        C = make_func('C', calls=[])

        result = order_functions_local([A, C])

        # A and C should be SEPARATE because there's no direct edge between them
        assert len(result) == 2
        chain_ids = [[f.function.function_id for f in chain] for chain in result]
        assert ['A'] in chain_ids
        assert ['C'] in chain_ids

    def test_missing_first_in_chain(self):
        """Test A calls B, B calls C, only B and C in results -> chain [B, C]."""
        B = make_func('B', calls=['C'])
        C = make_func('C', calls=[])

        result = order_functions_local([B, C])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['B', 'C']

    def test_missing_last_in_chain(self):
        """Test A calls B, B calls C, only A and B in results -> chain [A, B]."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=['C'])  # C is not in results

        result = order_functions_local([A, B])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A', 'B']


class TestIsolatedFunctions:
    """Tests for functions with no call relationships."""

    def test_single_isolated_function(self):
        """Test single function with no calls."""
        A = make_func('A', calls=[])

        result = order_functions_local([A])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A']

    def test_multiple_isolated_functions(self):
        """Test multiple functions with no connections."""
        A = make_func('A', calls=[])
        B = make_func('B', calls=[])
        C = make_func('C', calls=[])

        result = order_functions_local([A, B, C])

        # Each should be its own chain (each is an entry point)
        assert len(result) == 3
        chain_ids = [[f.function.function_id for f in chain] for chain in result]
        assert ['A'] in chain_ids
        assert ['B'] in chain_ids
        assert ['C'] in chain_ids

    def test_mixed_connected_and_isolated(self):
        """Test mix of connected chain and isolated functions."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=[])
        C = make_func('C', calls=[])  # Isolated

        result = order_functions_local([A, B, C])

        # Should have 2 chains: [A, B] and [C]
        assert len(result) == 2

        chain_ids = [[f.function.function_id for f in chain] for chain in result]

        # Longer chain first
        assert chain_ids[0] == ['A', 'B']
        assert chain_ids[1] == ['C']


class TestComplexScenarios:
    """Tests for complex call graph scenarios."""

    def test_diamond_pattern(self):
        """Test A calls B and C, both B and C call D."""
        A = make_func('A', calls=['B', 'C'])
        B = make_func('B', calls=['D'])
        C = make_func('C', calls=['D'])
        D = make_func('D', calls=[])

        result = order_functions_local([A, B, C, D])

        # Single entry point (A), so one chain
        assert len(result) == 1
        chain_ids = [f.function.function_id for f in result[0]]
        assert chain_ids[0] == 'A'
        assert set(chain_ids) == {'A', 'B', 'C', 'D'}

    def test_two_separate_entry_points_with_chains(self):
        """Test two disconnected call chains from different entry points."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=[])
        C = make_func('C', calls=['D'])
        D = make_func('D', calls=[])

        result = order_functions_local([A, B, C, D])

        # Two entry points (A and C), so two chains
        assert len(result) == 2

        chain_ids = [[f.function.function_id for f in chain] for chain in result]

        # Each chain should have its entry point first
        chain1 = next(c for c in chain_ids if 'A' in c)
        chain2 = next(c for c in chain_ids if 'C' in c)

        assert chain1 == ['A', 'B']
        assert chain2 == ['C', 'D']

    def test_false_edge_from_same_callee_name(self):
        """
        Test that functions aren't connected just because they call
        a function with the same name that's not in results.

        A calls X.process (X not in results)
        B calls Y.process (Y not in results)
        A and B should NOT be connected.
        """
        A = make_func('A', calls=['X.process'])
        B = make_func('B', calls=['Y.process'])

        result = order_functions_local([A, B])

        # A and B should be separate (no connection through missing functions)
        assert len(result) == 2

    def test_convergent_chains(self):
        """Test A->B->E and C->D->E converge at E."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=['E'])
        C = make_func('C', calls=['D'])
        D = make_func('D', calls=['E'])
        E = make_func('E', calls=[])

        result = order_functions_local([A, B, C, D, E])

        # Two entry points (A and C), so two chains
        assert len(result) == 2

        chain_ids = [[f.function.function_id for f in chain] for chain in result]

        # A's chain should be [A, B, E]
        a_chain = next(c for c in chain_ids if c[0] == 'A')
        assert a_chain == ['A', 'B', 'E']

        # C's chain should be [C, D, E]
        c_chain = next(c for c in chain_ids if c[0] == 'C')
        assert c_chain == ['C', 'D', 'E']

        # E appears in both chains
        assert 'E' in a_chain
        assert 'E' in c_chain


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self):
        """Test with empty function list."""
        result = order_functions_local([])
        assert result == []

    def test_self_recursive_call(self):
        """Test function that calls itself."""
        A = make_func('A', calls=['A'])

        result = order_functions_local([A])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A']

    def test_mutual_recursion(self):
        """Test A calls B, B calls A."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=['A'])

        result = order_functions_local([A, B])

        # No true entry point (cycle), but algorithm handles it
        assert len(result) == 1
        chain_ids = set(f.function.function_id for f in result[0])
        assert chain_ids == {'A', 'B'}

    def test_cycle_with_entry_point(self):
        """Test A calls B, B calls C, C calls B (cycle), A is entry point."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=['C'])
        C = make_func('C', calls=['B'])  # Cycle back to B

        result = order_functions_local([A, B, C])

        assert len(result) == 1
        chain_ids = [f.function.function_id for f in result[0]]
        assert chain_ids[0] == 'A'  # A should be first (entry point)
        assert set(chain_ids) == {'A', 'B', 'C'}


class TestChainOrdering:
    """Tests for chain ordering (longest first)."""

    def test_chains_sorted_by_length(self):
        """Test that chains are sorted by length, longest first."""
        # Create chains of different lengths
        A = make_func('A', calls=['B', 'C', 'D'])  # Chain of 4
        B = make_func('B', calls=[])
        C = make_func('C', calls=[])
        D = make_func('D', calls=[])
        E = make_func('E', calls=['F'])  # Chain of 2
        F = make_func('F', calls=[])
        G = make_func('G', calls=[])  # Chain of 1

        result = order_functions_local([A, B, C, D, E, F, G])

        # Should have 3 chains: [A,B,C,D], [E,F], [G]
        assert len(result) == 3

        # Verify sorted by length
        assert len(result[0]) == 4  # A's chain
        assert len(result[1]) == 2  # E's chain
        assert len(result[2]) == 1  # G's chain

    def test_same_length_chains_by_similarity(self):
        """Test that chains of same length are sorted by similarity."""
        A = make_func('A', calls=['B'], similarity=0.9)
        B = make_func('B', calls=[], similarity=0.8)
        C = make_func('C', calls=['D'], similarity=0.7)
        D = make_func('D', calls=[], similarity=0.6)

        # A has higher similarity than C
        result = order_functions_local([A, B, C, D])

        chain_ids = [[f.function.function_id for f in chain] for chain in result]

        # Both chains have length 2, A should come first (higher similarity)
        assert chain_ids[0][0] == 'A'
        assert chain_ids[1][0] == 'C'


# =============================================================================
# Tests for PCST Algorithm
# =============================================================================

class TestSteinerBridges:
    """Tests for the Steiner bridge finding algorithm."""

    def _single_node_chains(self, terminal_ids):
        """Helper: each terminal in its own single-node chain (disconnected)."""
        return [[make_func(tid)] for tid in terminal_ids]

    def test_finds_bridge_between_terminals(self):
        """Test finding a bridge node between two terminals."""
        import networkx as nx

        G = nx.DiGraph()
        # A -> B -> C where A and C are terminals
        G.add_edge('A', 'B', weight=1.0)
        G.add_edge('B', 'C', weight=1.0)

        terminals = {'A', 'C'}
        local_chains = self._single_node_chains(terminals)
        bridges = find_steiner_bridges(G, terminals, local_chains, cutoff=5.0)

        assert bridges == {'B'}

    def test_no_bridge_when_cutoff_too_small(self):
        """Test that no bridges are found when cutoff is too small."""
        import networkx as nx

        G = nx.DiGraph()
        # A -> B -> C with total weight 4.0
        G.add_edge('A', 'B', weight=2.0)
        G.add_edge('B', 'C', weight=2.0)

        terminals = {'A', 'C'}
        local_chains = self._single_node_chains(terminals)
        # Cutoff of 3.0 is less than path weight of 4.0
        bridges = find_steiner_bridges(G, terminals, local_chains, cutoff=3.0)

        assert bridges == set()

    def test_multiple_bridges(self):
        """Test finding multiple bridge nodes."""
        import networkx as nx

        G = nx.DiGraph()
        # A -> B -> C -> D where A and D are terminals
        G.add_edge('A', 'B', weight=1.0)
        G.add_edge('B', 'C', weight=1.0)
        G.add_edge('C', 'D', weight=1.0)

        terminals = {'A', 'D'}
        local_chains = self._single_node_chains(terminals)
        bridges = find_steiner_bridges(G, terminals, local_chains, cutoff=5.0)

        assert bridges == {'B', 'C'}

    def test_prefers_lower_weight_path(self):
        """Test that lower weight path is chosen over higher weight."""
        import networkx as nx

        G = nx.DiGraph()
        # Two paths from A to C:
        # A -> B1 -> C (weight 2.0)
        # A -> B2 -> C (weight 4.0)
        G.add_edge('A', 'B1', weight=1.0)
        G.add_edge('B1', 'C', weight=1.0)
        G.add_edge('A', 'B2', weight=2.0)
        G.add_edge('B2', 'C', weight=2.0)

        terminals = {'A', 'C'}
        local_chains = self._single_node_chains(terminals)
        bridges = find_steiner_bridges(G, terminals, local_chains, cutoff=5.0)

        # Should choose B1 (lower weight path)
        assert bridges == {'B1'}

    def test_terminal_not_in_graph(self):
        """Test handling when a terminal is not in the graph."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge('A', 'B', weight=1.0)

        # C is not in the graph
        terminals = {'A', 'C'}
        local_chains = self._single_node_chains(terminals)
        bridges = find_steiner_bridges(G, terminals, local_chains, cutoff=5.0)

        assert bridges == set()

    def test_no_bridge_between_same_chain_nodes(self):
        """Test that nodes already in the same chain don't get bridged."""
        import networkx as nx

        G = nx.DiGraph()
        # A -> X -> B and A -> Y -> B (two paths)
        # But A and B are already in the same local chain
        G.add_edge('A', 'X', weight=1.0)
        G.add_edge('X', 'B', weight=1.0)
        G.add_edge('A', 'Y', weight=1.0)
        G.add_edge('Y', 'B', weight=1.0)

        terminals = {'A', 'B'}
        # A and B are already connected in one chain
        chain_ab = [make_func('A', calls=['B']), make_func('B')]
        local_chains = [chain_ab]
        bridges = find_steiner_bridges(G, terminals, local_chains, cutoff=5.0)

        # No bridges needed — A and B are already in the same chain
        assert bridges == set()


class TestCallChainStats:
    """Tests for call chain statistics."""

    def test_stats_empty_input(self):
        """Test stats with empty input."""
        stats = get_call_chain_stats([])

        assert stats == {
            'num_chains': 0,
            'chain_sizes': [],
            'total_internal_calls': 0,
            'entry_points': 0,
            'bridge_count': 0
        }

    def test_stats_with_chains(self):
        """Test stats with connected functions."""
        A = make_func('A', calls=['B', 'C'])
        B = make_func('B', calls=[])
        C = make_func('C', calls=[])

        stats = get_call_chain_stats([A, B, C])

        assert stats['num_chains'] == 1
        assert stats['chain_sizes'] == [3]
        assert stats['total_internal_calls'] == 2
        assert stats['entry_points'] == 1
        assert stats['bridge_count'] == 0

    def test_stats_with_bridge_nodes(self):
        """Test stats counting bridge nodes."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=[])
        B.is_bridge = True  # Mark as bridge

        stats = get_call_chain_stats([A, B])

        assert stats['bridge_count'] == 1


# =============================================================================
# Tests for Async Main Function
# =============================================================================

class TestAsyncOrderFunctions:
    """Tests for the async order_functions_by_call_graph function."""

    @pytest.mark.asyncio
    async def test_fallback_to_local_without_repository_id(self):
        """Test that without repository_id, falls back to local algorithm."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=[])

        # No repository_id provided
        result = await order_functions_by_call_graph([A, B])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A', 'B']

    @pytest.mark.asyncio
    async def test_fallback_to_local_without_session(self):
        """Test that without session, falls back to local algorithm."""
        A = make_func('A', calls=['B'])
        B = make_func('B', calls=[])

        # repository_id but no session
        result = await order_functions_by_call_graph(
            [A, B],
            repository_id=uuid.uuid4()
        )

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A', 'B']

    @pytest.mark.asyncio
    async def test_empty_input_async(self):
        """Test async function with empty input."""
        result = await order_functions_by_call_graph([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_function_async(self):
        """Test async function with single function."""
        A = make_func('A', calls=[])

        result = await order_functions_by_call_graph([A])

        assert len(result) == 1
        assert [f.function.function_id for f in result[0]] == ['A']


# =============================================================================
# Tests for _merge_overlapping_chains
# =============================================================================

class TestMergeOverlappingChains:
    """Tests for merging chains that share significant node overlap."""

    def _build_graph_and_map(self, funcs):
        """Helper: build graph adjacency list and result_map from mock functions."""
        from collections import defaultdict
        result_map = {f.function.function_id: f for f in funcs}
        func_ids = set(result_map.keys())
        graph = defaultdict(list)
        for f in funcs:
            caller = f.function.function_id
            for callee in (f.function.calls or []):
                if callee in func_ids:
                    graph[caller].append(callee)
        return graph, result_map

    def test_merge_skipped_when_score_drops(self):
        """Merge is skipped when the union scores lower than originals."""
        # Chain1: [A, B] (scores 0.9, 0.8 -> avg 0.85)
        # Chain2: [B, C] (scores 0.8, 0.1 -> avg 0.45)
        # Union {A, B, C} avg = 0.6 < best original 0.85 -> keep both
        A = make_func('A', calls=['B'], similarity=0.9)
        B = make_func('B', calls=['C'], similarity=0.8)
        C = make_func('C', calls=[], similarity=0.1)

        graph, result_map = self._build_graph_and_map([A, B, C])

        chain1 = [A, B]
        chain2 = [B, C]

        merged = _merge_overlapping_chains([chain1, chain2], graph, result_map)

        # Union scores lower than best original -> keep both
        assert len(merged) == 2

    def test_high_overlap_merge_when_score_improves(self):
        """Chains merge to common core when core scores higher than originals."""
        # Chain1: [A, B, C, D, E] Chain2: [F, B, C, D, E]
        # Core {B, C, D, E} — high scores so core avg >= best original
        A = make_func('A', calls=['B'], similarity=0.1)
        B = make_func('B', calls=['C'], similarity=0.9)
        C = make_func('C', calls=['D'], similarity=0.8)
        D = make_func('D', calls=['E'], similarity=0.7)
        E = make_func('E', calls=[], similarity=0.6)
        F = make_func('F', calls=['B'], similarity=0.1)

        graph, result_map = self._build_graph_and_map([A, B, C, D, E, F])

        chain1 = [A, B, C, D, E]
        chain2 = [F, B, C, D, E]

        merged = _merge_overlapping_chains([chain1, chain2], graph, result_map)

        assert len(merged) == 1
        merged_ids = {f.function.function_id for f in merged[0]}
        assert merged_ids == {'B', 'C', 'D', 'E'}
        ids = [f.function.function_id for f in merged[0]]
        assert ids.index('B') < ids.index('C')
        assert ids.index('C') < ids.index('D')
        assert ids.index('D') < ids.index('E')

    def test_low_overlap_stays_separate(self):
        """Chains with <60% overlap should not be merged."""
        # Chain1: [A, B, C, D, E] (5 nodes)
        # Chain2: [F, G, H, D, E] (5 nodes)
        # Overlap: {D, E} = 2/5 = 0.4 < 0.6 -> no merge
        A = make_func('A', calls=['B'], similarity=0.9)
        B = make_func('B', calls=['C'], similarity=0.5)
        C = make_func('C', calls=['D'], similarity=0.4)
        D = make_func('D', calls=['E'], similarity=0.3)
        E = make_func('E', calls=[], similarity=0.2)
        F = make_func('F', calls=['G'], similarity=0.8)
        G = make_func('G', calls=['H'], similarity=0.6)
        H = make_func('H', calls=['D'], similarity=0.5)

        graph, result_map = self._build_graph_and_map([A, B, C, D, E, F, G, H])

        chain1 = [A, B, C, D, E]
        chain2 = [F, G, H, D, E]

        merged = _merge_overlapping_chains([chain1, chain2], graph, result_map)

        assert len(merged) == 2

    def test_single_node_chains_skip_merge(self):
        """Single-node chains should never be merged with each other."""
        A = make_func('A', calls=[], similarity=0.9)
        B = make_func('B', calls=[], similarity=0.8)

        graph, result_map = self._build_graph_and_map([A, B])

        chain1 = [A]
        chain2 = [B]

        merged = _merge_overlapping_chains([chain1, chain2], graph, result_map)

        # Both singletons should remain separate
        assert len(merged) == 2

    def test_transitive_merge(self):
        """A overlaps B, B overlaps C -> transitive merge extracts common core."""
        # Chain A: {n1, n2, n3, n4}
        # Chain B: {n2, n3, n4, n5}  overlap with A: 3/4=0.75
        # Chain C: {n3, n4, n5, n6}  overlap with B: 3/4=0.75
        # Core (nodes in 2+ chains): {n2, n3, n4, n5}
        # Give core high scores, boundary nodes low scores
        n1 = make_func('n1', calls=['n2'], similarity=0.1)
        n2 = make_func('n2', calls=['n3'], similarity=0.9)
        n3 = make_func('n3', calls=['n4'], similarity=0.8)
        n4 = make_func('n4', calls=['n5'], similarity=0.7)
        n5 = make_func('n5', calls=['n6'], similarity=0.6)
        n6 = make_func('n6', calls=[], similarity=0.1)

        graph, result_map = self._build_graph_and_map([n1, n2, n3, n4, n5, n6])

        chain_a = [n1, n2, n3, n4]
        chain_b = [n2, n3, n4, n5]
        chain_c = [n3, n4, n5, n6]

        merged = _merge_overlapping_chains([chain_a, chain_b, chain_c], graph, result_map)

        assert len(merged) == 1
        merged_ids = {f.function.function_id for f in merged[0]}
        assert merged_ids == {'n2', 'n3', 'n4', 'n5'}
        ids = [f.function.function_id for f in merged[0]]
        assert ids == ['n2', 'n3', 'n4', 'n5']

    def test_disjoint_chains_unchanged(self):
        """Completely disjoint chains should remain unchanged."""
        A = make_func('A', calls=['B'], similarity=0.9)
        B = make_func('B', calls=[], similarity=0.8)
        C = make_func('C', calls=['D'], similarity=0.7)
        D = make_func('D', calls=[], similarity=0.6)

        graph, result_map = self._build_graph_and_map([A, B, C, D])

        chain1 = [A, B]
        chain2 = [C, D]

        merged = _merge_overlapping_chains([chain1, chain2], graph, result_map)

        assert len(merged) == 2
        merged_id_sets = [{f.function.function_id for f in c} for c in merged]
        assert {'A', 'B'} in merged_id_sets
        assert {'C', 'D'} in merged_id_sets

    def test_core_preserves_calling_order(self):
        """Merged core chain should preserve topological calling order."""
        # Chain1: [A, B, C, D], Chain2: [F, B, C, D]
        # Core: {B, C, D} — give core high scores, entry points low
        A = make_func('A', calls=['B'], similarity=0.1)
        B = make_func('B', calls=['C'], similarity=0.9)
        C = make_func('C', calls=['D'], similarity=0.8)
        D = make_func('D', calls=[], similarity=0.7)
        F = make_func('F', calls=['B'], similarity=0.1)

        graph, result_map = self._build_graph_and_map([A, B, C, D, F])

        chain1 = [A, B, C, D]
        chain2 = [F, B, C, D]

        merged = _merge_overlapping_chains([chain1, chain2], graph, result_map)

        assert len(merged) == 1
        ids = [f.function.function_id for f in merged[0]]
        assert ids == ['B', 'C', 'D']


class TestTopologicalSortCallOrder:
    """Tests that topological sort preserves call order from the graph edges."""

    def _build_graph_and_map(self, funcs):
        """Helper: build graph adjacency list and result_map from mock functions."""
        from collections import defaultdict
        result_map = {f.function.function_id: f for f in funcs}
        func_ids = set(result_map.keys())
        graph = defaultdict(list)
        for f in funcs:
            caller = f.function.function_id
            for callee in (f.function.calls or []):
                if callee in func_ids:
                    graph[caller].append(callee)
        return graph, result_map

    def test_siblings_ordered_by_call_position_not_similarity(self):
        """Siblings should follow parent's call order, not similarity score.

        This is the core bug: when A calls [B, C, D] in that order,
        topological sort should emit B, C, D — not reorder by similarity.
        """
        # A calls B, C, D in that order. D has highest similarity.
        A = make_func('A', calls=['B', 'C', 'D'], similarity=0.9)
        B = make_func('B', calls=[], similarity=0.3)
        C = make_func('C', calls=[], similarity=0.5)
        D = make_func('D', calls=[], similarity=0.8)

        graph, result_map = self._build_graph_and_map([A, B, C, D])
        nodes = set(result_map.keys())

        result = _topological_sort_chain(nodes, graph, result_map)
        ids = [f.function.function_id for f in result]

        assert ids[0] == 'A'
        # B, C, D should follow A in call order, NOT similarity order
        assert ids.index('B') < ids.index('C')
        assert ids.index('C') < ids.index('D')

    def test_nested_chain_call_order(self):
        """Reproduces the _chunk_by_title scenario: parent calls [init, combined, chunks].

        Without the fix, similarity-based sort would reorder siblings.
        """
        parent = make_func('parent', calls=['init', 'combined', 'chunks'], similarity=0.9)
        init = make_func('init', calls=[], similarity=0.4)
        combined = make_func('combined', calls=[], similarity=0.6)
        chunks = make_func('chunks', calls=[], similarity=0.8)

        graph, result_map = self._build_graph_and_map([parent, init, combined, chunks])
        nodes = set(result_map.keys())

        result = _topological_sort_chain(nodes, graph, result_map)
        ids = [f.function.function_id for f in result]

        assert ids[0] == 'parent'
        assert ids[1] == 'init'
        assert ids[2] == 'combined'
        assert ids[3] == 'chunks'

    def test_diamond_preserves_call_order(self):
        """Diamond: A calls [B, C], both B and C call D.

        After A, B should come before C (call order), then D.
        """
        A = make_func('A', calls=['B', 'C'], similarity=0.9)
        B = make_func('B', calls=['D'], similarity=0.3)
        C = make_func('C', calls=['D'], similarity=0.8)
        D = make_func('D', calls=[], similarity=0.5)

        graph, result_map = self._build_graph_and_map([A, B, C, D])
        nodes = set(result_map.keys())

        result = _topological_sort_chain(nodes, graph, result_map)
        ids = [f.function.function_id for f in result]

        assert ids[0] == 'A'
        assert ids.index('B') < ids.index('C')  # Call order from A
        assert ids.index('C') < ids.index('D')  # D after both callers
