import os, sys

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List, Tuple, Dict, Optional, Iterator
from pysat.solvers import Minisat22, Glucose42, Cadical195

def enumerate_all_models(
    clauses: List[List[int]], num_vars: int, solver_cls=Cadical195
    ) -> Iterator[List[int]]:
    """
    Enumerate ALL satisfying assignments of a CNF by adding blocking clauses.
    """
    with solver_cls(bootstrap_with=clauses,) as s:
        while s.solve():
            model = s.get_model()
            # build assignment map for exactly 1..num_vars
            assign: Dict[int, bool] = {}
            for lit in model:
                v = abs(lit)
                if 1 <= v <= num_vars:
                    assign[v] = (lit > 0)

            # ensure all variables are assigned
            missing = [v for v in range(1, num_vars + 1) if v not in assign]
            if missing:
                raise ValueError(
                    f"Solver returned a partial model. Missing vars (first 10): {missing[:10]}. "
                    f"Pass the correct num_vars and ensure CNF references only 1..num_vars."
                )
            yield model
            # blocking clause: flip every literal in the found assignment
            block = [(-v if assign[v] else v) for v in range(1, num_vars + 1)]
            s.add_clause(block)


def SAT_test(clauses: List[List[int]]) -> None:
    """
    Test SAT solver on given clauses, print SAT/UNSAT and a single model if SAT.
    """
    with Cadical195(bootstrap_with=clauses, with_proof=True) as solver:
        sat = solver.solve()
        if sat:
            model = solver.get_model()
            return model
        else:
            return None