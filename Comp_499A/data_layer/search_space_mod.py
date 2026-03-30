from __future__ import annotations
import sys
import os

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Set
import numpy as np
Box = List[Tuple[int, int]]  # inclusive per axis


@dataclass(frozen=True)
class EnergyTerm:
    kind: str              # "attractor" | "repeller"
    center: np.ndarray     # shape (ndim,)
    amp: float             # energy amplitude
    spread: float          # sigma


class SearchSpace:
    def __init__(self, dims: Sequence[int]) -> None:
        self._dims = np.asarray(dims, dtype=np.int64)
        if self._dims.ndim != 1 or (self._dims <= 0).any():
            raise ValueError("dims must be a 1-D sequence of positive integers")
        self._ndim = int(self._dims.size)
        self._size = int(np.prod(self._dims, dtype=np.int64))

        self._gap_boxes: List[Box] = []
        self._terms: List[EnergyTerm] = []

    # basic props
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(d) for d in self._dims)

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self) -> int:
        return self._size

    # forbidden regions 
    def add_hard_box(self, box: Box) -> None:
        if len(box) != self._ndim:
            raise ValueError(f"box must have {self._ndim} (lo,hi) pairs")

        clipped: Box = []
        for dim, (lo, hi) in enumerate(box):
            d = int(self._dims[dim])
            lo_c = max(0, min(d - 1, int(lo)))
            hi_c = max(0, min(d - 1, int(hi)))
            if hi_c < lo_c:
                return  # empty box
            clipped.append((lo_c, hi_c))

        self._gap_boxes.append(clipped)

    def add_hard_boxes(self, boxes: Iterable[Box]) -> None:
        for b in boxes:
            self.add_hard_box(b)

    def clear_hard_boxes(self) -> None:
        self._gap_boxes.clear()

    def is_allowed(self, coords: Sequence[int]) -> bool:
        if len(coords) != self._ndim:
            raise ValueError(f"coords length must be {self._ndim}")
        # bounds check
        for i, c in enumerate(coords):
            if c < 0 or c >= self._dims[i]:
                return False

        # forbidden if inside any gap box
        for box in self._gap_boxes:
            inside = True
            for (lo, hi), c in zip(box, coords):
                if c < lo or c > hi:
                    inside = False
                    break
            if inside:
                return False
        return True

    # energy terms 
    def add_attractor(self, center, weight: float = 1.0, spread: float = 1.0) -> None:
        c = np.asarray(center, dtype=float)
        if c.shape != (self._ndim,):
            raise ValueError(f"center must have shape ({self._ndim},)")
        self._terms.append(EnergyTerm("attractor", c, float(weight), float(spread)))

    def add_repeller(self, center, strength: float = 0.5, spread: float = 1.0) -> None:
        c = np.asarray(center, dtype=float)
        if c.shape != (self._ndim,):
            raise ValueError(f"center must have shape ({self._ndim},)")
        self._terms.append(EnergyTerm("repeller", c, float(strength), float(spread)))

    def energy(self, coords: Sequence[int]) -> float:
        if not self.is_allowed(coords):
            return float("inf")
        x = np.asarray(coords, dtype=float)
        E = 0.0
        for t in self._terms:
            diff = x - t.center
            dist2 = float(diff @ diff)
            k = np.exp(-dist2 / (2.0 * (t.spread ** 2)))
            if t.kind == "attractor":
                E -= t.amp * k
            else:
                E += t.amp * k
        return E

    # sampling 
    def sample_uniform(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        max_tries: int = 50_000_000,
    ) -> Set[Tuple[int, ...]]:
        """
        Uniform over allowed points via rejection from the full domain.
        Good when forbidden volume is not overwhelming.
        """
        if rng is None:
            rng = np.random.default_rng()

        out: Set[Tuple[int, ...]] = set()
        tries = 0
        while len(out) < n:
            if tries >= max_tries:
                raise RuntimeError(
                    "Too many rejections; forbidden volume may be too large. "
                    "Use sample_gibbs() or reduce forbidden coverage."
                )
            pt = tuple(int(rng.integers(0, self._dims[i])) for i in range(self._ndim))
            if self.is_allowed(pt):
                out.add(pt)
            tries += 1
        return out

    def sample_gibbs(
        self,
        n: int,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        burn_in: int = 2000,
        steps_per_sample: int = 200,
        proposal: str = "local_step",
    ) -> Set[Tuple[int, ...]]:
        """
        Metropolis-Hastings sampling for π(x) ∝ exp(-E(x)/τ) over allowed points.

        proposal:
          - "local_step": +/-1 on one random axis (good mixing on grids)
          - "single_coord": resample one coordinate uniformly (bigger jumps)
        """
        if rng is None:
            rng = np.random.default_rng()
        τ = float(temperature)
        if τ <= 0:
            raise ValueError("temperature must be > 0")

        # find a start state
        x = None
        for _ in range(2000000):
            cand = tuple(int(rng.integers(0, self._dims[i])) for i in range(self._ndim))
            if self.is_allowed(cand):
                x = cand
                break
        if x is None:
            raise RuntimeError("No allowed point found to initialize MCMC.")

        Ex = self.energy(x)

        def propose(curr: Tuple[int, ...]) -> Tuple[int, ...]:
            arr = list(curr)
            j = int(rng.integers(0, self._ndim))
            if proposal == "local_step":
                step = 1 if rng.random() < 0.5 else -1
                arr[j] = int(np.clip(arr[j] + step, 0, self._dims[j] - 1))
            elif proposal == "single_coord":
                arr[j] = int(rng.integers(0, self._dims[j]))
            else:
                raise ValueError("unknown proposal")
            return tuple(arr)

        def mh_step(curr: Tuple[int, ...], Ecurr: float) -> Tuple[Tuple[int, ...], float]:
            cand = propose(curr)
            Ecand = self.energy(cand)
            if not np.isfinite(Ecand):
                return curr, Ecurr  # forbidden -> reject
            d = (Ecand - Ecurr) / τ
            if d <= 0.0 or rng.random() < np.exp(-d):
                return cand, Ecand
            return curr, Ecurr

        # burn-in
        for _ in range(burn_in):
            x, Ex = mh_step(x, Ex)

        # collect
        out: Set[Tuple[int, ...]] = set()
        while len(out) < n:
            for _ in range(steps_per_sample):
                x, Ex = mh_step(x, Ex)
            out.add(x)

        return out

    # ---------- misc ----------
    def summary(self) -> str:
        return (
            f"SearchSpace(shape={self.shape}, size={self.size}, "
            f"gaps={len(self._gap_boxes)}, terms={len(self._terms)})"
        )

    def __repr__(self) -> str:
        return self.summary()