from __future__ import annotations
import sys
import os

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Iterable, List, Sequence, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Box = List[Tuple[int, int]]  # e.g. [(a_lo,a_hi), (b_lo,b_hi), (c_lo,c_hi)]

class SearchSpace:
    """
    k-D discrete grid search space with optional hard-gap boxes.

    - dims: iterable of positive ints, the size along each axis (k = len(dims))
    - Maintains a boolean mask of allowed cells (True = allowed, False = forbidden).
    - Supports adding hard gaps via axis-aligned inclusive boxes.
    - Provides helpers for (un)raveling indices and uniform sampling over allowed cells.
    """

    def __init__(self, dims: Sequence[int]) -> None:
        self._dims = np.asarray(dims, dtype=int)
        if self._dims.ndim != 1 or (self._dims <= 0).any():
            raise ValueError("dims must be a 1-D sequence of positive integers")
        self._ndim = int(self._dims.size)
        self._size = int(np.prod(self._dims, dtype=np.int64))
        # Allowed mask stored in k-D shape for convenient slicing
        self._allow = np.ones(self._dims, dtype=bool)
        self._energy = np.zeros(self._dims, dtype=float)   # total energy field
        self._energy[~self._allow] = np.inf
        self._gap_boxes = []


    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(d) for d in self._dims)

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self) -> int:
        return self._size

    @property
    def allowed_count(self) -> int:
        return int(self._allow.sum())

    @property
    def allow_mask(self) -> np.ndarray:
        """Return a read-only view of the allowed mask."""
        m = self._allow.view()
        m.setflags(write=False)
        return m

    def add_hard_box(self, box: Box) -> None:
        """
        Add a hard-gap box specified as inclusive index ranges per axis.
        Example (2D): box=[(2,5),(7,9)] forbids all points with 2<=x<=5 and 7<=y<=9.
        """
        if len(box) != self._ndim:
            raise ValueError(f"box must have {self._ndim} (lo,hi) pairs")

        slices: List[slice] = []
        ranges = []  # to keep each axis range explicitly
        for dim, (lo, hi) in enumerate(box):
            d = int(self._dims[dim])
            lo_c = max(0, min(d - 1, int(lo)))
            hi_c = max(0, min(d - 1, int(hi)))
            if hi_c < lo_c:
                return
            slices.append(slice(lo_c, hi_c + 1))
            ranges.append(range(lo_c, hi_c + 1))
        self._allow[tuple(slices)] = False
        self._gap_boxes.append(box)  


    def add_hard_boxes(self, boxes: Iterable[Box]) -> None:
        for b in boxes:
            self.add_hard_box(b)

    def clear_hard_boxes(self) -> None:
        self._allow[...] = True

    def ravel(self, coords: Sequence[int]) -> int:
        """k-D → 1-D (flat) index."""
        if len(coords) != self._ndim:
            raise ValueError(f"coords length must be {self._ndim}")
        return int(np.ravel_multi_index(tuple(coords), self.shape))

    def unravel(self, idx: int) -> Tuple[int, ...]:
        """1-D (flat) → k-D index tuple."""
        if not (0 <= idx < self._size):
            raise IndexError("flat index out of bounds")
        return tuple(int(i) for i in np.unravel_index(idx, self.shape))


    # attractor: contributes NEGATIVE energy (deeper well) 
    def add_attractor(self, center, weight=1.0, spread=1.0):
        """
        ENERGY-STAGE VERSION (Gibbs):
        - `weight` is interpreted as an ENERGY AMPLITUDE (aka "mass" or depth).
        - `spread` is a (scalar) Gaussian-like σ controlling how wide the well is.
        Effect: E(x) ← E(x) - weight * exp(-||x-center||^2 / (2 σ^2))
        """
        center = np.asarray(center, dtype=float)
        # all grid coordinates (k-D), fully vectorized
        coords = np.indices(self._dims).reshape(self._ndim, -1).T
        diffs = coords - center
        dist2 = np.sum(diffs * diffs, axis=1)

        kernel = np.exp(-dist2 / (2.0 * float(spread)**2)).reshape(self._dims)

        # subtract (deepen) energy where allowed; leave disallowed as +inf
        self._energy -= float(weight) * kernel
        self._energy[~self._allow] = np.inf

    # repeller: contributes POSITIVE energy (higher hill) 
    def add_repeller(self, center, strength=0.5, spread=1.0):
        """
        ENERGY-STAGE VERSION (Gibbs):
        - `strength` is an ENERGY AMPLITUDE (height).
        - `spread` is Gaussian-like σ controlling the radius of influence.
        Effect: E(x) ← E(x) + strength * exp(-||x-center||^2 / (2 σ^2))
        """
        center = np.asarray(center, dtype=float)
        coords = np.indices(self._dims).reshape(self._ndim, -1).T
        diffs = coords - center
        dist2 = np.sum(diffs * diffs, axis=1)

        kernel = np.exp(-dist2 / (2.0 * float(spread)**2)).reshape(self._dims)

        # add (raise) energy where allowed; leave disallowed as +inf
        self._energy += float(strength) * kernel
        self._energy[~self._allow] = np.inf

    def energy_stats(self):
        finite = np.isfinite(self._energy)
        print(f"Emin={np.min(self._energy[finite]):.3f}, Emax={np.max(self._energy[finite]):.3f}")

    # turn energy into a Gibbs PMF (for discrete variables) 
    def gibbs_pmf(self, temperature: float = 1.0) -> np.ndarray:
        """
        Convert current energy field E into a probability field p(x) ∝ exp(-E/τ).
        - `temperature` τ > 0 controls sharpness (lower τ = sharper peaks).
        - Disallowed cells (E=+inf) get p=0.
        Numerically stabilized with a shift by min finite energy.
        Returns a k-D array summing to 1 over allowed cells.
        """
        τ = float(temperature)
        E = self._energy

        # mask for finite/allowed cells
        finite_mask = np.isfinite(E)
        if not finite_mask.any():
            raise RuntimeError("Energy field has no finite/allowed cells.")

        # numeric stabilization: subtract minimum finite energy before exponentiating
        Emin = np.min(E[finite_mask])
        logits = np.empty_like(E, dtype=float)
        logits[finite_mask] = -(E[finite_mask] - Emin) / τ
        logits[~finite_mask] = -np.inf  # ensures p=0 there

        # exponentiate and normalize
        p = np.exp(logits)
        Z = np.sum(p[finite_mask])
        if Z <= 0:
            raise RuntimeError("Partition function is zero; check energy construction.")
        p /= Z
        return p

    # sample directly from the Gibbs field (Gibbs-only regime)
    def sample_gibbs(self, n: int, temperature: float = 1.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        unique_coords = set()
        while len(unique_coords) < n:
            p = self.gibbs_pmf(temperature=temperature)
            flat = p.ravel()
            idxs = np.flatnonzero(flat > 0)
            probs = flat[idxs]
            picks = rng.choice(idxs, size=n, p=probs)
            coords = np.column_stack(np.unravel_index(picks, self.shape))
            unique_coords.update(map(tuple, coords))
        # truncate in case we overshoot slightly
        return set(list(unique_coords)[:n])
    

    # debug/summary 
    def summary(self) -> str:
        denied = self._size - self.allowed_count
        return (
            f"SearchSpace(shape={self.shape}, size={self.size}, "
            f"allowed={self.allowed_count}, denied={denied})"

        )

    def __repr__(self) -> str:
        return self.summary()

    # VISUALIZER ENTRYPOINT
    def visualize(self, figsize=(10,10), alpha=0.4, cmap="turbo"):
        """
        Visualize all gap boxes in this SearchSpace.
        Supports 2D or 3D.
        """
        fig = None
        ndim = self._ndim
        dims = self._dims

        if len(self._gap_boxes) == 0:
            print("No gap boxes to visualize.")
            return

        if ndim == 2:
            fig = self._visualize_2d(self._gap_boxes, dims, figsize, alpha, cmap)
        elif ndim == 3:
            fig = self._visualize_3d(self._gap_boxes, dims, figsize, alpha, cmap)
        else:
            raise NotImplementedError("Visualization only supports 2D or 3D search spaces.")

        return fig
    
    def _visualize_2d(self, gap_boxes, dims, figsize, alpha, cmap):

        W, H = dims
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")

        cmap_obj = plt.cm.get_cmap(cmap)
        # --- FIRST: paint entire domain black ---
        for x in range(W):
            for y in range(H):
                ax.add_patch(
                    plt.Rectangle(
                        (x, y), 1, 1,
                        facecolor="black",
                        edgecolor="none"
                    )
                )

        for i, box in enumerate(gap_boxes):
            (x0, x1), (y0, y1) = box[:2]
            w, h = (x1 - x0), (y1 - y0)
            color = cmap_obj(0.6)

            rect = plt.Rectangle(
                (x0, y0), w, h,
                facecolor=color,
                edgecolor="black",
                alpha=alpha
            )
            ax.add_patch(rect)
            ax.text(
                x0 + w/2, y0 + h/2,
                str(i),
                ha="center", va="center",
                fontsize=8, color="black"
            )

        ax.set_title(f"2D Global Gap Boxes ({len(gap_boxes)})")
        plt.tight_layout()
        return fig

    def _visualize_3d(self, gap_boxes, dims, figsize, alpha, cmap):
        W, H, D = dims
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(0, D)

        cmap_obj = plt.cm.get_cmap(cmap)

        # sort by depth
        sorted_boxes = sorted(
            enumerate(gap_boxes),
            key=lambda t: sum((b[0]+b[1])/2 for b in t[1]),
            reverse=True
        )
        for x in range(W):
            for y in range(H):
                for z in range(D):
                    self._draw_cube(
                        ax, x, x+1, y, y+1, z, z+1,
                        color="black",
                        alpha=1.0
                    )

        for i, box in sorted_boxes:
            (x0,x1), (y0,y1), (z0,z1) = box
            color = cmap_obj(0.6)
            self._draw_cube(ax, x0,x1, y0,y1, z0,z1, color, alpha)

            ax.text((x0+x1)/2, (y0+y1)/2, (z0+z1)/2,
                    str(i), color="black")

        ax.set_title(f"3D Global Gap Boxes ({len(gap_boxes)})")
        plt.tight_layout()
        return fig

    def _draw_cube(self, ax, x0, x1, y0, y1, z0, z1, color, alpha):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        vertices = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])

        faces = [
            [vertices[j] for j in [0,1,2,3]],
            [vertices[j] for j in [4,5,6,7]],
            [vertices[j] for j in [0,1,5,4]],
            [vertices[j] for j in [2,3,7,6]],
            [vertices[j] for j in [1,2,6,5]],
            [vertices[j] for j in [4,7,3,0]],
        ]
        poly = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="black",
            linewidths=0.6,
            alpha=alpha
        )
        ax.add_collection3d(poly)