#!/usr/bin/env python3
"""
Keyboard Layout Optimization via Simulated Annealing
Developer: Akash Somasundaram EE24B084

Notes:
- Cost is total Euclidean distance between consecutive characters.
- Coordinates are fixed (QWERTY-staggered grid). Optimization swaps assignments.

This base code uses Python "types" - these are optional, but very helpful
for debugging and to help with editing.

"""

import argparse
import json
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore


Point = Tuple[float, float]
Layout = Dict[str, Point]


def qwerty_coordinates(chars: str) -> Layout:
    """Return QWERTY grid coordinates for the provided character set.

    The grid is a simple staggered layout (units are arbitrary):
    - Row 0: qwertyuiop at y=0, x in [0..9]
    - Row 1: asdfghjkl at y=1, x in [0.5..8.5]
    - Row 2: zxcvbnm at y=2, x in [1..6]
    - Space at (4.5, 3)
    Characters not present in the grid default to the space position.
    """
    row0 = "qwertyuiop"
    row1 = "asdfghjkl"
    row2 = "zxcvbnm"

    coords: Layout = {}
    for i, c in enumerate(row0):
        coords[c] = (float(i), 0.0)
    for i, c in enumerate(row1):
        coords[c] = (0.5 + float(i), 1.0)
    for i, c in enumerate(row2):
        coords[c] = (1.0 + float(i), 2.0)
    coords[" "] = (4.5, 3.0)

    # Backfill for requested chars; unknowns get space position.
    space_xy = coords[" "]
    for ch in chars:
        if ch not in coords:
            coords[ch] = space_xy
    return coords


def initial_layout() -> Layout:
    """Create an initial layout mapping chars to some arbitrary positions of letters."""

    # Start with identity for letters and space; others mapped to space.
    base_keys = "abcdefghijklmnopqrstuvwxyz "

    # Get coords - or use coords of space as default
    layout = qwerty_coordinates(base_keys)
    return layout


def preprocess_text(text: str, chars: str) -> str:
    """Lowercase and filter to the allowed character set; map others to space."""
    text = text.lower()
    allowed = set(chars)
    return "".join(c if c in allowed else " " for c in text)


def path_length_cost(text: str, layout: Layout) -> float:
    """Sum Euclidean distances across consecutive characters in text."""
    cost=0.0
    for i in range(1, len(text)):
        x1, y1 = layout[text[i-1]]
        x2, y2 = layout[text[i]]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cost += dist
    return cost


######
# Define any other useful functions, such as to create new layout etc.
######
def swap_chars(layout: Layout) -> Layout:
    """Swap two random characters in the layout and return the new layout."""
    new_layout=layout.copy()
    ch1, ch2 = random.sample(list(new_layout.keys()), 2)
    new_layout[ch1], new_layout[ch2] = new_layout[ch2], new_layout[ch1]
    return new_layout

# Dataclass is like a C struct - you can use it just to store data if you wish
# It provides some convenience functions for assignments, printing etc.
@dataclass
class SAParams:
    iters: int = 50000
    t0: float = 1.0  # Initial temperature setting
    alpha: float = 0.999  # geometric decay per iteration
    epoch: int = 1  # iterations per temperature step (1 = per-iter decay)


def simulated_annealing(
    text: str,
    layout: Layout,
    params: SAParams,
    rng: random.Random,
) -> Tuple[Layout, float, List[float], List[float]]:
    """Simulated annealing to minimize path-length cost over character swaps.

    Returns best layout, best cost, and two lists:
    - best cost up to now (monotonically decreasing)
    - cost of current solution (may occasionally go up)
    These will be used for plotting
    """

    Lbest=[]
    Lcurrent=[]
    current_layout=layout.copy()
    best_layout=layout.copy()
    current_cost=path_length_cost(text,layout)
    best_cost=current_cost
    Temp=params.t0

    for i in range(params.iters):
        new_layout=swap_chars(current_layout)
        new_cost=path_length_cost(text,new_layout)
        delta=new_cost-current_cost             

        if delta<0 or rng.random() < math.exp(-delta/Temp):           # Accept new layout if better or with some probability if worse
            current_layout=new_layout
            current_cost=new_cost
            if current_cost<best_cost:
                best_layout=current_layout
                best_cost=current_cost
        Lbest.append(best_cost)
        Lcurrent.append(current_cost)

    if i % params.epoch==0:         # Decay temperature every epoch iterations
        Temp *= params.alpha

    return best_layout, best_cost, Lbest, Lcurrent

def plot_costs(
    layout: Layout, best_trace: List[float], current_trace: List[float]
) -> None:

    # Plot cost trace
    out_dir = "."
    plt.figure(figsize=(6, 3))
    plt.plot(best_trace, lw=1.5)
    plt.plot(current_trace, lw=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Best Cost vs Iteration")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cost_trace.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Plot layout scatter
    xs, ys, labels = [], [], []
    for ch, (x, y) in layout.items():
        xs.append(x)
        ys.append(y)
        labels.append(ch)

    plt.figure(figsize=(6, 3))
    plt.scatter(xs, ys, s=250, c="#1f77b4")
    for x, y, ch in zip(xs, ys, labels):
        plt.text(
            x,
            y,
            ch,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="#1f77b4", ec="none", alpha=0.9),
        )
    plt.gca().invert_yaxis()
    plt.title("Optimized Layout")
    plt.axis("equal")
    plt.tight_layout()
    path = os.path.join(out_dir, f"layout.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def load_text(filename) -> str:
    if filename is not None:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback demo text
    return (
        "the quick brown fox jumps over the lazy dog\n"
        "APL is the best course ever\n"
    )


def main(filename: str | None = None) -> None:
    rng = random.Random(0)
    chars = "abcdefghijklmnopqrstuvqxyz "

    # Initial assignment - QWERTY
    layout0 = initial_layout()

    # Prepare text and evaluate baseline
    raw_text = load_text(filename)
    text = preprocess_text(raw_text, chars)

    baseline_cost = path_length_cost(text, layout0)
    print(f"Baseline (QWERTY assignment) cost: {baseline_cost:.4f}")

    # Annealing - give parameter values
    params = SAParams(iters=5000, t0=1.0, alpha=0.99, epoch=1)         # Change these values to see how they affect results
    start = time.time()
    best_layout, best_cost, best_trace, current_trace = simulated_annealing(text, layout0, params, rng)
    dur = time.time() - start
    print(f"Optimized cost: {best_cost:.4f}  (improvement {(baseline_cost - best_cost):.4f})")
    print(f"Runtime: {dur:.2f}s over {params.iters} iterations")

    plot_costs(best_layout, best_trace, current_trace)


if __name__ == "__main__":
    main()
