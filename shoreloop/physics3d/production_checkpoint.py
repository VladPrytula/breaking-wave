"""Direct production-state checkpoints for long advanced-water runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


_ENERGY_FIELDS = (
    "molecular_loss",
    "sgs_loss",
    "wall_loss",
    "porous_loss",
    "numerical_residual",
    "positive_numerical_loss",
    "boundary_work",
    "boundary_surface_work",
    "steps",
)


def write_production_checkpoint(solver, path: str | Path) -> Path:
    """Write every carried field needed to continue a production run."""
    destination = Path(path)
    payload = {
        f"state_{name}": solver.state.field(name).numpy()
        for name in solver.state.FIELDS
    }
    payload.update({
        "solver_time": np.asarray(solver.time),
        "solver_steps": np.asarray(solver.steps),
        "contact_previous_alpha": solver.contact_line.previous_alpha.numpy(),
        "contact_primed": np.asarray(int(solver.contact_line.primed)),
    })
    for name in _ENERGY_FIELDS:
        payload[f"energy_{name}"] = np.asarray(
            getattr(solver.energy_ledger, name),
        )
    # A failed write must leave the previous complete checkpoint available.
    # The temporary file shares its filesystem so replacement is atomic.
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    temporary.replace(destination)
    return destination


def load_production_checkpoint(solver, path: str | Path) -> None:
    """Restore a production solver built from the same fixed scene."""
    with np.load(path) as saved:
        for name in solver.state.FIELDS:
            solver.state.field(name).assign(saved[f"state_{name}"])
        solver.time = float(saved["solver_time"])
        solver.steps = int(saved["solver_steps"])
        # One accepted solver step advances VOF once; keep its split-order phase.
        solver.advection.vof.steps = solver.steps
        solver.contact_line.previous_alpha.assign(
            saved["contact_previous_alpha"],
        )
        solver.contact_line.primed = bool(saved["contact_primed"].item())
        for name in _ENERGY_FIELDS:
            setattr(
                solver.energy_ledger,
                name,
                saved[f"energy_{name}"].item(),
            )
