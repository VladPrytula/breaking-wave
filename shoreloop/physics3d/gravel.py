"""Gravel-bed closures derived from the reference's grain size.

The reference beach (``IMG_4543.MOV``) is rounded gravel with a 3 cm median
grain (2-4 cm). Every number here follows from that grain size and a declared
porosity; nothing is tuned on the simulation.

Infiltration and drainage. Ergun (1952) gives the pore-scale resistance of a
packed bed of grains ``d`` at porosity ``phi``: permeability
``k = d^2 phi^3 / (150 (1 - phi)^2)`` and Forchheimer coefficient
``c_F = 1.75 / sqrt(150 phi^3)``. Water percolating under gravity alone
(hydraulic gradient one) moves at the flux ``q`` solving
``1 = (nu / (g k)) q + (c_F / (g sqrt(k))) q^2``, about 0.12 m/s for 3 cm
gravel, inertial rather than Darcy. Gravel's capillary suction is negligible
against that, so a swash sheet enters dry gravel at ``q`` (Green-Ampt with
no suction head) until the pores over the unsaturated depth ``L`` are full:
the surface layer holds ``phi`` per unit volume, fills at the rate
``q / (phi L)`` and, once exposed, percolates to the water table in
``phi L / q`` (``shoreloop/physics3d/porous.py`` keeps its relaxation form;
these are its constants).

Where the bed can take water. A gravel beach is saturated at the water
table, which sits at the still-water level, so the pore volume a swash can
fill is ``phi`` times the bed's height above that level, capped at ``L``;
``PorousStorageConfig.water_table`` binds the capacity to it (M11 AW-1120:
the uniform layer put ``30 mm`` of capacity at the waterline, where the real
bed holds none).

Bed resistance. The swash sheet (``10-15 mm``) is thinner than the grain
(``30 mm``): it flows among the grain tops, inside the roughness sublayer,
and its resistance is the form drag of the grains, ``1/2 rho C_d a |u| u``
per unit volume with the frontal area density of a bed of spheres at solid
fraction ``1 - phi``, ``a = 3 (1 - phi) / (2 d50)``, and ``C_d`` of order
one for bluff bodies in a dense array (Nepf 1999 for canopies). It acts on
the water within one grain of the bed (``PorousDragConfig`` law
``canopy``), ``15.5 |u| /s`` for the reference grain, the magnitude
AW-607's grid-scale bracket (``17.9 |u| /s`` over ``8 cm``) had found by
measurement. The wall stress under the canopy is the first capture's Navier
slip. The fully rough log-law wall function from Nikuradse's
``k_s = 2 d50`` (``z0 = k_s / 30``, ``wall_stress.py`` law ``log``) was the
first form of this bed: measured on the AW-1119 probe it under-braked the
sheet (a Darcy-Weisbach ``f`` of ``0.5`` against the bracket's ``1.4``) and
the swash overshot the reference; it stays available as ``gravel_wall_stress``.
"""

from __future__ import annotations

import math

from shoreloop.physics3d.porous import PorousDragConfig, PorousStorageConfig
from shoreloop.physics3d.wall_stress import WallStressConfig

GRAIN_D50 = 0.03            # m, the reference's median pebble (2-4 cm)
POROSITY = 0.38             # loose rounded gravel, 0.35-0.40
NIKURADSE_GRAINS = 2.0      # k_s / d50 of a gravel bed
KARMAN = 0.41
GRAVITY = 9.81              # m/s^2
KINEMATIC_VISCOSITY = 1.0e-6  # m^2/s, water
LAYER_DEPTH = 0.08          # m, the most unsaturated gravel a column offers the swash (the first capture's storage layer)
CANOPY_DRAG_COEFFICIENT = 1.0  # bluff grains in a dense array


def ergun_permeability(grain: float = GRAIN_D50, porosity: float = POROSITY) -> float:
    """Pore-scale permeability of a packed bed, m^2."""
    return grain ** 2 * porosity ** 3 / (150.0 * (1.0 - porosity) ** 2)


def ergun_forchheimer(porosity: float = POROSITY) -> float:
    """Dimensionless inertial resistance coefficient of a packed bed."""
    return 1.75 / math.sqrt(150.0 * porosity ** 3)


def unit_gradient_flux(grain: float = GRAIN_D50, porosity: float = POROSITY, *,
                       kinematic_viscosity: float = KINEMATIC_VISCOSITY,
                       gravity: float = GRAVITY) -> float:
    """Percolation flux under gravity alone, m/s: the bed's conductivity at unit gradient."""
    k = ergun_permeability(grain, porosity)
    linear = kinematic_viscosity / (gravity * k)
    quadratic = ergun_forchheimer(porosity) / (gravity * math.sqrt(k))
    return (-linear + math.sqrt(linear * linear + 4.0 * quadratic)) / (2.0 * quadratic)


def roughness_length(grain: float = GRAIN_D50) -> float:
    """Fully rough log-law z0 of a gravel bed, m."""
    return NIKURADSE_GRAINS * grain / 30.0


def gravel_storage(grain: float = GRAIN_D50, porosity: float = POROSITY,
                   layer_depth: float = LAYER_DEPTH, *,
                   water_table: float | None = None) -> PorousStorageConfig:
    """Storage layer whose capacity, fill rate and drainage follow from the grain size.

    ``water_table`` (an elevation) binds each column's capacity to the bed's
    height above it; unset, every column offers the full layer.
    """
    flux = unit_gradient_flux(grain, porosity)
    pore_depth = porosity * layer_depth
    return PorousStorageConfig(
        layer_depth=layer_depth, capacity_fraction=porosity,
        infiltration_rate=flux / pore_depth, drainage_time=pore_depth / flux,
        water_table=water_table,
    )


def frontal_area_density(grain: float = GRAIN_D50, porosity: float = POROSITY) -> float:
    """Frontal area of the grains per unit bed volume, 1/m (spheres at solid fraction 1 - phi)."""
    return 1.5 * (1.0 - porosity) / grain


def gravel_drag(grain: float = GRAIN_D50, porosity: float = POROSITY,
                drag_coefficient: float = CANOPY_DRAG_COEFFICIENT) -> PorousDragConfig:
    """Grain canopy drag on the water within one grain of the bed."""
    return PorousDragConfig(
        law="canopy", drag_coefficient=drag_coefficient,
        frontal_area_density=frontal_area_density(grain, porosity), layer_depth=grain,
    )


def gravel_wall_stress(grain: float = GRAIN_D50, *, wall_viscosity: float = 2.0e-4,
                       boulder_slip_length: float = 0.006,
                       beach_slip_length: float = 0.012) -> WallStressConfig:
    """Log-law beach stress from the grain roughness (the measured first form, not the gravel bed's)."""
    return WallStressConfig(
        wall_viscosity=wall_viscosity, beach_slip_length=beach_slip_length,
        boulder_slip_length=boulder_slip_length, law="log",
        roughness_length=roughness_length(grain),
    )
