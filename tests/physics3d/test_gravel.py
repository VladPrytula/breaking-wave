"""AW-1120: the gravel bed's laws follow from the grain size and dissipate."""

import math

import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.gravel import (
    CANOPY_DRAG_COEFFICIENT, GRAIN_D50, LAYER_DEPTH, POROSITY, ergun_forchheimer,
    ergun_permeability, frontal_area_density, gravel_drag, gravel_storage,
    gravel_wall_stress, roughness_length, unit_gradient_flux,
)
from shoreloop.physics3d.porous import (
    PorousBedModel, PorousDragConfig, canopy_weight_from_geometry, porous_capacity_from_geometry,
)
from shoreloop.physics3d.solids import measure_solid_flux
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.vof import initialize_box
from shoreloop.physics3d.wall_stress import WallStressConfig, WallStressModel
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def test_gravel_constants_follow_from_the_grain_size():
    k = ergun_permeability()
    assert k == pytest.approx(GRAIN_D50 ** 2 * POROSITY ** 3 / (150.0 * (1.0 - POROSITY) ** 2))
    assert ergun_forchheimer() == pytest.approx(1.75 / math.sqrt(150.0 * POROSITY ** 3))
    q = unit_gradient_flux()
    # The flux solves Ergun at unit hydraulic gradient, in the inertial regime for gravel.
    assert 1.0e-6 / (9.81 * k) * q + ergun_forchheimer() / (9.81 * math.sqrt(k)) * q * q == pytest.approx(1.0)
    assert 0.05 < q < 0.3
    storage = gravel_storage()
    assert storage.capacity_fraction == POROSITY and storage.layer_depth == LAYER_DEPTH
    assert storage.infiltration_rate == pytest.approx(q / (POROSITY * LAYER_DEPTH))
    assert storage.drainage_time == pytest.approx(POROSITY * LAYER_DEPTH / q)
    assert storage.infiltration_rate * storage.drainage_time == pytest.approx(1.0)
    wall = gravel_wall_stress()
    assert wall.law == "log" and wall.roughness_length == pytest.approx(2.0 * GRAIN_D50 / 30.0)
    assert roughness_length() < 0.5 * 0.01  # below the production grid's first-cell height
    with pytest.raises(ValueError):
        WallStressConfig(law="log")
    drag = gravel_drag()
    assert drag.law == "canopy" and drag.layer_depth == GRAIN_D50
    assert frontal_area_density() == pytest.approx(1.5 * (1.0 - POROSITY) / GRAIN_D50)
    assert drag.canopy_rate == pytest.approx(0.5 * CANOPY_DRAG_COEFFICIENT * frontal_area_density())
    assert 10.0 < drag.canopy_rate < 20.0  # 1/m: the magnitude AW-607's measured bracket had, 17.9 /m
    with pytest.raises(ValueError):
        PorousDragConfig(law="canopy")


def _sloped_beach(water_table):
    # 0.15 slope from x = 0.3 m; 25 mm x 20 mm cells, the development spacing.
    grid = Grid3DConfig(nx=40, ny=2, nz=12, Lx=1.0, Ly=0.1, Lz=0.24)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    initialize_box(state, (grid.x0, grid.y0, grid.z0), (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + grid.Lz))
    cut = initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(toe_x=0.3, slope=0.15, base_elevation=0.0)))
    return cfg, state, cut


def test_capacity_follows_the_bed_height_above_the_water_table_and_the_canopy_sits_on_the_grains():
    water_table = 0.06  # bed reaches it at x = 0.7 m
    cfg, state, cut = _sloped_beach(water_table)
    g = cfg.grid
    storage = gravel_storage(water_table=water_table)
    capacity = porous_capacity_from_geometry(cut, cfg, storage)
    x = g.x0 + (np.arange(g.nx) + 0.5) * g.dx
    bed = np.clip((x - 0.3) * 0.15, 0.0, None)
    per_area = capacity.sum(axis=2)[:, 0] * g.dz           # metres of water per unit plan area
    expected = POROSITY * np.clip(bed - water_table, 0.0, LAYER_DEPTH)
    interior = (x > 0.35) & (x < 0.95)
    assert np.all(per_area[(x < 0.7) & interior] == 0.0)   # saturated below the water table
    assert per_area[(x > 0.75) & interior] == pytest.approx(expected[(x > 0.75) & interior], abs=0.6 * POROSITY * g.dz)
    assert per_area.max() > 0.5 * POROSITY * (bed.max() - water_table)
    uniform = porous_capacity_from_geometry(cut, cfg, gravel_storage())
    assert np.all(uniform.sum(axis=2)[:, 0][(x > 0.35) & (x < 0.7)] * g.dz > 0.0)
    weight = canopy_weight_from_geometry(cut, cfg, gravel_drag())
    canopy = (weight * cut.theta).sum(axis=2)[:, 0] * g.dz  # metres of water column inside the canopy
    beach = (x > 0.35) & (x < 0.95)
    assert canopy[beach] == pytest.approx(GRAIN_D50, abs=1e-3)
    assert np.all(canopy[x < 0.3] == 0.0)
    assert weight.max() <= 1.0 and np.all(weight[cut.theta == 0.0] == 0.0)
    # The water of a bed cut cell all sits among the grain tops: full weight there.
    cut_cells = (cut.theta > 0.0) & (cut.theta < 1.0) & np.broadcast_to(beach[:, None, None], cut.theta.shape)
    assert np.all(weight[cut_cells] == 1.0)


def test_canopy_drag_is_dissipative_and_linear_in_speed():
    results = []
    for speed in (0.4, 0.8):
        cfg, state, cut = _sloped_beach(None)
        state.fill_("u", speed)
        state.fill_("v", 0.0)
        state.fill_("w", 0.0)
        model = PorousBedModel.from_geometry(cfg, cut, drag_settings=gravel_drag())
        results.append((state, model.apply_drag(state, 0.002)))
    slow, fast = results[0][1], results[1][1]
    assert slow.energy_loss > 0.0 and fast.energy_loss > 0.0
    assert fast.maximum_damping_rate == pytest.approx(2.0 * slow.maximum_damping_rate, rel=1e-3)
    assert fast.maximum_damping_rate == pytest.approx(gravel_drag().canopy_rate * 0.8, rel=0.05)
    assert float(np.min(results[1][0].get_interior("u"))) < 0.8
    assert float(np.min(model.dissipation.numpy())) >= 0.0


def _sheet_over_beach(speed: float, discretization: str):
    grid = Grid3DConfig(nx=12, ny=6, nz=8, Lx=1.2, Ly=0.6, Lz=0.8)
    cfg = Solver3DConfig(grid=grid, device=DEVICE, pressure_discretization=discretization)
    state = State3D(cfg)
    initialize_box(state, (grid.x0, grid.y0, grid.z0),
                   (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + grid.Lz))
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(toe_x=2.0, slope=0.0, base_elevation=0.2)))
    state.fill_("u", speed)
    state.fill_("v", 0.0)
    state.fill_("w", 0.0)
    model = WallStressModel(cfg, gravel_wall_stress())
    return state, model.apply(state, 0.02)


@pytest.mark.parametrize("discretization", ["ghost_fluid", "liquid_mimetic"])
def test_log_law_stress_is_quadratic_in_speed_and_dissipative(discretization):
    slow, slow_result = _sheet_over_beach(0.5, discretization)
    fast, fast_result = _sheet_over_beach(1.0, discretization)
    assert slow_result.energy_loss > 0.0 and fast_result.energy_loss > 0.0
    # Rate proportional to the speed: the damping rate doubles, the work more than quadruples.
    assert fast_result.maximum_damping_rate == pytest.approx(2.0 * slow_result.maximum_damping_rate, rel=1e-3)
    assert fast_result.energy_loss > 4.0 * slow_result.energy_loss
    assert measure_solid_flux(fast).absolute_flux == 0.0
    z = fast.grid.coordinates("u")[2]
    first_fluid = int(np.flatnonzero(z > 0.2)[0])
    assert float(np.mean(fast.get_interior("u")[:, :, first_fluid])) < 1.0
