import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.topology import (
    TopologyTracker,
    _breaking_column_support,
)
from shoreloop.physics3d.vof import initialize_plane


def _state() -> State3D:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=12, ny=4, nz=12,
                          Lx=1.2, Ly=0.4, Lz=1.2),
        device=DEVICE,
    )
    return State3D(cfg)


def test_height_field_interface_has_no_overturning_signals():
    state = _state()
    initialize_plane(state, (0.08, 0.0, 1.0), 0.55)

    measured = TopologyTracker(state.cfg).sample(state)
    assert measured.mixed_interface_cells > 0
    assert measured.downward_interface_cells == 0
    assert measured.downward_interface_x_centroid is None
    assert measured.downward_interface_x_shoreward_p95 is None
    assert measured.multi_transition_columns == 0
    assert measured.supported_downward_interface_cells == 0
    assert measured.maximum_vertical_transitions <= 1
    assert measured.water_components == 1
    assert measured.component_mergers == 0
    assert measured.component_splits == 0


def test_synthetic_overhang_activates_geometry_and_component_events():
    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, dtype=np.float32)
    alpha[:, :, :3] = 1.0
    alpha[:, :, 3] = 0.25
    state.set_interior("alpha", alpha)
    tracker = TopologyTracker(state.cfg)
    baseline = tracker.sample(state)
    assert baseline.water_components == 1

    alpha[3:9, :, 5] = 0.25
    alpha[3:9, :, 6] = 1.0
    alpha[3:9, :, 7] = 0.25
    state.set_interior("alpha", alpha)
    overhang = tracker.sample(state)
    assert overhang.downward_interface_cells == 24
    assert overhang.downward_interface_x_centroid == pytest.approx(0.6)
    assert overhang.downward_interface_x_shoreward_p95 == pytest.approx(0.85)
    assert overhang.multi_transition_columns > 0
    assert overhang.maximum_vertical_transitions >= 3
    assert overhang.supported_downward_interface_cells == 24
    assert overhang.supported_downward_interface_x_centroid == pytest.approx(
        0.6,
    )
    assert (
        overhang.supported_downward_interface_x_shoreward_p95
        == pytest.approx(0.85)
    )
    assert overhang.largest_connected_multi_transition_columns == 24
    assert overhang.connected_multi_transition_x_span == pytest.approx(0.6)
    assert overhang.connected_multi_transition_y_span == pytest.approx(0.4)
    assert overhang.upper_sheet_thickness_minimum == pytest.approx(0.1)
    assert overhang.upper_sheet_thickness_median == pytest.approx(0.1)
    assert overhang.upper_sheet_thickness_maximum == pytest.approx(0.1)
    assert overhang.water_components == 2
    assert overhang.component_splits == 1

    alpha[5, :, 3:6] = 1.0
    state.set_interior("alpha", alpha)
    merged = tracker.sample(state)
    assert merged.water_components == 1
    assert merged.component_mergers == 1

    offshore = TopologyTracker(state.cfg, x_bounds=(0.0, 0.25)).sample(state)
    assert offshore.downward_interface_cells == 0
    assert offshore.multi_transition_columns == 0
    assert offshore.largest_connected_multi_transition_columns == 0
    assert offshore.connected_multi_transition_x_centroid is None


def test_connected_overturn_support_wraps_across_periodic_alongshore_seam():
    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, dtype=np.float32)
    alpha[:, :, :3] = 1.0
    alpha[:, :, 3] = 0.25
    alpha[3:9, (0, 3), 5] = 0.25
    alpha[3:9, (0, 3), 6] = 1.0
    alpha[3:9, (0, 3), 7] = 0.25
    state.set_interior("alpha", alpha)

    measured = TopologyTracker(state.cfg).sample(state)

    assert measured.multi_transition_columns == 12
    assert measured.largest_connected_multi_transition_columns == 12
    assert measured.connected_multi_transition_y_span == pytest.approx(0.2)


def test_fold_support_excludes_larger_component_without_downward_interface():
    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, dtype=np.float32)
    alpha[:, :, :3] = 1.0
    alpha[1:5, :, 6] = 1.0
    alpha[8:10, 1:3, 6] = 1.0
    solid = np.zeros_like(alpha, dtype=np.int32)
    downward = np.zeros(alpha.shape[:2], dtype=np.int32)
    downward[8:10, 1:3] = 2

    measured = _breaking_column_support(
        alpha, solid, downward, state.cfg, 0, state.cfg.grid.nx,
    )

    assert measured["largest_connected_multi_transition_columns"] == 4
    assert measured["supported_downward_interface_cells"] == 8
    assert measured["supported_downward_interface_x_centroid"] == pytest.approx(
        0.9,
    )
    assert measured[
        "supported_downward_interface_x_shoreward_p95"
    ] == pytest.approx(0.95)

    downward.fill(0)
    downward[0, 0] = 2
    unlinked = _breaking_column_support(
        alpha, solid, downward, state.cfg, 0, state.cfg.grid.nx,
    )
    assert unlinked["supported_downward_interface_cells"] == 0
    assert unlinked["largest_connected_multi_transition_columns"] == 0
    assert unlinked["supported_downward_interface_x_shoreward_p95"] is None


def test_fold_path_combines_all_spatially_supported_directional_lobes():
    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, dtype=np.float32)
    alpha[:, :, :3] = 1.0
    alpha[1:3, 0:2, 6] = 1.0
    alpha[4:7, 0, 6] = 1.0
    alpha[8:11, 2:4, 6] = 1.0
    solid = np.zeros_like(alpha, dtype=np.int32)
    downward = np.zeros(alpha.shape[:2], dtype=np.int32)
    downward[1:3, 0:2] = 1
    downward[4:7, 0] = 100
    downward[8:11, 2:4] = 2

    measured = _breaking_column_support(
        alpha, solid, downward, state.cfg, 0, state.cfg.grid.nx,
    )

    assert measured["largest_connected_multi_transition_columns"] == 6
    assert measured["supported_downward_interface_cells"] == 16
    assert measured["supported_downward_interface_x_centroid"] == pytest.approx(
        0.7625,
    )
    assert measured[
        "supported_downward_interface_x_shoreward_p95"
    ] == pytest.approx(1.05)
