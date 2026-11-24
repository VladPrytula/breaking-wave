"""Resolvedness of retained planes: exact planes resolve, stacked sheets do not."""

import numpy as np
import pytest

from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.resolvedness import (
    MIXTURE_STACKED_PARTIAL, UNRESOLVED_ANGLE_DEG, InterfaceResolvedness,
    measure_interface_resolvedness,
)
from tests.physics3d.test_height_normals import _plane, _state


def _measure(state):
    plic = reconstruct_plic(state, estimate_interface_normals(state).normals)
    return plic, measure_interface_resolvedness(state, plic)


def _interior_cells(result, margin):
    i, j, k = result.cell_indices
    inside = np.ones(len(i), bool)
    for index, n in zip((i, j, k), result.shape, strict=True):
        inside &= (index >= margin) & (index < n - margin)
    return inside


@pytest.mark.parametrize("normal", [(1.1, .07, 1), (.07, 1, 1.1), (-1.1, .07, -.9)])
def test_exact_anisotropic_plane_is_resolved_everywhere(normal):
    state = _state()
    _plane(state, normal, 0.0031)
    _, result = _measure(state)
    # Height-function normals are verified five cells from the open boundaries;
    # the stencil of a measured cell must stay inside that region.
    inside = _interior_cells(result, 6)
    assert inside.sum() > 30
    assert np.all(result.angle_deg[inside] >= 0.0)
    assert np.max(result.angle_deg[inside]) < 1e-3
    # float32 plane volumes with one small normal component round at 2e-5.
    assert np.max(result.residual[inside]) < 1e-4
    # A steep plane stacks partial cells in a column; its planes agree, so it
    # is not a mixture.
    i, j, k = result.cell_indices
    interior_column = np.ones(result.shape[:2], bool)
    interior_column[i[~inside], j[~inside]] = False
    assert interior_column[6:-6, 6:-6].sum() > 30
    assert not np.any(result.unresolved[inside])
    assert not np.any(result.stacked_unresolved[interior_column])
    assert not np.any(result.mixture_columns[interior_column])


def test_horizontal_plane_facets_cover_the_cell_footprint():
    state = _state()
    g = state.cfg.grid
    _plane(state, (0, 0, 1), 0.3 * g.dz)
    _, result = _measure(state)
    assert len(result.cells) == g.nx * g.ny
    np.testing.assert_allclose(result.facet_area_m2, g.dx * g.dy, rtol=1e-5)
    assert np.max(result.angle_deg) < 1e-3
    assert np.max(result.residual) < 1e-6


def _column_state(column):
    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, np.float32)
    alpha[:, :, :len(column)] = np.asarray(column, np.float32)
    state.set_interior("alpha", alpha)
    return state


def test_two_sheet_column_is_unresolved_and_mixture():
    # Water, a 10 mm air gap, then a second sheet: three partial cells in four.
    state = _column_state([1, 1, 0.5, 0, 0.6, 0.4])
    _, result = _measure(state)
    assert np.all(result.mixture_columns)
    assert np.all(result.stacked_partial == 3)
    assert np.all(result.stacked_unresolved == 3)
    _, _, k = result.cell_indices
    lower_sheet_floor = k == 4
    assert np.all(result.angle_deg[lower_sheet_floor] > UNRESOLVED_ANGLE_DEG)
    assert np.all(result.residual[lower_sheet_floor] > 0.1)
    g = state.cfg.grid
    np.testing.assert_allclose(
        result.summary()["mixture_partial_volume_m3"],
        1.5 * g.nx * g.ny * g.dx * g.dy * g.dz, rtol=1e-5)
    single = _column_state([1, 1, 0.5])
    _, control = _measure(single)
    assert not np.any(control.mixture_columns)
    assert np.all(control.stacked_partial == 1)
    smeared = _column_state([1, 0.8, 0.6, 0.4, 0.2])
    _, smeared_result = _measure(smeared)
    assert np.max(smeared_result.angle_deg) < 1.0
    assert np.min(smeared_result.residual) > 0.1
    assert np.all(smeared_result.mixture_columns)
    # Two stacked partial cells whose planes agree are one sheet.
    two_cell_sheet = _column_state([1, 0.9, 0.1])
    _, two_cell_result = _measure(two_cell_sheet)
    assert np.all(two_cell_result.stacked_partial == MIXTURE_STACKED_PARTIAL)
    assert not np.any(two_cell_result.mixture_columns)
    assert np.max(control.angle_deg) < 1e-3


def test_measurement_reads_without_writing_and_round_trips(tmp_path):
    state = _column_state([1, 1, 0.5, 0, 0.6, 0.4])
    plic = reconstruct_plic(state, estimate_interface_normals(state).normals)
    before = {name: state.get_interior(name).copy() for name in ("alpha", "u", "v", "w")}
    planes = (plic.normals.numpy().copy(), plic.offsets.numpy().copy())
    result = measure_interface_resolvedness(state, plic)
    for name, field in before.items():
        np.testing.assert_array_equal(state.get_interior(name), field)
    np.testing.assert_array_equal(plic.normals.numpy(), planes[0])
    np.testing.assert_array_equal(plic.offsets.numpy(), planes[1])
    saved = InterfaceResolvedness.read(result.write(tmp_path / "resolvedness.npz"))
    assert saved.summary() == result.summary()
    np.testing.assert_array_equal(saved.cells, result.cells)
