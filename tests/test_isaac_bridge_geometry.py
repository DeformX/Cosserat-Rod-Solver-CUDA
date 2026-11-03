"""The Kit-free part of ropesim.isaac_bridge: tessellated primitives are closed meshes whose bakes match the
closed-form distances, and bake_body shells an open mesh instead of refusing it."""
import numpy as np
import pytest

pytest.importorskip("warp")      # the bakes need Warp (pip extra `bake`)

from ropesim import sdf  # noqa: E402
from ropesim.isaac_bridge import ColliderSpec, _box_mesh, _cylinder_mesh, _sphere_mesh, bake_body, weld  # noqa: E402


def _check(grid, phi_ref, pts, tol):
    phi, inside = grid.sample(pts)
    assert inside.all()
    err = np.abs(phi - phi_ref(pts))
    assert err.max() < tol, err.max()


def test_box_mesh_is_closed_and_bakes_to_the_box_sdf():
    V, F = _box_mesh(np.array([0.2, 0.2, 0.1]))
    assert sdf.is_closed(F)
    g = bake_body(V, F, ColliderSpec("x", spacing=0.01, margin=0.05, round=0.0))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.22, 0.22, (2000, 3)) * np.array([1, 1, 0.55])
    _check(g, sdf.sdf_box((0.2, 0.2, 0.1)), pts, 5e-3)                 # interpolation across edges at 1 cm cells


def test_sphere_and_cylinder_meshes():
    V, F = _sphere_mesh(0.1)
    assert sdf.is_closed(F)
    g = bake_body(V, F, ColliderSpec("x", spacing=0.005, margin=0.03))
    pts = np.random.default_rng(1).uniform(-0.12, 0.12, (2000, 3))
    _check(g, sdf.sdf_sphere(0.1), pts, 3e-3)                # 24-segment tessellation
    for axis in range(3):
        V, F = _cylinder_mesh(0.05, 0.3, axis)
        assert sdf.is_closed(F)
        g = bake_body(V, F, ColliderSpec("x", spacing=0.005, margin=0.03))
        keep = [d for d in range(3) if d != axis]
        pts = np.random.default_rng(2).uniform(-0.06, 0.06, (2000, 3)); pts[:, axis] *= 0.5   # well inside the length
        _check(g, lambda p: np.linalg.norm(p[:, keep], axis=1) - 0.05, pts, 3e-3)


def test_capsule_is_a_union_of_closed_parts():
    V, F = _cylinder_mesh(0.05, 0.3, 2, capsule=True)
    assert sdf.is_closed(F)                                    # three closed components, overlapping
    g = bake_body(V, F, ColliderSpec("x", spacing=0.005, margin=0.03))
    phi, _ = g.sample(np.array([[0, 0, 0.0], [0, 0, 0.19], [0, 0, 0.21], [0.049, 0, 0.0], [0, 0.07, 0.0]]))
    assert phi[0] < -0.045 and phi[1] < -0.005 and phi[2] > 0.005 and abs(phi[3]) < 2e-3 and abs(phi[4] - 0.02) < 2e-3


def test_open_mesh_becomes_a_shell():
    V, F = _box_mesh(np.array([0.2, 0.2, 0.1]))
    F = F[:-2]                                                 # drop one face: open
    assert not sdf.is_closed(F)
    with pytest.warns(UserWarning, match="is open"):
        g = bake_body(V, F, ColliderSpec("x", spacing=0.01, margin=0.05))
    phi, _ = g.sample(np.array([[0, 0, 0.0], [0, 0, -0.1], [0, 0, 0.1]]))
    assert phi[0] > 0.05 and abs(phi[1] + 0.01) < 3e-3         # hollow inside; the bottom face is a 2-cell shell
    assert phi[2] > 0.05                                       # the dropped (top) face is gone


def test_rounding_fattens_the_shape():
    V, F = _box_mesh(np.array([0.2, 0.2, 0.1]))
    g = bake_body(V, F, ColliderSpec("x", spacing=0.01, margin=0.05, round=0.02))
    phi, _ = g.sample(np.array([[0, 0, 0.11], [0, 0, 0.13], [0.2, 0.2, 0.1]]))
    assert abs(phi[0] + 0.01) < 2e-3 and abs(phi[1] - 0.01) < 2e-3   # the top face moved up by `round`
    assert phi[2] < 0.0                                        # the true corner is inside the rounded body


def test_unwelded_export_is_welded_before_the_closedness_test():
    """Exported colliders duplicate vertices along every seam (review 3: all 7 UR5 links look open); the bake
    welds first and only an actually open mesh becomes a shell (and says so)."""
    V, F = _box_mesh(np.array([0.2, 0.2, 0.1]))
    Vu, Fu = V[F.reshape(-1)], np.arange(F.size).reshape(-1, 3)             # one vertex copy per corner of every triangle
    assert not sdf.is_closed(Fu)
    Vw, Fw = weld(Vu, Fu)
    assert len(Vw) == 8 and sdf.is_closed(Fw)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # a welded, closed collider warns about nothing
        g = bake_body(Vu, Fu, ColliderSpec("x", spacing=0.01, margin=0.05))
    phi, _ = g.sample(np.array([[0, 0, 0.0], [0, 0, 0.1], [0, 0, 0.15]]))
    assert phi[0] < -0.09 and abs(phi[1]) < 2e-3 and abs(phi[2] - 0.05) < 2e-3   # solid, surface on the true face
    with pytest.warns(UserWarning, match="lid is open"):
        bake_body(Vu, Fu[:-2], ColliderSpec("x", spacing=0.01, margin=0.05), "lid")
