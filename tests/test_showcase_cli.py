"""Regression checks for showcase failures and hand exporter paths."""
import runpy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_twist_divergence_exits_without_replay(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from scene import common

    read_positions = common.device_positions
    reads = 0

    def divergent_positions(batch):
        nonlocal reads
        positions = read_positions(batch)
        reads += 1
        if reads == 2:  # first recording frame, after the settling diagnostic
            positions[0, 0, 0] = np.nan
        return positions

    monkeypatch.setattr(common, "device_positions", divergent_positions)
    out = tmp_path / "twist.html"
    monkeypatch.setattr(sys, "argv", ["twist.py", "--settle", "0", "--duration", "0.1",
                                      "--substeps", "1", "--itr", "1", "--out", str(out)])
    with pytest.raises(SystemExit, match="[Dd][Ii][Vv][Ee][Rr][Gg]") as error:
        runpy.run_path(str(EXAMPLES / "twist.py"), run_name="__main__")
    assert error.value.code
    assert not out.exists()
    assert not out.with_suffix(".npz").exists()


def test_strings_resolves_paths_before_starting_hand_exporter(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    hand = tmp_path / "model" / "right_hand.xml"
    hand.parent.mkdir()
    hand.write_text("<mujoco/>\n")
    # The external exporter is replaced with a probe so this test needs no
    # MuJoCo install or Shadow Hand meshes. The CLI and subprocess are real.
    interpreter = tmp_path / "hand-python"
    interpreter.write_text(f"#!{sys.executable}\n" + """
from pathlib import Path
import sys

model, spec, result = map(Path, sys.argv[3:])
assert model.is_file() and model.parent == Path.cwd()
assert spec.is_file()
result.with_suffix('.marker').write_text('paths resolved')
sys.exit(23)
""")
    interpreter.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "strings.py"), "--hand_xml", "model/right_hand.xml",
         "--work", "work", "--venv_python", "./hand-python"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert (tmp_path / "work" / "hand.marker").is_file(), result.stderr
