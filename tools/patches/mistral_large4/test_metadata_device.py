import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE = importlib.util.find_spec("cann_ops_transformer")
if PACKAGE is None:
    pytest.skip("Install and patch cann_ops_transformer before running these SDK tests.", allow_module_level=True)
SOURCE = Path(PACKAGE.origin).parent / "utils" / "metadata_device.py"


@pytest.fixture
def runtime(monkeypatch):
    spec = importlib.util.spec_from_file_location("metadata_device_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = [0]
    counts = {0: (32, 64), 2: (36, 72)}
    calls = []

    def query(index, attribute):
        calls.append((index, attribute))
        return counts[index][0 if attribute == module.ACL_DEV_ATTR_AICORE_CORE_NUM else 1], 0

    monkeypatch.setattr(
        module.torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: current[0],
            get_device_properties=lambda *_: SimpleNamespace(cube_core_num=32, vector_core_num=64),
        ),
        raising=False,
    )
    monkeypatch.setattr(module.acl.rt, "get_device_info", query)
    return module, current, counts, calls


def test_core_counts_follow_native_device_instead_of_torch_cube_properties(runtime):
    module, current, _, calls = runtime
    assert module.get_metadata_core_counts() == (32, 64)
    current[0] = 2
    assert module.get_metadata_core_counts() == (36, 72)
    assert module.get_metadata_core_counts(0) == (32, 64)
    assert len(calls) == 4  # One successful query pair per device.


@pytest.mark.parametrize("counts", [(20, 40), (32, 64), (36, 72), (48, 96)])
def test_core_counts_have_no_sku_capacity_floor(runtime, counts):
    module, _, capabilities, _ = runtime
    capabilities[0] = counts
    assert module.get_metadata_core_counts() == counts


@pytest.mark.parametrize("count,status", [(32, 1), (0, 0), (-1, 0)])
def test_query_errors_do_not_return_or_cache_an_undersized_default(runtime, monkeypatch, count, status):
    module, _, _, _ = runtime
    monkeypatch.setattr(module.acl.rt, "get_device_info", lambda *_: (count, status))
    with pytest.raises(RuntimeError, match="Cannot query metadata core count"):
        module.get_metadata_core_counts()
    monkeypatch.setattr(module.acl.rt, "get_device_info", lambda *_: (24, 0))
    assert module.get_metadata_core_counts() == (24, 24)


@pytest.mark.parametrize("device,error", [(True, TypeError), (-1, ValueError), ("cpu", ValueError)])
def test_invalid_devices_fail_before_a_native_query(runtime, device, error):
    module, _, _, calls = runtime
    with pytest.raises(error):
        module.get_metadata_core_counts(device)
    assert not calls
