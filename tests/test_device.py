import pytest

from shoreloop import device


def test_prepare_device_retries_only_primary_context_startup(monkeypatch):
    class FakeCudaDevice:
        is_cuda = True

    selected = FakeCudaDevice()
    attempts = 0
    sleeps = []

    def set_device(candidate):
        nonlocal attempts
        assert candidate is selected
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Failed to acquire primary context for device cuda:0")

    monkeypatch.setattr(device.wp, "init", lambda: None)
    monkeypatch.setattr(device.wp, "get_device", lambda name: selected)
    monkeypatch.setattr(device.wp, "set_device", set_device)
    monkeypatch.setattr(device.time, "sleep", sleeps.append)

    assert device.prepare_device("cuda:0") is selected
    assert attempts == 3
    assert sleeps == [0.25, 0.5]

    def fail_for_another_reason(candidate):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(device.wp, "set_device", fail_for_another_reason)
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        device.prepare_device("cuda:0")
