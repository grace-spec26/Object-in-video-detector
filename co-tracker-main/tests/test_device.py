import unittest

from cotracker.device import resolve_default_device


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeMps:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeBackends:
    def __init__(self, mps_available):
        self.mps = FakeMps(mps_available)


class FakeTorch:
    def __init__(self, cuda_available=False, mps_available=False):
        self.cuda = FakeCuda(cuda_available)
        self.backends = FakeBackends(mps_available)


class DeviceTest(unittest.TestCase):
    def test_resolve_default_device_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(
            resolve_default_device(FakeTorch(cuda_available=True, mps_available=True)),
            "cuda",
        )
        self.assertEqual(
            resolve_default_device(FakeTorch(cuda_available=False, mps_available=True)),
            "mps",
        )
        self.assertEqual(
            resolve_default_device(FakeTorch(cuda_available=False, mps_available=False)),
            "cpu",
        )


if __name__ == "__main__":
    unittest.main()
