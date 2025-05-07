import unittest
import numpy as np
import attention


class TestAttention(unittest.TestCase):
    def test_attention_v0(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                m1 = np.random.rand(128, 128).astype(dtype)
                m2 = np.random.rand(128, 128).astype(dtype)
                res = attention.attention(m1, m2)
                np.testing.assert_allclose(m1 @ m2, res, atol=1e-4)

    def test_attention_v1(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                m1 = np.random.rand(128, 128).astype(dtype)
                m2 = np.random.rand(128, 128).astype(dtype)
                res = attention.attention(m1, m2, version=1)
                np.testing.assert_allclose(m1 @ m2, res, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
