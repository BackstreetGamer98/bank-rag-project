import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import RateLimiter


class RateLimitTests(unittest.TestCase):
    def test_blocks_after_limit_then_recovers(self):
        rl = RateLimiter(3, 60)
        self.assertTrue(all(rl.allow("a", now=t) for t in (0, 1, 2)))
        self.assertFalse(rl.allow("a", now=3))
        self.assertTrue(rl.allow("a", now=62))

    def test_clients_are_counted_separately(self):
        rl = RateLimiter(1, 60)
        self.assertTrue(rl.allow("a", now=0))
        self.assertTrue(rl.allow("b", now=0))
        self.assertFalse(rl.allow("a", now=1))


if __name__ == "__main__":
    unittest.main()
