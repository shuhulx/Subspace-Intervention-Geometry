"""Illustrative geometry tests, separate from pretrained-model evidence."""
import math
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def remove(x, u):
    norm = dot(u, u)
    if norm == 0:
        return list(x)
    return [a - dot(x, u) / norm * b for a, b in zip(x, u)]


def joint_remove(x, directions, tolerance=1e-12):
    basis = []
    for d in directions:
        q = list(d)
        for u in basis:
            q = remove(q, u)
        n = math.sqrt(dot(q, q))
        if n > tolerance:
            basis.append([a / n for a in q])
    for u in basis:
        x = remove(x, u)
    return x


class ProjectionTest(unittest.TestCase):
    def test_sequential_reintroduces_component(self):
        u = [1.0, 0.0, 0.0]
        v = [1.0, 1.0, 0.0]
        x = [2.0, 3.0, 4.0]
        self.assertAlmostEqual(dot(remove(x, u), u), 0.0)
        self.assertGreater(abs(dot(remove(remove(x, u), v), u)), 1.0)

    def test_joint_and_rank_deficiency(self):
        x = [2.0, 3.0, 4.0]
        u = [1.0, 0.0, 0.0]
        v = [1.0, 1.0, 0.0]
        y = joint_remove(x, [u, v, u, [2.0, 0.0, 0.0]])
        self.assertAlmostEqual(dot(y, u), 0.0)
        self.assertAlmostEqual(dot(y, v), 0.0)
        self.assertAlmostEqual(y[2], 4.0)

    def test_order(self):
        x = [2.0, 3.0, 4.0]
        u = [1.0, 0.0, 0.0]
        v = [1.0, 1.0, 0.0]
        a = joint_remove(x, [u, v])
        b = joint_remove(x, [v, u])
        for s, t in zip(a, b):
            self.assertAlmostEqual(s, t)


if __name__ == '__main__':
    unittest.main()
