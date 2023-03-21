import numpy.testing
import unittest
import numpy as np

from geometry_utils import triangulation, geometry


class TestTriangulation(unittest.TestCase):
    def test_sdr_no_noise(self):
        n_poses = 5
        problem = triangulation.get_aholt_problem(n_poses, angle_jitter_deg=10)
        observations = problem.get_observations(0.)

        for method in [
            triangulation.TriangulationSDR,
            triangulation.TriangulationFractionalSDR,
            triangulation.RobustTriangulationSDR,
            triangulation.RobustTriangulationFractionalSDR,
        ]:
            with self.subTest(str(method)):
                sdr = method(problem.poses, observations, problem.K_inv)
                sdr.solve(eps_abs=1e-12, eps_rel=1e-12, solver='SCS', verbose=True)
                result = sdr.get_solution()
                self.assertTrue(result["success"])

                # check that the original point is recovered
                np.testing.assert_array_almost_equal(result["estimated_point"], problem.point)

    def check_triangulation(self, sdr, problem, observations):
        sdr.solve(eps_abs=1e-12, eps_rel=1e-12, solver='SCS', verbose=True)
        result = sdr.get_solution()
        self.assertTrue(result["success"])

        # check that the optimal value of the sdr is the same as for the recovered solution
        reprojections = geometry.reproject(result["estimated_point"], problem.poses, problem.K)
        np.testing.assert_array_almost_equal(np.linalg.norm(reprojections - observations) ** 2,
                                             sdr.objective() / sdr.scale ** 2)
        return result["estimated_point"]

    def check_sdr_with_noise(self, n_poses, K, name):
        problem = triangulation.get_aholt_problem(n_poses, angle_jitter_deg=10, K=K)
        observations = problem.get_observations(0.1)
        for method in [
            triangulation.TriangulationSDR,
            triangulation.RobustTriangulationSDR,
            triangulation.TriangulationFractionalSDR,
            triangulation.RobustTriangulationFractionalSDR,
        ]:
            with self.subTest(f' {name}: {method}'):
                sdr = method(problem.poses, observations, problem.K)
                point_est = self.check_triangulation(sdr, problem, observations)

                sdr_scaled = method(problem.poses, observations, problem.K, scale=0.1)
                point_est2 = self.check_triangulation(sdr_scaled, problem, observations)
                np.testing.assert_array_almost_equal(point_est, point_est2)

    def test_sdr_with_noise(self):
        n_poses = 5
        self.check_sdr_with_noise(n_poses, np.eye(3) + np.random.randn(3, 3) * 0.1, "same intrinsics")
        self.check_sdr_with_noise(n_poses, np.eye(3)[None] + np.random.randn(n_poses, 3, 3) * 0.1,
                                  "different intrinsics")

    @staticmethod
    def check_triangulation_with_outliers(sdr, problem, observations, inlier_mask):
        sdr.solve(eps_abs=1e-12, eps_rel=1e-12, solver='SCS', verbose=True)
        result = sdr.get_solution()

        np.testing.assert_array_almost_equal(result["estimated_inlier_mask"], inlier_mask)
        reprojections = geometry.reproject(result["estimated_point"], problem.poses, problem.K_inv)
        np.testing.assert_array_almost_equal(
            np.linalg.norm((reprojections - observations)[inlier_mask]) ** 2 + (1 - inlier_mask).sum(),
            sdr.objective() / sdr.scale ** 2
        )
        return result["estimated_point"]

    def test_sdr_with_outliers(self):
        # will fail occasionally due to non-tight relaxation
        n_poses = 7
        problem = triangulation.get_aholt_problem(n_poses, angle_jitter_deg=10)
        observations = problem.get_observations(0.01)
        observations, inlier_mask = triangulation.add_outliers(observations, 1, 10, 10)

        for method in [
            triangulation.RobustTriangulationSDR,
            triangulation.RobustTriangulationFractionalSDR,
        ]:
            # with self.subTest(str(method)):
            sdr = method(problem.poses, observations, problem.K)
            point_est = self.check_triangulation_with_outliers(sdr, problem, observations, inlier_mask)

            sdr_scaled = method(problem.poses, observations, problem.K, scale=0.1)
            point_est2 = self.check_triangulation_with_outliers(sdr_scaled, problem, observations, inlier_mask)
            np.testing.assert_array_almost_equal(point_est, point_est2)

    @staticmethod
    def check_algebraic(n_poses, K):
        problem = triangulation.get_aholt_problem(n_poses, angle_jitter_deg=10, K=K)
        observations = problem.get_observations(0.)
        triangulated_point, s = triangulation.triangulate_algebraic(
            observations,
            problem.poses,
            K=K,
        )
        numpy.testing.assert_array_almost_equal(triangulated_point, problem.point)

    def test_algebraic(self):
        n_poses = 7
        self.check_algebraic(n_poses, np.eye(3) + np.random.randn(3, 3) * 0.1)
        self.check_algebraic(n_poses, np.eye(3)[None] + np.random.randn(n_poses, 3, 3) * 0.1)


class TestGeometry(unittest.TestCase):
    def check_fundamental(self, n_poses, K):
        problem = triangulation.get_aholt_problem(
            n_poses,
            angle_jitter_deg=10.,
            K=K,
        )
        observations = geometry.homogenize(problem.get_observations(0.))
        F = geometry.get_relative_fundamental(problem.poses, problem.K_inv)

        for i in range(n_poses):
            for j in range(n_poses):
                res = observations[i] @ F[i, j] @ observations[j]
                self.assertAlmostEqual(res, 0.)

    def test_fundamental(self):
        n_poses = 5
        self.check_fundamental(n_poses, np.eye(3)[None] + np.random.randn(n_poses, 3, 3) * 0.1)
        self.check_fundamental(n_poses, np.eye(3) + np.random.randn(3, 3) * 0.1)


class TestMatrixMath(unittest.TestCase):
    def test_l2_matrix(self):
        x = np.random.randn(100)

        M = triangulation.get_l2_matrix(x, one_first=False)
        u = np.hstack([np.random.randn(100), 1])
        self.assertAlmostEqual(u @ M @ u, ((u[:-1] - x) ** 2).sum())

        M = triangulation.get_l2_matrix(x, one_first=True)
        u = np.hstack([1, np.random.randn(100)])
        self.assertAlmostEqual(u @ M @ u, ((u[1:] - x) ** 2).sum())

    def test_robust_l2_matrix(self):
        x = np.random.randn(10, 2)
        c = np.random.randn(10)

        M = triangulation.get_robust_l2_matrix(x, c)
        u = np.random.randn(10, 2)
        theta = np.random.randint(2, size=10)
        z = np.hstack([np.hstack([u, theta[:, None]]).ravel(), 1])
        self.assertAlmostEqual(z @ M @ z, ((u - theta[:, None] * x) ** 2).sum() + ((1 - theta) * c).sum())


if __name__ == '__main__':
    unittest.main()
