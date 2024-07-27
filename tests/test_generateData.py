import unittest
from generateData import GenerateData


class TestGenerateData(unittest.TestCase):
    def setUp(self):
        self.domain = [[-1, 1], [-2, 2]]  # Example domain
        self.generator = GenerateData(self.domain)

    def test_samplePoints(self):
        pointCount = 100
        sampledPoints = self.generator.samplePoints(pointCount)

        # Check if the shape of the sampled points is correct
        self.assertEqual(sampledPoints.shape, (pointCount, self.generator.dim))

        # Check if the sampled points are within the specified domain
        for i in range(self.generator.dim):
            self.assertTrue(
                (sampledPoints[:, i] >= self.domain[i][0]).all()
                and (sampledPoints[:, i] <= self.domain[i][1]).all()
            )


if __name__ == "__main__":
    unittest.main()
