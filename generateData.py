import numpy as np
import torch


class GenerateData(object):
    """Generate data points."""

    def __init__(self, domain):
        self.domain = domain
        self.dim = len(self.domain)

    def samplePoints(self, pointCount):
        """Sample the interior of the domain.

        Parameters:
            domain: tuple of lists
                eg: ([0, 1], [0, 1])
            pointCount: int
                eg: 100

        Returns:
            xPoint: list of numpy.array
        """

        xPoint = []
        for i in range(self.dim):
            xPoint.append(
                np.random.uniform(
                    low=self.domain[i][0], high=self.domain[i][1], size=(pointCount, 1)
                )
            )

        xPoint = torch.tensor(xPoint, requires_grad=True)[:, :, 0].T.float()
        return xPoint

    def sampleGrid(self, nPoint=100):
        gridPoints = np.meshgrid(*[np.linspace(-1, 1, nPoint) for i in range(self.dim)])
        gridPoints = (
            torch.tensor(list(gridPoints), requires_grad=True)
            .T.float()
            .reshape(-1, self.dim)
        )
        return gridPoints
