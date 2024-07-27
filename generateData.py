import numpy as np
import torch


import numpy as np
import torch


class GenerateData:
    """Generate data points."""

    def __init__(self, domain):
        self.domain = domain
        self.dim = len(self.domain)

    def samplePoints(self, pointCount):
        """Sample the interior of the domain.

        Parameters:
            pointCount: int
                eg: 100

        Returns:
            xPoint: torch.Tensor
        """
        xPoint = []
        for i in range(self.dim):
            xPoint.append(
                np.random.uniform(
                    low=self.domain[i][0], high=self.domain[i][1], size=(pointCount, 1)
                )
            )

        # Convert list of numpy arrays to a single numpy array, then to a tensor
        xPoint = np.array(xPoint).squeeze().T
        xPoint = torch.tensor(xPoint, requires_grad=True, dtype=torch.float32)
        return xPoint

    def sampleGrid(self, nPoint=100):
        gridPoints = np.meshgrid(*[np.linspace(-1, 1, nPoint) for _ in range(self.dim)])

        # Stack the arrays, transpose, and reshape
        gridPoints = np.stack(gridPoints, axis=-1).reshape(-1, self.dim)

        # Convert to tensor
        gridPoints = torch.tensor(gridPoints, requires_grad=True, dtype=torch.float32)
        return gridPoints
