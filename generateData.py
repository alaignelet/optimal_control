import numpy as np
import torch


class GenerateData:
    """Generate data points.

    This class provides methods to generate random sample points within a specified domain.
    """

    def __init__(self, domain):
        self.domain = domain
        self.dim = len(self.domain)

    def samplePoints(self, pointCount):
        """Sample the interior of the domain.

        This method generates random sample points within the specified domain.

        Parameters:
            pointCount (int): The number of sample points to generate.

        Returns:
            torch.Tensor: A tensor containing the sampled points.
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

    def sampleGrid(self, nPoint):
        """Sample the grid within the domain.

        This method generates a grid of sample points within the specified domain.

        Parameters:
            nPoint (int): The number of points along each dimension of the grid.

        Returns:
            torch.Tensor: A tensor containing the grid points.
        """
        gridPoints = np.meshgrid(*[np.linspace(-1, 1, nPoint) for _ in range(self.dim)])

        # Stack the arrays, transpose, and reshape
        gridPoints = np.stack(gridPoints, axis=-1).reshape(-1, self.dim)

        # Convert to tensor
        gridPoints = torch.tensor(gridPoints, requires_grad=True, dtype=torch.float32)
        return gridPoints
