import torch
from torch import nn
from abc import ABC, abstractmethod
import math


class Positivity(ABC):
    """Interface for function that makes weights positive."""

    @abstractmethod
    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        """Transform raw weight to positive weight."""
        ...

    def inverse_transform(self, pos_weight: torch.Tensor) -> torch.Tensor:
        """Transform positive weight to raw weight before transform."""
        return self.__call__(pos_weight)


class NoPositivity(Positivity):
    """
    Dummy for positivity function.

    This should make it easier to compare ICNNs to regular networks.
    """

    def __call__(self, weight):
        return weight


class NegExpPositivity(Positivity):
    """
    Make weights positive by applying exponential function on negative values during forward pass.

    References
    ----------
    Sivaprasad et al. (2021)
        The Curious Case of Convex Neural Networks.
    """

    def __call__(self, weight):
        return torch.where(weight < 0, weight.exp(), weight)


class ExponentialPositivity(Positivity):
    """
    Make weights positive by applying exponential function during forward pass.
    """

    def __call__(self, weight):
        return torch.exp(weight)

    def inverse_transform(self, pos_weight):
        return torch.log(pos_weight)


class LazyClippedPositivity(Positivity):
    """
    Make weights positive by clipping negative weights after each update.

    References
    ----------
    Amos et al. (2017)
        Input-Convex Neural Networks.
    """

    def __call__(self, weight):
        with torch.no_grad():
            weight.clamp_(0)

        return weight


class ClippedPositivity(Positivity):
    """
    Make weights positive by using applying ReLU during forward pass.
    """

    def __call__(self, weight):
        return torch.relu(weight)


class ConvexLinear(nn.Linear):
    """Linear layer with positive weights."""

    def __init__(self, *args, positivity: Positivity = None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given as kwarg for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.positivity(self.weight), self.bias)


class TraditionalInitialiser:
    """
    Initialisation for regular networks using variance scaling.
    """

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in, bias is None)
        weight_mean_sq, weight_var = weight_dist
        torch.nn.init.normal_(weight, weight_mean_sq**0.5, weight_var**0.5)
        if bias is not None:
            bias_mean, bias_var = bias_dist
            torch.nn.init.normal_(bias, bias_mean, bias_var**0.5)

    def compute_parameters(
        self, fan_in: int, no_bias: bool = False
    ) -> tuple[tuple[float, float], tuple[float, float] | None]:
        return (0.0, self.gain / fan_in), (0.0, 0.0)


class ConvexInitialiser:
    """
    Initialisation method for input-convex networks.

    Parameters
    ----------
    var : float, optional
        The target variance fixed point.
        Should be a positive number.
    corr : float, optional
        The target correlation fixed point.
        Should be a value between -1 and 1, but typically positive.
    bias_noise : float, optional
        The fraction of variance to originate from the bias parameters.
        Should be a value between 0 and 1
    alpha : float, optional
        Scaling parameter for leaky ReLU.
        Should be a positive number.

    Examples
    --------
    Default initialisation

    >>> icnn = torch.nn.Sequential(
    ...     torch.nn.Linear(200, 400),
    ...     torch.nn.ReLU(),
    ...     ConvexLinear(400, 300),
    ... )
    >>> torch.nn.init.kaiming_uniform_(icnn[0].weight, nonlinearity="linear")
    >>> torch.nn.init.zeros_(icnn[0].bias)
    >>> convex_init = ConvexInitialiser()
    >>> w1, b1 = icnn[1].parameters()
    >>> convex_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and torch.isclose(b1.var(), torch.zeros(1))

    Initialisation with random bias parameters

    >>> convex_bias_init = ConvexInitialiser(bias_noise=0.5)
    >>> convex_bias_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and b1.var() > 0
    """

    @staticmethod
    @torch.no_grad()
    def init_log_normal_(
        weight: torch.Tensor, mean_sq: float, var: float
    ) -> torch.Tensor:
        """
        Initialise weights with samples from a log-normal distribution.

        Parameters
        ----------
        weight : torch.Tensor
            The parameter to be initialised.
        mean_sq : float
            The squared mean of the normal distribution underlying the log-normal.
        var : float
            The variance of the normal distribution underlying the log-normal.

        Returns
        -------
        weight : torch.Tensor
            A reference to the inputs that have been modified in-place.
        """
        log_mom2 = math.log(mean_sq + var)
        log_mean = math.log(mean_sq) - log_mom2 / 2.0
        log_var = log_mom2 - math.log(mean_sq)
        return torch.nn.init.normal_(weight, log_mean, log_var**0.5).exp_()

    def __init__(
        self,
        var: float = 1.0,
        corr: float = 0.5,
        bias_noise: float = 0.0,
        alpha: float = 0.0,
    ):
        self.target_var = var
        self.target_corr = corr
        self.bias_noise = bias_noise
        self.relu_scale = 2.0 / (1.0 + alpha**2)

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        if bias is None:
            raise ValueError(
                "Principled Initialisation for ICNNs requires bias parameter"
            )

        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in)
        weight_mean_sq, weight_var = weight_dist
        self.init_log_normal_(weight, weight_mean_sq, weight_var)

        bias_mean, bias_var = bias_dist
        torch.nn.init.normal_(bias, bias_mean, bias_var**0.5)

    def compute_parameters(
        self, fan_in: int
    ) -> tuple[tuple[float, float], tuple[float, float] | None]:
        """
        Compute the distribution parameters for the initialisation.

        Parameters
        ----------
        fan_in : int
            Number of incoming connections.

        Returns
        -------
        (weight_mean_sq, weight_var) : tuple of 2 float
            The squared mean and variance for weight parameters.
        (bias_mean, bias_var): tuple of 2 float, optional
            The mean and variance for the bias parameters.
            If `no_bias` is `True`, `None` is returned instead.
        """
        target_mean_sq = self.target_corr / self.corr_func(fan_in)
        target_variance = self.relu_scale * (1.0 - self.target_corr) / fan_in

        shift = fan_in * (target_mean_sq * self.target_var / (2 * math.pi)) ** 0.5
        bias_var = 0.0
        if self.bias_noise > 0.0:
            target_variance *= 1 - self.bias_noise
            bias_var = self.bias_noise * (1.0 - self.target_corr) * self.target_var

        return (target_mean_sq, target_variance), (-shift, bias_var)

    def corr_func(self, fan_in: int) -> float:
        """Helper function for correlation (cf. $f_\mathrm{c}$, eq. 35)."""
        rho = self.target_corr
        mix_mom = (1 - rho**2) ** 0.5 + rho * math.acos(-rho)
        return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)


class LinearSkip(nn.Module):
    """
    Fully-connected skip-connection with learnable parameters.

    The learnable parameters of this skip-connection must not be positive
    if they skip to any hidden layer from the input.
    This is the kind of skip-connection that is commonly used in ICNNs.
    """

    def __init__(self, in_features: int, out_features: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Linear(in_features, out_features, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)


def get_model(
    name: str, num_hidden: int = 32, bad_init: bool = False, skip: bool = False
):
    num_in, num_out = 2, 1
    if name == "logreg":
        return nn.Linear(num_in, num_out)

    if name == "fc":
        positivity = NoPositivity()
    elif name == "convex":
        positivity = ExponentialPositivity()
    elif name == "icnn":
        positivity = LazyClippedPositivity()
    else:
        raise ValueError(f"unknown model name: {name}")

    model = nn.Sequential(
        nn.Linear(num_in, num_hidden),
        nn.ReLU(),
        ConvexLinear(num_hidden, num_hidden, positivity=positivity),
        nn.ReLU(),
        ConvexLinear(num_hidden, num_out, positivity=positivity),
    )

    init = (
        TraditionalInitialiser(gain=2.0)
        if name == "fc" or bad_init
        else ConvexInitialiser()
    )
    for idx in range(len(model)):
        if isinstance(model[idx], ConvexLinear):
            init(model[idx].weight, model[idx].bias)

    if skip:
        new_model = LinearSkip(num_in, num_hidden, model[1:5])
        new_model = LinearSkip(num_in, num_out, nn.Sequential(new_model, *model[5:]))
        model = nn.Sequential(model[0], new_model)

    return model
