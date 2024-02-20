import math
from typing import Mapping, Optional, Callable, Tuple, List

import torch
from torch.optim import Optimizer

from fedkseed.pytorch_utils import get_optimizer_parameters_grouped_with_decay


class RandomWalkOptimizer(Optimizer):
    """
    Random Walk Optimizer

    This optimizer performs a `random` walk update for the parameters of the model.
    """

    def __init__(self, params, lr, weight_decay, grad_clip, defaults=None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        if defaults is None:
            defaults = dict(lr=lr, weight_decay=weight_decay)
        else:
            defaults = dict(defaults)
            defaults.update(lr=lr, weight_decay=weight_decay)
        super(RandomWalkOptimizer, self).__init__(params, defaults)

    @classmethod
    def from_model(cls, model, lr, weight_decay, grad_clip, **kwargs):
        optimizer_grouped_parameters = get_optimizer_parameters_grouped_with_decay(model, weight_decay)
        kwargs["lr"] = lr
        kwargs["weight_decay"] = weight_decay
        kwargs["grad_clip"] = grad_clip
        return cls(optimizer_grouped_parameters, **kwargs)

    def directional_derivative_step(
        self, directional_derivative_seed: int, directional_derivative_value: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        perform a step update for the parameters of the model
        along the random direction z with the learning rate lr and the step size grad_projected_value

        Input:
        - closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if self.grad_clip > 0.0:
            if abs(directional_derivative_value) > self.grad_clip:
                return torch.FloatTensor([torch.nan])

        torch.manual_seed(directional_derivative_seed)
        for param_group in self.param_groups:
            weight_decay = param_group["weight_decay"]
            for param in param_group["params"]:
                z = torch.normal(
                    mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype
                )
                if weight_decay is not None:
                    param.data = param.data - self.lr * (directional_derivative_value * z + weight_decay * param.data)
                else:
                    param.data = param.data - self.lr * (directional_derivative_value * z)

        return directional_derivative_value

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        raise NotImplementedError(
            "use random_step instead of step for RandomWalkOptimizer \
            since we need pass the `seed` and `grad_projected_value`"
        )


class ZerothOrderOptimizer(RandomWalkOptimizer):
    def __init__(self, params, lr, eps, weight_decay, grad_clip):
        self.eps = eps
        defaults = dict(eps=eps)
        super(ZerothOrderOptimizer, self).__init__(params, lr, weight_decay, grad_clip, defaults)

    def zeroth_order_step(
        self, directional_derivative_seed: int, closure: Callable[[], torch.FloatTensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        perform a step update for the parameters of the model along the
        random direction z generated by the `directional_derivative_seed`
        with the learning rate lr
        and the step size of calculated namely `directional_derivative_value`

        Input:
        - directional_derivative_seed: the seed for generating the random direction z
        - closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Output:
        - directional_derivative_value: the gradient projected value
        - loss_right: the loss of the model with the perturbed parameters x + eps * z
        - loss_left: the loss of the model with the perturbed parameters x - eps * z
        """

        # x -> x + eps * z
        self.random_perturb_parameters(directional_derivative_seed, scaling_factor=1.0)
        loss_right = closure()

        # x + eps * z -> x - eps * z
        self.random_perturb_parameters(directional_derivative_seed, scaling_factor=-2.0)
        loss_left = closure()

        # x - eps * z -> x
        self.random_perturb_parameters(directional_derivative_seed, scaling_factor=1.0)

        if torch.isnan(loss_right):
            return loss_right, loss_right, loss_left
        if torch.isnan(loss_left):
            return loss_left, loss_right, loss_left

        # ∇f(x) · z = D_z f(x) ≈ (f(x + eps * z) - f(x - eps * z)) / (2 * eps)
        directional_derivative_value = (loss_right - loss_left) / (2 * self.eps)
        # perform update for the random direction z * grad_projected_value
        self.directional_derivative_step(directional_derivative_seed, directional_derivative_value)

        return directional_derivative_value, loss_right, loss_left

    def random_perturb_parameters(self, directional_derivative_seed: int, scaling_factor: float):
        """
        Perturb the parameters with random direction z generated by the directional_derivative_seed

        for each parameter theta, the update is theta = theta + scaling_factor * z * eps

        Input:
        - seed: the seed for generating the random direction z
        - scaling_factor: the scaling factor for the random direction z

        Output:
        - None
        """
        torch.manual_seed(directional_derivative_seed)
        for param_group in self.param_groups:
            eps = param_group["eps"]
            for param in param_group["params"]:
                if param.requires_grad:
                    z = torch.normal(
                        mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype
                    )
                    param.data = param.data + scaling_factor * eps * z


class KSeedZerothOrderOptimizer(ZerothOrderOptimizer):
    def __init__(self, params, seed_candidates: torch.LongTensor, seed_probabilities: torch.FloatTensor, lr, eps, weight_decay, grad_clip):
        self.seed_candidate = seed_candidates
        self.seed_probabilities = seed_probabilities
        self.directional_derivative_history: Mapping[int, List[float]] = {seed.item(): [] for seed in seed_candidates}
        self.sample_random_generator = torch.Generator()
        super(KSeedZerothOrderOptimizer, self).__init__(params, lr, eps, weight_decay, grad_clip)

    def sample(self) -> int:
        sampled = torch.multinomial(
            input=self.seed_probabilities,
            num_samples=1,
            generator=self.sample_random_generator,
        )[0].item()
        return self.seed_candidate[sampled].item()

    def step(self, closure: Callable[[], torch.FloatTensor] = None) -> torch.FloatTensor:
        if closure is None:
            # closure is required for the zeroth_order_step, but we
            # don't raise an error here to maintain compatibility with
            # the third-party tools that use the `step` method without
            # providing the closure in training loop, e.g., HuggingFace Transformers
            return torch.FloatTensor([torch.nan])
        return self.kseed_zeroth_order_step(closure)

    def kseed_zeroth_order_step(self, closure: Callable[[], torch.FloatTensor]) -> torch.FloatTensor:
        """
        Performs a single optimization step.

        1. Sample a random seed for sampling z
        2. Perturb the parameters with the random direction(-z * eps, z * eps) for evaluating the model on the batch, and compute the loss(loss1, loss2)
        3. Compute the directional derivative value: grad_projected_value = (loss_right - loss_left) / (2 * eps)
        4. Perform the directional derivative step update for the parameters of the model along the random direction z with the learning rate lr and the step size grad_projected_value


        Input:
        - closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise ValueError("closure must not be None")

        # sample the random seed for sampling z for perturbing parameters.
        seed = self.sample()
        directional_derivative_value, loss_right, loss_left = self.zeroth_order_step(seed, closure)
        if math.isnan(directional_derivative_value):
            return directional_derivative_value

        # record the directional_derivative_value for the seed
        self.directional_derivative_history[seed].append(directional_derivative_value.item())

        return loss_right  # TODO: return loss_left or loss_right or average of both?
