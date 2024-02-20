from typing import List, Mapping

import torch
from fedkseed.zo_utils import probability_from_amps, directional_derivative_step
from fedkseed.pytorch_utils import get_optimizer_parameters_grouped_with_decay


class KSeedServer:
    def __init__(
        self,
        candidate_seeds: torch.LongTensor = None,
        lr=1e-4,
        k=4096,
        low=1,
        high=100000000000,
        bias_loss_clip=0.5,
        init_grad_projected_value=0.0,
        weight_decay=0.0,
        grad_clip=0.0,
    ):
        if candidate_seeds is None:
            candidate_seeds = torch.randint(low, high, (k,))

        self.candidate_seeds = candidate_seeds
        self.bias_loss_clip = bias_loss_clip
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.grad_projected_value_history = {seed: [init_grad_projected_value] for seed in self.candidate_seeds}
        self.grad_projected_value_agg = {seed: 0.0 for seed in self.candidate_seeds}
        self.has_history = False

    def get_seed_candidates(self) -> torch.LongTensor:
        return self.candidate_seeds

    def get_seed_probabilities(self):
        if self.has_history:
            amps = [self.grad_projected_value_history[seed] for seed in self.candidate_seeds]
            probabilities = probability_from_amps(amps, self.bias_loss_clip)
        else:
            k = len(self.candidate_seeds)
            probabilities = torch.ones(k, dtype=torch.float) / k

        return probabilities

    def update_history(self, directional_derivative_history: Mapping[int, List[float]]):
        for seed in self.candidate_seeds:
            self.grad_projected_value_history[seed].extend(directional_derivative_history[seed])
            self.grad_projected_value_agg[seed] += sum(directional_derivative_history[seed])
        self.has_history = True

    def model_update_reply(self, model):
        """
        Update the model with the aggregated directional derivative values

        Warning: when learning rate scheduler is used in client side, this method will not work properly.
        FIXME: Maybe we should control the learning rate in the server side.
        """
        if self.has_history:
            param_groups = get_optimizer_parameters_grouped_with_decay(model, self.weight_decay)
            for seed, grad in self.grad_projected_value_agg.items():
                if grad != 0.0:
                    directional_derivative_step(param_groups, seed, grad, lr=self.lr, weight_decay=self.weight_decay)
        return model
