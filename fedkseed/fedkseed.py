from typing import List, Mapping

import torch


class KSeedServer:
    def __init__(
        self,
        candidate_seeds: torch.LongTensor,
        lr,
        weight_decay,
        grad_clip,
        bias_loss_clip=0.5,
        init_grad_projected_value=0.0,
    ):
        self.candidate_seeds = candidate_seeds
        self.bias_loss_clip = bias_loss_clip
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.grad_projected_value_history = {seed: [init_grad_projected_value] for seed in self.candidate_seeds}
        self.grad_projected_value_agg = {seed: 0.0 for seed in self.candidate_seeds}
        self.has_history = False

    @classmethod
    def build(
        cls,
        k,
        low=1,
        high=100000000000,
        bias_loss_clip=0.5,
        init_grad_projected_value=0.0,
        learning_rate=0.1,
        weight_decay=0.0,
        grad_clip=0.0,
    ):
        return KSeedServer(
            torch.randint(low, high, (k,)),
            learning_rate,
            weight_decay,
            grad_clip,
            bias_loss_clip,
            init_grad_projected_value,
        )

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
        from fedkseed.optimizer import RandomWalkOptimizer

        if self.has_history:
            optimizer = RandomWalkOptimizer.from_model(
                model, lr=self.lr, weight_decay=self.weight_decay, grad_clip=self.grad_clip
            )
            for seed, grad in self.grad_projected_value_agg.items():
                if grad != 0.0:
                    optimizer.directional_derivative_step(seed, grad)
        return model


def probability_from_amps(amps: List[List[float]], clip):
    """
    Get the probability distribution from the amplitude history

    formula: amp_i = clamp(amp_i, -clip, clip).abs().mean()
             amp_i = (amp_i - min(amp)) / (max(amp) - min(amp))
             prob_i = softmax(amp)_i

    :param amps: list of amplitude history
    :param clip: the clipping value
    :return:
    """
    amp = torch.stack([torch.Tensor(amp).clamp_(-clip, clip).abs_().mean() for amp in amps])
    return (amp - amp.min()).div_(amp.max() - amp.min() + 1e-10).softmax(0)
