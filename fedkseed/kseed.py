from typing import List, Mapping

import numpy as np
from fedkseed.optimizer import RandomWalkOptimizer


class KSeedServer:
    def __init__(
        self,
        candidate_seeds: List[int],
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
        candidate_seeds = np.random.randint(low, high, k)
        return KSeedServer(
            list(candidate_seeds), learning_rate, weight_decay, grad_clip, bias_loss_clip, init_grad_projected_value
        )

    def get_candidate_seeds_with_probabilities(self):
        if self.has_history:
            history_list = [self.grad_projected_value_history[seed] for seed in self.candidate_seeds]
            mean_grad_history = np.array(
                [
                    np.mean(np.abs(np.clip(history_cur_seed, -self.bias_loss_clip, self.bias_loss_clip)))
                    for history_cur_seed in history_list
                ]
            )

            def softmax(vec):
                vec = vec - np.max(vec)
                exp_x = np.exp(vec)
                softmax_x = exp_x / np.sum(exp_x)
                return softmax_x

            def min_max_norm(vec):
                min_val = np.min(vec)
                return (vec - min_val) / (np.max(vec) + 1e-10 - min_val)

            probabilities = softmax(min_max_norm(mean_grad_history))
            sum_prob = np.sum(probabilities)
            if sum_prob != 1.0:
                probabilities /= sum_prob
        else:
            probabilities = np.array([1.0 / len(self.candidate_seeds) for _ in self.candidate_seeds])

        return {seed: prob for seed, prob in zip(self.candidate_seeds, probabilities)}

    def update_history(self, directional_derivative_history: Mapping[int, List[float]]):
        for seed in self.candidate_seeds:
            self.grad_projected_value_history[seed].extend(directional_derivative_history[seed])
            self.grad_projected_value_agg[seed] += sum(directional_derivative_history[seed])
        self.has_history = True

    def build_model(self, model):
        if self.has_history:
            optimizer = RandomWalkOptimizer.from_model(
                model, lr=self.lr, weight_decay=self.weight_decay, grad_clip=self.grad_clip
            )

            for seed, grad in self.grad_projected_value_agg.items():
                if grad != 0.0:
                    optimizer.directional_derivative_step(seed, grad)
        return model
