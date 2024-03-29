import copy
import logging
from dataclasses import dataclass, field
from typing import List, Mapping

import torch
from fate.arch.context import Context

from fedkseed.pytorch_utils import get_optimizer_parameters_grouped_with_decay
from fedkseed.trainer import KSeedTrainingArguments
from fedkseed.trainer import KSeedZOExtendedTrainer
from fedkseed.zo_utils import probability_from_amps, directional_derivative_step, get_even_seed_probabilities

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, ctx: Context, seed_candidates: torch.LongTensor, model, args, data_collator, train_dataset, eval_dataset
    ):
        self.ctx = ctx
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.seed_candidates = seed_candidates
        self.k = len(seed_candidates)
        self.clients = ctx.hosts
        self.model = None

    def load_model(self):
        raise NotImplementedError

    def train(self):
        direction_derivative_history = {seed.item(): [self.args.grad_initial] for seed in self.seed_candidates}
        direction_derivative_sum = None
        seed_probabilities = None
        for aggregation_iter, sub_ctx in self.ctx.ctxs_range(self.args.num_aggregations):
            # step1: re-calculate sample probabilities for each seed
            if seed_probabilities is None:
                seed_probabilities = get_even_seed_probabilities(self.k)
            else:
                seed_probabilities = probability_from_amps(
                    [direction_derivative_history[seed.item()] for seed in self.seed_candidates],
                    self.args.bias_loss_clip,
                )

            # step2(rpc): remote call to the clients to get the directional derivative history
            # proposal
            for client in sub_ctx.hosts:
                client.put(
                    "train_once",
                    (
                        False,
                        {
                            "seed_candidates": self.seed_candidates,
                            "seed_probabilities": seed_probabilities,
                            "direction_derivative_sum": direction_derivative_sum,
                        },
                    ),
                )

            if direction_derivative_sum is None:
                direction_derivative_sum = {seed.item(): 0.0 for seed in self.seed_candidates}
            # wait for reply and update the directional derivative history
            for client in sub_ctx.hosts:
                client_directional_derivative_history = client.get("direction_derivative_history")
                for seed, history in client_directional_derivative_history.items():
                    # torch.LongTensor -> int
                    seed = int(seed)
                    if seed not in direction_derivative_history:
                        direction_derivative_history[seed] = []
                    direction_derivative_history[seed].extend(history)
                    direction_derivative_sum[seed] += sum(history)

            # step3: evaluate to get stopping condition if necessary
            if self.should_stop():
                break

    def should_stop(self):
        return False

    def evaluate(self):
        pass


class ClientTrainer:
    def __init__(self, ctx: Context, model, args, train_dataset, eval_dataset, data_collator, tokenizer):
        self.ctx = ctx
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.weight_decay = args.weight_decay
        self.model_0 = model

    def serve_loop(self):
        for i, sub_ctx in self.ctx.ctxs_range(self.args.num_aggregations):
            # step1: wait for the server to send the seed candidates and probabilities or exit signal
            logger.info(f"training loop started: {i}")
            should_exit, kwargs = sub_ctx.arbiter.get("train_once")
            seed_candidates = kwargs["seed_candidates"]
            seed_probabilities = kwargs["seed_probabilities"]
            direction_derivative_sum = kwargs["direction_derivative_sum"]
            logger.info(
                f"should_exit: {should_exit}, seed_candidates: {seed_candidates}, seed_probabilities: {seed_probabilities}"
            )
            if should_exit:
                break

            # step2: start the training loop
            direction_derivative_history = self.train_once(
                seed_candidates, seed_probabilities, direction_derivative_sum
            )

            # step3: send the directional derivative history to the server
            sub_ctx.arbiter.put("direction_derivative_history", direction_derivative_history)

    def train_once(self, seed_candidates, seed_probabilities, direction_derivative_sum) -> Mapping[int, List[float]]:
        # build model
        model = copy.deepcopy(self.model_0)
        model.to(self.args.device)
        if direction_derivative_sum is not None:
            param_groups = get_optimizer_parameters_grouped_with_decay(model, self.weight_decay)
            for seed, grad in direction_derivative_sum.items():
                if grad != 0.0:
                    directional_derivative_step(
                        param_groups, seed, grad, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
                    )

        # train
        trainer = KSeedZOExtendedTrainer(
            model=model,
            args=self.args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        trainer.configure_seed_candidates(seed_candidates, seed_probabilities)
        trainer.train()
        logger.info(f"evaluate: {trainer.evaluate()}")
        # get directional derivative history
        return trainer.get_directional_derivative_history()


@dataclass
class FedKSeedTrainingArguments(KSeedTrainingArguments):
    num_aggregations: int = field(default=10, metadata={"help": "The number of aggregations to perform."})
    bias_loss_clip: float = field(default=1000.0, metadata={"help": "The bias loss clip value."})
    grad_initial: float = field(
        default=0.0, metadata={"help": "The initial value for the directional derivative history."}
    )
