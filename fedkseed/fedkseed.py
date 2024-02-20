from typing import List, Mapping

import torch
from fedkseed.zo_utils import probability_from_amps, directional_derivative_step, get_even_seed_probabilities
from fedkseed.pytorch_utils import get_optimizer_parameters_grouped_with_decay
from fedkseed.trainer import KSeedZOExtendedTrainer


class Trainer:
    def __init__(self, seed_candidates: torch.LongTensor, model, args, data_collator, train_dataset, eval_dataset):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.seed_candidates = seed_candidates
        self.k = len(seed_candidates)
        self.clients = None

    def train(self):
        direction_derivative_history = {}
        seed_probabilities = None
        for epoch in range(self.args.num_train_epochs):
            # step1: re-calculate sample probabilities for each seed
            if seed_probabilities is None:
                seed_probabilities = get_even_seed_probabilities(self.k)
            else:
                seed_probabilities = probability_from_amps(
                    [direction_derivative_history[seed] for seed in self.seed_candidates], self.args.bias_loss_clip
                )

            # step2(rpc): remote call to the clients to get the directional derivative history
            clients_direction_derivative_histories = self.clients_step(
                self.clients, self.seed_candidates, seed_probabilities, direction_derivative_history
            )

            # update the directional derivative history
            for client_directional_derivative_history in clients_direction_derivative_histories:
                for seed, history in client_directional_derivative_history.items():
                    if seed not in direction_derivative_history:
                        direction_derivative_history[seed] = []
                    direction_derivative_history[seed].extend(history)

            # step3: evaluate to get stopping condition if necessary
            if self.should_stop():
                break

    def should_stop(self):
        return False

    def clients_step(
        self, clients, seed_candidates, seed_probabilities, grad_projected_value_agg
    ) -> List[Mapping[int, List[float]]]:
        history = []
        for client in clients:
            self.proposal_once(client, seed_candidates, seed_probabilities, grad_projected_value_agg)
        for client in clients:
            directional_derivative_history = self.wait_for_reply(client)
            history.append(directional_derivative_history)
        return history

    def proposal_once(self, client, seed_candidates, seed_probabilities, grad_projected_value_agg):
        raise NotImplementedError

    def wait_for_reply(self, client):
        raise NotImplementedError

    def evaluate(self):
        pass


class ClientTrainer:
    def __init__(self, args, data_collator, train_dataset, eval_dataset):
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.weight_decay = args.weight_decay
        self.model_0 = None

    def get_model_0(self):
        if self.model_0 is None:
            self.model_0 = self.load_initial_model()
        return self.model_0.copy()

    def load_initial_model(self):
        raise NotImplementedError

    def wait_for_server(self):
        raise NotImplementedError

    def send_to_server(self, direction_derivative_history):
        pass

    def serve_loop(self):
        while True:
            # step1: wait for the server to send the seed candidates and probabilities or exit signal
            should_exit, args = self.wait_for_server()
            if should_exit:
                break
            seed_candidates = args.seed_candidates
            seed_probabilities = args.seed_probabilities
            grad_projected_value_agg = args.grad_projected_value_agg

            # step2: start the training loop
            direction_derivative_history = self.train_once(
                seed_candidates, seed_probabilities, grad_projected_value_agg
            )

            # step3: send the directional derivative history to the server
            self.send_to_server(direction_derivative_history)

    def train_once(
        self, seed_candidates, seed_probabilities, grad_projected_value_agg=None
    ) -> Mapping[int, List[float]]:
        # build model
        model = self.load_initial_model()
        if grad_projected_value_agg is not None:
            param_groups = get_optimizer_parameters_grouped_with_decay(model, self.weight_decay)
            for seed, grad in grad_projected_value_agg.items():
                if grad != 0.0:
                    directional_derivative_step(
                        param_groups, seed, grad, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
                    )
        # train
        trainer = KSeedZOExtendedTrainer(
            model=model,
            args=self.args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        trainer.configure_seed_candidates(seed_candidates, seed_probabilities)
        trainer.train()
        # get directional derivative history
        return trainer.get_directional_derivative_history()
