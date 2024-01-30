import logging
from dataclasses import dataclass, field
from typing import Dict, Union, Any, Tuple
from typing import Optional, List, Callable

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, EvalPrediction, DataCollator
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.utils import add_start_docstrings

from fedkseed.optimizer import KSeedZerothOrderOptimizer

logger = logging.getLogger(__name__)


class KSeedZOExtendedTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Union["KSeedTrainingArguments", TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.args = args
        self._candidate_seed_probabilities = None
        self.kseed_optimizer = None

    @property
    def candidate_seed_probabilities(self):
        if self._candidate_seed_probabilities is None:
            raise ValueError("candidate_seed_probabilities is not set.")
        return self._candidate_seed_probabilities

    def set_candidate_seed_probabilities(self, candidate_seed_probabilities):
        self._candidate_seed_probabilities = candidate_seed_probabilities

    @staticmethod
    def backport_mode(args):
        if hasattr(args, "enable_kseed_optim") and args.enable_kseed_optim:
            return False
        return True

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        hook to do the step with KSeedZerothOrderOptimizer
        """
        if not KSeedZOExtendedTrainer.backport_mode(self.args):
            model.eval()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                # zeroth order optimization needs forward pass twice in an optimization step,
                # so we need to wrap the forward pass in a closure
                def closure() -> torch.FloatTensor:
                    return self.compute_loss(model, inputs, return_outputs=False)

                # we don't use step() method of KSeedZerothOrderOptimizer here
                # because `Trainer` wraps the optimizer that is subclass of `torch.optim.Optimizer` and
                # returns nothing from the step method
                loss = self.kseed_optimizer.kseed_zeroth_order_step(closure=closure)
            return loss.detach()
        else:
            return super().training_step(model, inputs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        hook to add KSeedZerothOrderOptimizer
        """
        if not KSeedZOExtendedTrainer.backport_mode(self.args):
            # some of the code is copied from Trainer.create_optimizer_and_scheduler
            opt_model = self.model
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = KSeedZerothOrderOptimizer(
                optimizer_grouped_parameters,
                candidate_seed_probabilities=self.candidate_seed_probabilities,
                lr=self.args.learning_rate,
                eps=self.args.eps,
                weight_decay=self.args.weight_decay,
                grad_clip=self.args.grad_clip,
            )
            # we need to keep the reference to the optimizer before wrapped by `Trainer`
            self.kseed_optimizer = self.optimizer
            # FIXME: lr_scheduler here may lead to a bug, because it's not used in KSeedZerothOrderOptimizer
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        else:
            super().create_optimizer_and_scheduler(num_training_steps)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class KSeedTrainingArguments(TrainingArguments):
    """
    Args:
        enable_kseed_optim (`bool`, *optional*, defaults to `False`):
    """

    enable_kseed_optim: bool = field(
        default=False, metadata={"help": "Whether to use KSeedZerothOrderOptimizer or not."}
    )
    eps: float = field(default=0.0005, metadata={"help": "Epsilon value for KSeedZerothOrderOptimizer."})
    grad_clip: float = field(default=-100.0, metadata={"help": "Gradient clip value for KSeedZerothOrderOptimizer."})
