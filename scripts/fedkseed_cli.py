def load_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x, labels=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)

            if labels is not None:
                loss = F.nll_loss(output, labels)
                return {"loss": loss, "logits": output}
            return {"logits": output}

    return Net()


def load_data():
    import torch
    import torchvision

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )
    data = torchvision.datasets.MNIST(root="data", download=True, transform=transform, train=True)
    eval_data = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"x": pixel_values, "labels": labels}

    return data, eval_data, collate_fn


def run_client(ctx, train_args):
    from fedkseed.fedkseed import ClientTrainer

    data, eval_data, collate_fn = load_data()

    model = load_model()
    trainer = ClientTrainer(ctx, model, train_args, data, eval_data, collate_fn)
    trainer.serve_loop()


def run_server(ctx, train_args):
    from fedkseed.fedkseed import Trainer
    from fedkseed.zo_utils import build_seed_candidates

    data, eval_data, collate_fn = load_data()
    seeds_candidates = build_seed_candidates(train_args.k, low=0, high=2**32)
    model = load_model()

    trainer = Trainer(ctx, seeds_candidates, model, train_args, data, eval_data, collate_fn)
    trainer.train()


def main():
    import logging
    from rich.logging import RichHandler

    logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

    from fedkseed.fedkseed import ClientTrainer, FedKSeedTrainingArguments
    from dataclasses import dataclass, field

    @dataclass
    class FedKSeedTrainingRunnerArguments(FedKSeedTrainingArguments):
        role: str = field(default="arbiter")

    from transformers import HfArgumentParser

    (fedkseed_train_args,) = HfArgumentParser((FedKSeedTrainingRunnerArguments,)).parse_args_into_dataclasses()

    from fate.arch import Context
    from fate.arch.computing import ComputingBuilder
    from fate.arch.federation import FederationBuilder

    computing_session_id = "computing"
    federation_session_id = "federation"
    parties = [("arbiter", "10000"), ("host", "10000")]
    computing_session = ComputingBuilder(computing_session_id=computing_session_id).build_standalone("standalone/")

    if fedkseed_train_args.role == "arbiter":
        federation = FederationBuilder(
            federation_session_id=federation_session_id, party=("arbiter", "10000"), parties=parties
        ).build_standalone(computing_session=computing_session)
        ctx = Context(federation=federation, computing=computing_session)
        run_server(ctx, fedkseed_train_args)
    else:
        federation = FederationBuilder(
            federation_session_id=federation_session_id, party=("host", "10000"), parties=parties
        ).build_standalone(computing_session=computing_session)
        ctx = Context(federation=federation, computing=computing_session)
        run_client(ctx, fedkseed_train_args)


if __name__ == "__main__":
    main()
