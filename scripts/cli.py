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


def main(train_args):

    data, eval_data, collate_fn = load_data()
    model = load_model()

    from fedkseed.fedkseed import KSeedServer
    from fedkseed.trainer import KSeedZOExtendedTrainer

    trainer = KSeedZOExtendedTrainer(
        model=model, args=train_args, data_collator=collate_fn, train_dataset=data, eval_dataset=eval_data
    )

    if trainer.k_seed_zo_mode(train_args):
        init_grad_projected_value = 0.0
        k_seed_server = KSeedServer.build(
            k=4096,
            init_grad_projected_value=init_grad_projected_value,
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            grad_clip=train_args.grad_clip,
        )
        seeds_candidates = k_seed_server.get_seed_candidates()
        seed_probabilities = k_seed_server.get_seed_probabilities()
        trainer.configure_seed_candidates(seeds_candidates, seed_probabilities)

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    from transformers import HfArgumentParser
    from fedkseed.args import KSeedTrainingArguments

    (kseed_train_args,) = HfArgumentParser((KSeedTrainingArguments,)).parse_args_into_dataclasses()

    main(kseed_train_args)
