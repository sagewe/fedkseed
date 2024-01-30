import rich


def generate_prompt_sql(input, context, output=""):
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input}

### Context:
{context}

### Response:
{output}"""


def main(
    device="mps",
    lr=1e-4,
    eps=0.0005,
    weight_decay=1e-2,
    grad_clip=-100.0,
    num_epochs=16,
    num_clients=20,
    client_sample_size=5,
    batch_size=64,
    init_grad_projected_value=0.0,
):
    import copy

    import torchvision
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    import numpy as np

    from fedkseed.kseed import KSeedServer
    from fedkseed.optimizer import KSeedZerothOrderOptimizer

    device = torch.device(device)

    from fedkseed.cnn import Net

    model_w0 = Net()
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )
    data = torchvision.datasets.MNIST(root="data", download=True, transform=transform, train=True)
    eval_data = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=batch_size)

    k_seed_server = KSeedServer.build(
        k=4096,
        init_grad_projected_value=init_grad_projected_value,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    _evaluate_model_cnn(model_w0, eval_dataloader, device)
    clients = list(range(num_clients))
    from rich.progress import Progress

    with Progress() as progress:
        round_task = progress.add_task("Round", total=num_epochs)
        for round_number in range(num_epochs):
            candidate_seeds_probabilities = k_seed_server.get_candidate_seeds_with_probabilities()
            round_model = k_seed_server.build_model(copy.deepcopy(model_w0))

            client_task = progress.add_task("Client", total=client_sample_size)
            for client_id in np.random.choice(clients, client_sample_size, replace=False):
                client_model = copy.deepcopy(round_model)
                optimizer = KSeedZerothOrderOptimizer.from_model(
                    client_model,
                    candidate_seed_probabilities=candidate_seeds_probabilities,
                    lr=lr,
                    eps=eps,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                )
                # progress_bar = tqdm(range(len(train_dataloader)))
                client_model.eval()

                loss_total_train = 0.0
                num_trained = 0
                train_task = progress.add_task(f"Training client {client_id}", total=len(train_dataloader))
                for data, target in train_dataloader:
                    data, target = data.to(device), target.to(device)

                    def closure():
                        output = client_model(data)
                        loss = torch.nn.functional.nll_loss(output, target)

                        return loss.detach().item()

                    loss = optimizer.step(closure)

                    if loss is not None:
                        loss_total_train += loss
                        num_trained += len(data)
                        # train_task.
                        # progress_bar.set_description(
                        #     f"client {client_id} train at round {round_number}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                        # )
                    progress.update(train_task, advance=1, description=f"[Training client {client_id}]loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}")

                _evaluate_model_cnn(client_model, eval_dataloader, device)

                k_seed_server.update_history(optimizer.directional_derivative_history)
                progress.update(client_task, advance=1)
            progress.update(round_task, advance=1)

    # model.to(device)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

    # for epoch in range(1, num_epochs + 1):
    #     model.train()
    #     for batch_idx, (data, target) in enumerate(train_dataloader):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = torch.nn.functional.nll_loss(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         if batch_idx % 100 == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 0, batch_idx * len(data), len(train_dataloader.dataset),
    #                 100. * batch_idx / len(train_dataloader), loss.item()))
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data, target in train_dataloader:
    #             data, target = data.to(device), target.to(device)
    #             output = model(data)
    #             test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()
    #     test_loss /= len(train_dataloader.dataset)
    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(train_dataloader.dataset),
    #         100. * correct / len(train_dataloader.dataset)))


# def load_cnn():
#     from fedkseed.cnn import Net
#
#     model = Net()
#     return model
#
#
# def load_data(batch_size=8):
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#     dataset = datasets.load_dataset("yelp_review_full")
#
#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
#     tokenized_dataset = tokenized_dataset.remove_columns(["text"])
#     tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
#     tokenized_dataset.set_format("torch")
#     train_test_tokenized_dataset = tokenized_dataset["train"].train_test_split(
#         train_size=0.01, shuffle=True, seed=42
#     )
#     train_dataloader = DataLoader(
#         train_test_tokenized_dataset["train"], shuffle=True, batch_size=batch_size
#     )
#     return train_dataloader


def _first_order_optimize(model, train_dataloader, device, num_epochs, lr, eps, weight_decay, grad_clip):
    import copy
    import torch
    from tqdm.auto import tqdm

    client_model = copy.deepcopy(model)
    print(_evaluate_model_cnn(client_model, train_dataloader, device))
    client_model.to(device)
    client_model.train()
    optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(len(train_dataloader)))
    for round_number in range(num_epochs):
        loss_total_train = 0.0
        num_trained = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = client_model(**batch)
            loss = outputs.loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), grad_clip)
            optimizer.step()
            loss_total_train += loss
            num_trained += len(batch["input_ids"])
            progress_bar.set_description(
                f"client train at epoch {round_number}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
            )
            progress_bar.update(1)
        print(_evaluate_model(client_model, train_dataloader, device))


def _zeroth_order_optimize(model, train_dataloader, device, num_epochs, lr, eps, weight_decay, grad_clip):
    from fedkseed.kseed import KSeedServer, KSeed
    from fedkseed.optimizer import configure_optimizers
    import copy
    import torch
    from tqdm.auto import tqdm

    k_seed_server = KSeedServer.build(5)
    probabilities = k_seed_server.calculate_probabilities()
    print(_evaluate_model(model, train_dataloader, device))
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    client_model = copy.deepcopy(model)
    for round_number in range(num_epochs):
        client_model.to(device)
        client_model.eval()
        k_seed = KSeed(k_seed_server.candidate_seeds, probabilities)
        optimizer = configure_optimizers(client_model, k_seed, lr, eps, weight_decay, grad_clip)
        loss_total_train = 0.0
        num_trained = 0
        for batch in train_dataloader:
            logits, loss = optimizer.step(_get_closure(client_model, batch, device))

            if (not torch.isnan(loss)) and loss != 0.0:
                loss_total_train += loss
                num_trained += len(batch["input_ids"])
                progress_bar.set_description(
                    f"client train at epoch {round_number}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                )
            progress_bar.update(1)

        print(_evaluate_model(client_model, train_dataloader, device))


def _evaluate_model(model, eval_dataloader, device):
    from tqdm.auto import tqdm
    import torch
    import evaluate

    metric = evaluate.load("accuracy")

    model.eval()
    model.to(device)
    progress_bar_eval = tqdm(range(len(eval_dataloader)))
    with torch.inference_mode():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            references = batch["labels"]
            metric.add_batch(predictions=predictions, references=references)
            progress_bar_eval.update(1)
    return metric.compute()


def _evaluate_model_cnn(model, eval_dataloader, device):
    from tqdm.auto import tqdm
    import torch
    import evaluate

    metric = evaluate.load("accuracy")

    model.eval()
    model.to(device)
    progress_bar_eval = tqdm(range(len(eval_dataloader)))
    with torch.inference_mode():
        for data, target in eval_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)
            metric.add_batch(predictions=predictions, references=target)
            progress_bar_eval.update(1)

    accuracy = metric.compute()
    progress_bar_eval.set_description(f"accuracy: {accuracy}")
    return accuracy


def _get_closure(model, batch, device):
    def closure():
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        return outputs.logits.detach(), outputs.loss.detach()

    return closure


if __name__ == "__main__":
    import fire

    fire.Fire(main)
