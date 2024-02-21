def load_model(model_args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True
    )


def load_data(dataset_args):
    from datasets import load_dataset
    from transformers.data.data_collator import DataCollatorWithPadding
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(dataset_args.tokenizer_name_or_path)

    return (
        tokenizer,
        load_dataset(dataset_args.dataset_name)["train"],
        None,
        DataCollatorWithPadding(tokenizer=tokenizer),
    )


def run_client(ctx, train_args, model_args, dataset_args):
    from fedkseed.fedkseed import ClientTrainer

    tokenizer, data, eval_data, collate_fn = load_data(dataset_args)

    model = load_model(model_args)
    trainer = ClientTrainer(ctx, model, train_args, data, eval_data, collate_fn, tokenizer)
    trainer.serve_loop()


def run_server(ctx, train_args, model_args, dataset_args):
    from fedkseed.fedkseed import Trainer
    from fedkseed.zo_utils import build_seed_candidates

    tokenizer, data, eval_data, collate_fn = load_data(dataset_args)
    seeds_candidates = build_seed_candidates(train_args.k, low=0, high=2**32)
    model = load_model(model_args)

    trainer = Trainer(ctx, seeds_candidates, model, train_args, data, eval_data, collate_fn)
    trainer.train()


def main():
    import logging
    from rich.logging import RichHandler

    logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

    from fedkseed.fedkseed import FedKSeedTrainingArguments
    from dataclasses import dataclass, field

    @dataclass
    class FedKSeedTrainingRunnerArguments(FedKSeedTrainingArguments):
        role: str = field(default="arbiter")

    @dataclass
    class ModelArguments:
        model_name_or_path: str = field(
            default="datajuicer/LLaMA-1B-dj-refine-150B",
            metadata={
                "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
            },
        )

    @dataclass
    class DatasetArguments:
        dataset_name: str = field(
            default="databricks/databricks-dolly-15k",
            metadata={"help": "The name of the dataset to use."},
        )
        tokenizer_name_or_path: str = field(
            default="datajuicer/LLaMA-1B-dj-refine-150B",
            metadata={"help": "The name of the tokenizer to use."},
        )

    from transformers import HfArgumentParser

    (fedkseed_train_args, fedkseed_model_args, fedkseed_dataset_args) = HfArgumentParser(
        (FedKSeedTrainingRunnerArguments, ModelArguments, DatasetArguments)
    ).parse_args_into_dataclasses()

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
        run_server(ctx, fedkseed_train_args, model_args=fedkseed_model_args, dataset_args=fedkseed_dataset_args)
    else:
        federation = FederationBuilder(
            federation_session_id=federation_session_id, party=("host", "10000"), parties=parties
        ).build_standalone(computing_session=computing_session)
        ctx = Context(federation=federation, computing=computing_session)
        run_client(ctx, fedkseed_train_args, model_args=fedkseed_model_args, dataset_args=fedkseed_dataset_args)


if __name__ == "__main__":
    main()
