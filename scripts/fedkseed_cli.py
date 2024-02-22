def load_model(model_args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, device_map="cpu", trust_remote_code=True
    )


INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)


def load_data(dataset_args):
    from datasets import load_dataset
    from transformers.data.data_collator import DataCollatorForLanguageModeling
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(dataset_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        # For some instructions there is an input that goes along with the instruction, providing context for the
        # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
        # some piece of information from it.  The response is that information to extract.  In other cases there is
        # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
        # born.
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec

    dataset = load_dataset(dataset_args.dataset_name, split="train").train_test_split(test_size=0.1)
    dataset = dataset.map(_add_text)
    dataset = dataset.map(tokenize_function, batched=True)

    return (
        tokenizer,
        dataset['train'],
        dataset['test'],
        DataCollatorForLanguageModeling(tokenizer=tokenizer),
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
    #model = load_model(model_args)
    #print(model)

    trainer = Trainer(ctx, seeds_candidates, None, train_args, data, eval_data, collate_fn)
    trainer.train()


def main():
    import logging
    from rich.logging import RichHandler

    logging.basicConfig(level="DEBUG", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

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
