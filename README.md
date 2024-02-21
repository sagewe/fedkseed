## FedKseed

This repo is a temp repo for the project to properly implement the FedKseed algorithm in transformers.
The original paper is [here](https://arxiv.org/pdf/2312.06353.pdf).

This project is still working in progress and the code is not ready for use yet.

## Quick Start

### local training

```bash
use the following command to run the training script without k-seed optimization:

```bash
python3 scripts/kseed_zo_cli.py --output_dir tmp_trainer
```

use the following command to run the training script with k-seed optimization:

```bash
python3 scripts/kseed_zo_cli.py --output_dir tmp_trainer
```

### federated learning

use the following command to run the fedkseed algorithm:

arbiter:

```bash
python3 scripts/fedkseed_cli.py --output_dir tmp_trainer --save_strategy no --learning_rate 1e-4 --role arbiter
```

client:

```bash
python3 scripts/fedkseed_cli.py --output_dir tmp_trainer --save_strategy no --learning_rate 1e-4 --role client
```

