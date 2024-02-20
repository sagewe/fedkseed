## FedKseed

This repo is a temp repo for the project to properly implement the FedKseed algorithm in transformers.
The original paper is [here](https://arxiv.org/pdf/2312.06353.pdf).

This project is still working in progress and the code is not ready for use yet.

## Quick Start

use the following command to run the training script without k-seed optimization:

```bash
python3 scripts/cli.py --output_dir tmp_trainer
```

use the following command to run the training script with k-seed optimization:

```bash
python3 scripts/cli.py --output_dir tmp_trainer --zo_optim=False
```
