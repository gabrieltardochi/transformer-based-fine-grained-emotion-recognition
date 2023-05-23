import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup

from src.dataset import GoEmotionsDataset
from src.model import EmotionDistilBERT, EmotionRoBERTa
from src.train import optimization_loop
from src.utils import (get_AdamW_with_LLRD, parse_param_dict_for_dumping,
                       set_seed)


def main(params):
    set_seed(params["seed"])
    print("Reading data...")
    data = pd.read_parquet(
        os.path.join(
            params["data_dir"], f"{params['dataset']}_goemotions_folds.parquet"
        )
    )
    if params["dev_run"]:
        print(">>>>>>>>>>>>>> FAST DEV RUN <<<<<<<<<<<<<<")
        data = data.sample(frac=0.1)
    data = data.reset_index(drop=True)
    data_labels = data.columns.tolist()[2:]
    num_labels = len(data_labels)
    folds = sorted(data["fold"].unique())

    print("Params ID: " + params["params_id"])
    print(params)
    params_df = pd.DataFrame(parse_param_dict_for_dumping(params), index=[0])
    params_save_path = "{}.csv".format(params["params_id"])
    params_df.to_csv(
        os.path.join(
            os.path.curdir, params["output_dir"], "run_params", params_save_path
        ),
        index=False,
    )
    for fold in folds:
        print(f"### Fold {fold} ###")

        train_set = GoEmotionsDataset(
            params["model_name"],
            data[data["fold"] != fold]["text"].values,
            data[data["fold"] != fold].iloc[:, 2:].values,
            params["additional_special_tokens"],
            params["padding"],
            params["truncation"],
            params["max_length"],
            params["return_tensors"],
        )

        valid_set = GoEmotionsDataset(
            params["model_name"],
            data[data["fold"] == fold]["text"].values,
            data[data["fold"] == fold].iloc[:, 2:].values,
            params["additional_special_tokens"],
            params["padding"],
            params["truncation"],
            params["max_length"],
            params["return_tensors"],
        )

        # Creating instances of training and validation dataloaders
        train_loader = DataLoader(
            train_set,
            batch_size=params["batch_size"],
            num_workers=os.cpu_count(),
            shuffle=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=params["batch_size"],
            num_workers=os.cpu_count(),
            shuffle=False,
            drop_last=False,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg = AutoConfig.from_pretrained(params["model_name"])
        if "roberta" in params["model_name"]:
            net = EmotionRoBERTa(
                model=params["model_name"],
                dropout=params["dropout"],
                hidden_size=cfg.hidden_size,
                labels=num_labels,
            )

        elif "distilbert" in params["model_name"]:
            net = EmotionDistilBERT(
                model=params["model_name"],
                dropout=params["dropout"],
                hidden_size=cfg.hidden_size,
                labels=num_labels,
            )
        else:
            raise NotImplementedError("Pretrained model not available yet")

        if train_set.num_added_tokens > 0:
            net.transformer_model.resize_token_embeddings(len(train_set.tokenizer))

        if not params["do_not_reinit_layers"]:
            net._do_reinit(params["reinit_n_layers"])
        if params["freeze_pretrained"]:
            for param in net.transformer_model.parameters():
                param.requires_grad = False

        net.to(device)

        criterion = nn.BCEWithLogitsLoss().to(device)
        opti = get_AdamW_with_LLRD(
            net,
            num_hidden_layers=cfg.num_hidden_layers,
            init_lr=params["llrd_init_lr"],
            mult_factor=params["llrd_mult_factor"],
            weight_decay=params["weight_decay"],
        )
        num_training_steps = params["epochs"] * len(train_loader)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=opti,
            num_warmup_steps=params["warmup_steps_ratio"] * num_training_steps,
            num_training_steps=num_training_steps,
        )
        optimization_loop(
            net,
            criterion,
            opti,
            lr_scheduler,
            train_loader,
            valid_loader,
            device,
            params.copy(),
            data_labels,
            fold,
        )


if __name__ == "__main__":
    # Define the available command-line arguments
    parser = argparse.ArgumentParser(
        description="Script to run fine-tuning with cross validation for a set of parameters and a selected pretrained transformer model on GoEmotions"
    )

    parser.add_argument(
        "--dev-run",
        action="store_true",
        help="Flag to run in dev mode",
    )
    parser.add_argument(
        "--params-id",
        type=str,
        help="Identifier for the hyperparameter config being experimented",
        required=True,
    )
    parser.add_argument(
        "--data-dir", type=str, help="Path to the data directory", default="data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the data directory",
        default="experiments",
    )
    parser.add_argument(
        "--dataset", type=str, help='Either "raw" or "preprocessed"', default="raw"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Huggingface pretrained model name",
        default="distilbert-base-uncased",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=3)
    parser.add_argument(
        "--llrd-init-lr",
        type=float,
        help="Initial learning rate for the last layer and head in the LLRD",
        default=5e-5,
    )
    parser.add_argument(
        "--reinit-n-layers",
        type=int,
        help="Number of last transformer layers to reinitialize (ignored if do-not-reinit-layers is TRUE)",
        default=2,
    )
    parser.add_argument(
        "--do-not-reinit-layers",
        action="store_true",
        help="Flag to deactivate layers reinitialization",
    )
    parser.add_argument(
        "--llrd-mult-factor", type=float, help="LLRD multiplication factor", default=0.9
    )
    parser.add_argument("--weight-decay", type=float, help="Weight decay", default=1e-2)
    parser.add_argument(
        "--warmup-steps-ratio",
        type=float,
        help="Ratio of total steps to perform warmup",
        default=0.1,
    )
    parser.add_argument(
        "--freeze-pretrained",
        action="store_true",
        help="Flag to freeze pretrained layers",
    )
    parser.add_argument(
        "--additional-special-tokens",
        type=list,
        help="Additional special tokens",
        default=["[NAME]", "[RELIGION]"],
    )
    parser.add_argument(
        "--padding", type=str, help="Tokenizer padding strategy", default="max_length"
    )
    parser.add_argument(
        "--truncation",
        action="store_true",
        help="Flag to force tokenizer to truncate",
        default=True,
    )
    parser.add_argument(
        "--return-tensors",
        type=str,
        help="Tokenizer type of tensors to return",
        default="pt",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Tokenizer maximum sequence length in tokens",
        default=50,
    )
    parser.add_argument(
        "--freq-eval-iters",
        type=int,
        help="Number of times to evaluate on the last epoch",
        default=10,
    )
    parser.add_argument(
        "--dropout", type=float, help="Classifier dropout", default=0.35
    )
    parser.add_argument(
        "--seed", type=float, help="Seed used to make results reproducible", default=1
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the parsed arguments to a dictionary
    params = vars(args)
    main(params)
