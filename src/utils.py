import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AdamW


def parse_param_dict_for_dumping(params):
    return {
        (k): (v if type(v) not in [list, dict] else json.dumps(v))
        for k, v in params.items()
    }


def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_label_th(yt, yp, num_folds=5, start=0.1, end=0.95, step=0.05):
    tmp = pd.DataFrame({"yt": yt, "yp": yp})
    tmp = tmp.sample(frac=1)
    tmp = tmp.reset_index(drop=True)
    tmp["fold"] = tmp.index.values % num_folds
    chosen_ths = []
    for fold in range(num_folds):
        fold_yt = tmp[tmp.fold == fold].yt.values
        fold_yp = tmp[tmp.fold == fold].yp.values
        best_th = 0
        best_th_f1 = 0
        for this_th in np.arange(start, end, step):
            this_th = float(this_th)
            this_th_f1 = f1_score(fold_yt, (fold_yp > this_th).astype(int))
            if this_th_f1 > best_th_f1:
                best_th = this_th
                best_th_f1 = this_th_f1
        chosen_ths.append(best_th)
    avg_th = np.mean(chosen_ths)
    return avg_th


def get_AdamW_with_LLRD(model, num_hidden_layers, init_lr, mult_factor, weight_decay):
    """Layer-wise Learning Rate Decay AdamW"""
    opt_parameter_groups = []
    named_parameters = list(model.named_parameters())
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "layer_norm.bias",
    ]

    # === Pooler and regressor ======================================================
    head_no_weight_decay_params = [
        p
        for n, p in named_parameters
        if ("classifier" in n or "pooler" in n) and any(nd in n for nd in no_decay)
    ]
    head_weight_decay_params = [
        p
        for n, p in named_parameters
        if ("classifier" in n or "pooler" in n) and not any(nd in n for nd in no_decay)
    ]

    head_no_weight_decay_opt_param_group = {
        "params": head_no_weight_decay_params,
        "lr": init_lr,
        "weight_decay": 0.0,
    }
    opt_parameter_groups.append(head_no_weight_decay_opt_param_group)

    head_weight_decay_opt_param_group = {
        "params": head_weight_decay_params,
        "lr": init_lr,
        "weight_decay": weight_decay,
    }
    opt_parameter_groups.append(head_weight_decay_opt_param_group)

    # === N Hidden layers ==========================================================
    lr = init_lr
    for layer in range(num_hidden_layers - 1, -1, -1):
        encoder_no_weight_decay_params = [
            p
            for n, p in named_parameters
            if (f"encoder.layer.{layer}." in n or f"transformer.layer.{layer}." in n)
            and any(nd in n for nd in no_decay)
        ]
        encoder_weight_decay_params = [
            p
            for n, p in named_parameters
            if (f"encoder.layer.{layer}." in n or f"transformer.layer.{layer}." in n)
            and not any(nd in n for nd in no_decay)
        ]

        encoder_no_weight_decay_opt_param_group = {
            "params": encoder_no_weight_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
        }
        opt_parameter_groups.append(encoder_no_weight_decay_opt_param_group)

        encoder_weight_decay_opt_param_group = {
            "params": encoder_weight_decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        opt_parameter_groups.append(encoder_weight_decay_opt_param_group)

        lr *= mult_factor  # decay lr in reverse so that first layers and embeddings weights update less

    # === Embeddings layer ==========================================================

    embeddings_no_weight_decay_params = [
        p
        for n, p in named_parameters
        if "embeddings" in n and any(nd in n for nd in no_decay)
    ]
    embeddings_weight_decay_params = [
        p
        for n, p in named_parameters
        if "embeddings" in n and not any(nd in n for nd in no_decay)
    ]

    embeddings_no_weight_decay_opt_param_group = {
        "params": embeddings_no_weight_decay_params,
        "lr": lr,
        "weight_decay": 0.0,
    }
    opt_parameter_groups.append(embeddings_no_weight_decay_opt_param_group)

    embeddings_weight_decay_opt_param_group = {
        "params": embeddings_weight_decay_params,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    opt_parameter_groups.append(embeddings_weight_decay_opt_param_group)

    return AdamW(opt_parameter_groups, lr=init_lr)
