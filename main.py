import os
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoConfig
from src.dataset import GoEmotionsDataset
from src.model import EmotionRoBERTa, EmotionDistilBERT
from src.utils import get_AdamW_with_LLRD

def main(params, exp_folder, data_folder, dev_run):
    print("Reading data...")
    data = pd.read_parquet(os.path.join(data_folder, params['dataset']))
    if dev_run:
        data = data.sample(frac=0.1)
        data = data.reset_index(drop=True)
    num_labels = data.iloc[:, 2:].shape[1]
    folds = data["fold"].unique()
    
    print('Params ID: ' + params["params_id"])
    for fold in folds:
        print(f"### Fold {fold} ###")
        
        train_set = GoEmotionsDataset(params["model_name"], data[data["fold"] != fold]["text"].values, data[data["fold"] != fold].iloc[:, 2:].values,
                                      params["additional_special_tokens"], params["padding"], params["truncation"], params["max_length"],
                                      params["return_tensors"])
        
        valid_set = GoEmotionsDataset(params["model_name"], data[data["fold"] == fold]["text"].values, data[data["fold"] == fold].iloc[:, 2:].values,
                                      params["additional_special_tokens"], params["padding"], params["truncation"], params["max_length"],
                                      params["return_tensors"])

        # Creating instances of training and validation dataloaders
        train_loader = DataLoader(train_set, batch_size=params["bs"], num_workers=os.cpu_count(), shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=params["bs"], num_workers=os.cpu_count(), shuffle=False, drop_last=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg = AutoConfig.from_pretrained(params["model_name"])
        if "roberta" in params["model_name"]:
            net = EmotionRoBERTa(model=params["model_name"], dropout=params["dropout"], hidden_dim=params["hidden_dim"],
                                hidden_size=cfg.hidden_size, labels=num_labels)
            
        elif "distilbert" in params["model_name"]:
            net = EmotionDistilBERT(model=params["model_name"], dropout=params["dropout"], hidden_dim=params["hidden_dim"],
                                hidden_size=cfg.hidden_size, labels=num_labels)
        else:
            raise NotImplementedError("Pretrained model not available yet")
        
        if train_set.num_added_tokens > 0:
            net.transformer_model.resize_token_embeddings(len(train_set.tokenizer))
        
        if params["do_reinit_layers"]:
            net._do_reinit(params["reinit_n_layers"])
        if params["freeze_pretrained"]:
            for param in net.transformer_model.parameters():
                param.requires_grad = False

        net.to(device)

        criterion = nn.BCEWithLogitsLoss().to(device)
        opti = get_AdamW_with_LLRD(net, num_hidden_layers=, params["init_lr"], params["init_lr"], params["layer_mult_factor"], params["weight_decay"])
        num_training_steps = params["epochs"] * len(train_loader)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=params["num_warmup_steps_perc"] * num_training_steps, num_training_steps=num_training_steps)
        train(net, criterion, opti, lr_scheduler, train_loader, valid_loader, device, params.copy(), data_labels, fold)
        

if __name__ == "__main__":
    params = dict(
        dataset = ["goemotions_folds"],
        model_name = ["roberta-base"],
        bs = [32],
        epochs = [4],
        head_lr = [5e-5],
        init_lr = [5e-5],
        reinit_n_layers = [0],
        do_reinit_layers = [False],
        layer_mult_factor = [0.9],
        weight_decay = [1e-2],
        num_warmup_steps_perc = [0.05],
        freeze_pretrained = [False],
        additional_special_tokens = [['[NAME]', '[RELIGION]']],
        padding = ["max_length"],
        truncation = [True],
        pos_weights = [False],
        return_tensors = ["pt"],
        max_length = [50],
        hidden_dim = [50],
        hidden_size = [768],
        dropout = [0.35],
    )