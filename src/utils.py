import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AdamW


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def find_label_th(yt, yp, num_folds=1, start=0.02, end=0.99, step=0.01):
    tmp = pd.DataFrame({"yt": yt, "yp": yp})
    tmp = tmp.sample(frac=1)
    tmp = tmp.reset_index(drop=True)
    tmp["fold"] = tmp.index.values % num_folds
    chosen_ths = []
    for fold in range(num_folds):
        fold_yt = tmp[tmp.fold == fold].yt.values
        fold_yp = tmp[tmp.fold == fold].yp.values
        best_th = 0.3
        best_th_f1 = -1
        for this_th in np.arange(start, end, step):
            this_th = float(this_th)
            this_th_f1 = f1_score(fold_yt, (fold_yp > this_th).astype(int))
            if this_th_f1 > best_th_f1:
                best_th = this_th
                best_th_f1 = this_th_f1
        chosen_ths.append(best_th)
    avg_th = np.mean(chosen_ths)
    return avg_th


def get_AdamW_with_LLRD(model, num_hidden_layers, head_lr=3.6e-6, init_lr=3.5e-6, layer_mult_factor=0.9, weight_decay=0.01):
    """ Layer-wise Learning Rate Decay AdamW """
    opt_parameter_groups = []
    named_parameters = list(model.named_parameters()) 
       
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight", "layer_norm.bias"]
    lr = init_lr
    
    # === Pooler and regressor ======================================================  
    head_no_weight_decay_params = [p for n,p in named_parameters if ("classifier" in n or "pooler" in n) 
                and any(nd in n for nd in no_decay)]
    head_weight_decay_params = [p for n,p in named_parameters if ("classifier" in n or "pooler" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_no_weight_decay_opt_param_group = {"params": head_no_weight_decay_params, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameter_groups.append(head_no_weight_decay_opt_param_group)
        
    head_weight_decay_opt_param_group = {"params": head_weight_decay_params, "lr": head_lr, "weight_decay": weight_decay}    
    opt_parameter_groups.append(head_weight_decay_opt_param_group)
                
    # === N Hidden layers ==========================================================
    
    for layer in range(num_hidden_layers-1,-1,-1):        
        encoder_no_weight_decay_params = [p for n,p in named_parameters if (f"encoder.layer.{layer}." in n or f"transformer.layer.{layer}." in n)
                    and any(nd in n for nd in no_decay)]
        encoder_weight_decay_params = [p for n,p in named_parameters if (f"encoder.layer.{layer}." in n or f"transformer.layer.{layer}." in n)
                    and not any(nd in n for nd in no_decay)]
        
        encoder_no_weight_decay_opt_param_group = {"params": encoder_no_weight_decay_params, "lr": lr, "weight_decay": 0.0}
        opt_parameter_groups.append(encoder_no_weight_decay_opt_param_group)   
                            
        encoder_weight_decay_opt_param_group = {"params": encoder_weight_decay_params, "lr": lr, "weight_decay": weight_decay}
        opt_parameter_groups.append(encoder_weight_decay_opt_param_group)       
        
        lr *= layer_mult_factor  # decay lr in reverse so that first layers dont change much   
        
    # === Embeddings layer ==========================================================
    
    embeddings_no_weight_decay_params = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    embeddings_weight_decay_params = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embeddings_no_weight_decay_opt_param_group = {"params": embeddings_no_weight_decay_params, "lr": lr, "weight_decay": 0.0} 
    opt_parameter_groups.append(embeddings_no_weight_decay_opt_param_group)
        
    embeddings_weight_decay_opt_param_group = {"params": embeddings_weight_decay_params, "lr": lr, "weight_decay": weight_decay} 
    opt_parameter_groups.append(embeddings_weight_decay_opt_param_group)        
    
    return AdamW(opt_parameter_groups, lr=init_lr)