import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.utils import find_label_th


def evaluate(net, device, criterion, dataloader, col_names):
    net.eval()
    running_loss = 0
    count = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for it, (batch, labels) in enumerate(dataloader):
            count += 1
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            logits = net(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            logits = F.sigmoid(logits)
            for yp, yt in zip(
                logits.cpu().detach().numpy(), labels.cpu().detach().numpy()
            ):
                y_pred.append(yp)
                y_true.append(yt)
    net.train()
    y_true = np.stack(y_true, axis=0)
    y_pred = np.stack(y_pred, axis=0)
    loss = running_loss / count
    th_by_label = np.array(
        [
            find_label_th(yt=y_true[:, label], yp=y_pred[:, label])
            for label in range(len(col_names))
        ]
    )
    y_pred = (y_pred > th_by_label).astype(int)
    report = classification_report(
        y_true, y_pred, target_names=col_names, output_dict=True
    )
    return loss, report


def optimization_loop(
    net,
    criterion,
    opti,
    lr_scheduler,
    train_loader,
    val_loader,
    device,
    params,
    col_names,
    fold,
):
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 2  # print the training loss 2 times per epoch
    last_epoch_freq_eval_iters = params["freq_eval_iters"]
    last_epoch_eval_every = nb_iterations // last_epoch_freq_eval_iters

    epochs = params["epochs"]
    for ep in range(epochs):
        if ep == (epochs - 1):
            last_epoch_best_loss = val_loss
            last_epoch_best_class_report = class_report
            last_epoch_best_f1_score = class_report["macro avg"]["f1-score"]
        net.train()
        count = 0
        running_loss = 0.0
        for it, (batch, labels) in enumerate(tqdm(train_loader)):
            count += 1
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)

            opti.zero_grad()
            # Obtaining the logits from the model
            logits = net(inputs)
            # Computing loss
            loss = criterion(logits, labels)
            # Backpropagating the gradients
            loss.backward()
            opti.step()
            # Adjust the learning rate based on the number of iterations.
            lr_scheduler.step()
            running_loss += loss.item()
            if (it + 1) % print_every == 0:  # Print training loss information
                print(
                    "\nIteration {}/{} of epoch {} complete. Training loss : {}".format(
                        it + 1, nb_iterations, ep + 1, running_loss / count
                    )
                )
            if ep == (epochs - 1) and (it + 1) % last_epoch_eval_every == 0:
                val_loss, class_report = evaluate(
                    net, device, criterion, val_loader, col_names
                )  # Compute validation loss
                print(
                    "\nFinal Epoch, validation Loss: {} / Macro F1 score: {} (FREQ EVAL ITER)".format(
                        val_loss, class_report["macro avg"]["f1-score"]
                    )
                )
                if class_report["macro avg"]["f1-score"] > last_epoch_best_f1_score:
                    last_epoch_best_loss = val_loss
                    last_epoch_best_class_report = class_report
                    last_epoch_best_f1_score = class_report["macro avg"]["f1-score"]
        if ep != (epochs - 1):
            val_loss, class_report = evaluate(
                net, device, criterion, val_loader, col_names
            )  # Compute validation loss
            print(
                "\nEpoch {}, validation Loss: {} / Macro F1 score: {}".format(
                    ep + 1, val_loss, class_report["macro avg"]["f1-score"]
                )
            )
            print(f"Epoch {ep+1}, fold {fold} complete!\n")
        else:
            print(
                "\nEpoch {}, validation Loss: {} / Macro F1 score: {}".format(
                    ep + 1, last_epoch_best_loss, last_epoch_best_f1_score
                )
            )
            print(f"Epoch {ep+1}, fold {fold} complete!\n")
            print("Saving run metrics..")
            report = pd.DataFrame(class_report).T
            report_save_path = "params_id{}__fold{}__f1macro{}.csv".format(
                params["params_id"],
                fold,
                last_epoch_best_class_report["macro avg"]["f1-score"],
            )
            report.to_csv(
                os.path.join(
                    os.path.curdir,
                    params["output_dir"],
                    "run_metrics",
                    report_save_path,
                ),
                index=True,
            )
    del loss
    torch.cuda.empty_cache()
