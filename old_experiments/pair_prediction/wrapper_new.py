import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import EmotionCausePairClassifierModel
from sklearn.model_selection import KFold, train_test_split
from data_loader import *
import importlib
import random
import wandb

from models import EmotionCausePairClassifierModel

def init_wandb(args, fold):
    return run

class Wrapper():
    def __init__(self, args):
        self.k = args.kfold
        self.seed = args.seed
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.l2_reg = args.l2_reg
        self.device = args.device
        self.max_convo_len = args.max_convo_len
        self.batch_size = args.batch_size
        self.threshold = args.threshold
        self.model = None
        self.criterion = None
        self.optimizer = None

    def run(self, args):
        cause_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
        emotion_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
        for fold_id in range(1, self.k + 1):
            wandb.init(
                project="mecpe_task1",
                config={
                "epochs":args.num_epochs,
                "lr":args.lr,
                "batch_size":args.batch_size,
                "threshold":args.threshold,
                "max_convo_len":args.max_convo_len,
                },
                entity='arefa2001',
                name="fold" + str(fold_id),
                reinit=True,
            )
            print("\n\n>>>>>>>>>>>>>>>>>>FOLD %d<<<<<<<<<<<<<<<<<<<<<<<<" % (fold_id))
            self.train_loader = build_train_data(args, fold_id)
            self.val_loader = build_inference_data(args, fold_id, data_type='valid')
            self.test_loader = build_inference_data(args, fold_id, data_type='test')

            train_losses = []
            val_losses = []
            train_accuracy_list = []
            val_accuracy_list = []
            val_precision_list = []
            val_recall_list = []
            val_f1_list = []
            # Store best f1 across all epochs
            best_val_f1 = None


            # Model, loss fn and optimizer
            self.model = EmotionCausePairClassifierModel(args)
            self.model.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss() # apply reduction = 'none'?
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            # Training and Validation loop
            for epoch in range(self.num_epochs):
                # Training
                training_epoch_loss, total_correct, total_samples = self.train(epoch)
                train_accuracy = total_correct / total_samples

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Train Loss: {training_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                train_losses.append(training_epoch_loss)
                train_accuracy_list.append(train_accuracy)

                # Evaluation
                val_epoch_loss, tp, fp, fn = self.evaluate(epoch)
                if (tp + fp + fn) == 0:
                    val_accuracy = 0.0
                else:
                    val_accuracy = (tp) / (tp + fp + fn)
                if (tp + fp) == 0:
                    val_precision = 0.0
                else:
                    val_precision = (tp)/(tp + fp)
                if (tp + fn) == 0:
                    val_recall = 0.0
                else:
                    val_recall = tp / (tp + fn)
                if (val_precision + val_recall) == 0:
                    val_f1 = 0
                else:
                    val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall)

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                val_losses.append(val_epoch_loss)
                val_accuracy_list.append(val_accuracy)
                val_precision_list.append(val_precision)
                val_recall_list.append(val_recall)
                val_f1_list.append(val_f1)

                # Store best f1 across all folds
                if best_val_f1 == None or val_f1 > best_val_f1:
                    best_val_f1 = val_f1

                print("\n>>>>>>>>>>>>>>EPOCH END<<<<<<<<<<<<<<<<")

            # Calculate mean of the validation metrics for this fold
            mean_accuracy = np.mean(val_accuracy_list)
            mean_precision = np.mean(val_precision_list)
            mean_recall = np.mean(val_recall_list)
            mean_f1 = np.mean(val_f1_list)

            print(f"\nFold %d" % (fold + 1))
            print(f"Mean Accuracy: {mean_accuracy:.4f}")
            print(f"Mean Precision: {mean_precision:.4f}")
            print(f"Mean Recall: {mean_recall:.4f}")
            print(f"Mean F1: {mean_f1:.4f}")
            print("\n>>>>>>>>>>>>>>>>>>FOLD END<<<<<<<<<<<<<<<<<<<<<<<<")

            wandb.finish()

            cause_aprfb['acc'].append(mean_accuracy)
            cause_aprfb['p'].append(mean_precision)
            cause_aprfb['r'].append(mean_recall)
            cause_aprfb['f'].append(mean_f1)
            cause_aprfb['b'].append(best_val_f1)

        return cause_aprfb, emotion_aprfb

    def train(self, epoch):
        train_epoch_loss = 0.
        total_correct = 0.
        total_samples = 0.
        with tqdm(total=len(self.train_loader)) as prog_bar:
            for step, batch in enumerate(self.train_loader, 1):#step=batch_idx, data=batch
                given_emotion_idxs_b, adj_b, convo_len_b, y_emotions_b, y_causes_b, y_mask_b, \
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b = batch

                batch_loss, correct = self.update(given_emotion_idxs_b, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b)
                total_samples += len(given_emotion_idxs_b)
                total_correct += correct
                train_epoch_loss += batch_loss
                prog_bar.set_description("Epoch: %d\tStep: %d\tTraining Loss: %0.4f" % (epoch+1, step, batch_loss))

                wandb.log({"epoch":epoch+1, "step_train_loss":batch_loss, "step_train_acc":correct/len(given_emotion_idxs_b)})
                prog_bar.update()
        return train_epoch_loss / len(self.train_loader), total_correct, total_samples

    def update(self, emotion_idxs_b, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b):
        self.model.train()
        y_preds = self.model(emotion_idxs_b, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b)
        y_causes_b = torch.tensor(y_causes_b, dtype=torch.float32).to(self.device)
        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        y_preds = y_preds.masked_select(y_mask_b)
        y_causes_b = y_causes_b.masked_select(y_mask_b)
        binary_y_preds = (y_preds > self.threshold).float()

        loss = self.criterion(y_preds, y_causes_b)
        loss = torch.sum(loss) / len(y_preds)

        correct = (binary_y_preds == y_causes_b).sum().item()
        # print(f"Correct = {correct}")

        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), correct

    def evaluate(self, epoch):
        val_epoch_loss = 0.
        total_samples = 0
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        with tqdm(total=len(self.val_loader)) as prog_bar:
            with torch.no_grad():
                for step, batch in enumerate(self.val_loader, 1):#step=batch_idx, data=batch
                    given_emotion_idxs_b, adj_b, convo_len_b, y_emotions_b, y_causes_b, y_mask_b,\
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b = batch
                    batch_loss, tp, fp, fn = self.update_val(given_emotion_idxs_b, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b)

                    total_samples += len(given_emotion_idxs_b)
                    tot_tp += tp
                    tot_fp += fp
                    tot_fn += fn
                    val_epoch_loss += batch_loss
                    if (tp + fp + fn) == 0:
                        val_accuracy = 0.0
                    else:
                        val_accuracy = (tp) / (tp + fp + fn)
                    if (tp + fp) == 0:
                        val_precision = 0.0
                    else:
                        val_precision = (tp)/(tp + fp)
                    if (tp + fn) == 0:
                        val_recall = 0.0
                    else:
                        val_recall = tp / (tp + fn)
                    if (val_precision + val_recall) == 0:
                        val_f1 = 0
                    else:
                        val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall)
                    prog_bar.set_description("Epoch: %d\tStep: %d\tValidation Loss: %0.4f" % (epoch+1, step, batch_loss))
                    prog_bar.update()
                    wandb.log({"epoch":epoch+1, "step_val_loss":batch_loss, "step_val_acc":val_accuracy, "step_val_precision":val_precision, "step_val_recall":val_recall, "step_val_f1":val_f1})
        return val_epoch_loss / len(self.val_loader), tot_tp.cpu(), tot_fp.cpu(), tot_fn.cpu()

    def update_val(self, emotion_idxs_b, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b):
        self.model.eval()
        y_preds = self.model(emotion_idxs_b, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b)
        y_causes_b = torch.tensor(y_causes_b, dtype=torch.float32).to(self.device)
        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        y_preds = y_preds.masked_select(y_mask_b)
        y_causes_b = y_causes_b.masked_select(y_mask_b)

        binary_y_preds = (y_preds > self.threshold).float()
        loss = self.criterion(y_preds, y_causes_b)
        loss = torch.sum(loss) / len(y_preds)
        print("y preds")
        print(y_preds)

        tp, fp, fn = self.tp_fp_fn(binary_y_preds, y_causes_b)
        return loss.item(), tp, fp, fn

    def tp_fp_fn(self, predictions, labels):
        print("predictions batch")
        print(predictions)
        print("labels batch")
        print(labels)

        predictions = predictions.to(torch.int)
        labels = labels.to(torch.int)

        tp = torch.sum(predictions & labels)
        fp = torch.sum(predictions & ~labels)
        fn = torch.sum(~predictions & labels)

        return tp, fp, fn

    # define functions for saving and loading models per fold

