import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from data_loader2 import *
import importlib
import random
import wandb

from models import EmotionCausePairExtractorModel

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
        pair_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
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
                name=f"fold{fold_id}",
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
            best_val_f1_c = None
            best_val_f1_e = None


            # Model, loss fn and optimizer
            self.model = EmotionCausePairExtractorModel(args)
            self.model.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean') # apply reduction = 'none'?
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            # Training and Validation loop
            for epoch in range(self.num_epochs):
                training_epoch_loss, total_correct, total_samples = self.train(epoch)
                train_accuracy_c = total_correct[0] / total_samples[0]
                train_accuracy_e = total_correct[1] / total_samples[0]

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Train Loss: {training_epoch_loss:.4f}, Train Accuracy Cause: {train_accuracy_c:.4f}, Train Accuracy Emotion: {train_accuracy_e:.4f}")
                train_losses.append(training_epoch_loss)
                train_accuracy_list.append([train_accuracy_c, train_accuracy_e])

                # Evaluation
                val_epoch_loss, tp, fp, fn = self.evaluate(epoch)
                # Cause
                if (tp[0] + fp[0] + fn[0]) == 0:
                    val_accuracy_c = 0.0
                else:
                    val_accuracy_c = (tp[0]) / (tp[0] + fp[0] + fn[0])
                if (tp[0] + fp[0]) == 0:
                    val_precision_c = 0.0
                else:
                    val_precision_c = (tp[0])/(tp[0] + fp[0])
                if (tp[0] + fn[0]) == 0:
                    val_recall_c = 0.0
                else:
                    val_recall_c = tp[0] / (tp[0] + fn[0])
                if (val_precision_c + val_recall_c) == 0:
                    val_f1_c = 0
                else:
                    val_f1_c = (2 * val_precision_c * val_recall_c) / (val_precision_c + val_recall_c)
                # Emotion
                if (tp[1] + fp[1] + fn[1]) == 0:
                    val_accuracy_e = 0.0
                else:
                    val_accuracy_e = (tp[1]) / (tp[1] + fp[1] + fn[1])
                if (tp[1] + fp[1]) == 0:
                    val_precision_e = 0.0
                else:
                    val_precision_e = (tp[1])/(tp[1] + fp[1])
                if (tp[1] + fn[1]) == 0:
                    val_recall_e = 0.0
                else:
                    val_recall_e = tp[1] / (tp[1] + fn[1])
                if (val_precision_e + val_recall_e) == 0:
                    val_f1_e = 0
                else:
                    val_f1_e = (2 * val_precision_e * val_recall_e) / (val_precision_e + val_recall_e)

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Val Loss: {val_epoch_loss:.4f}\nCause: Val Accuracy: {val_accuracy_c:.4f}, Val Precision: {val_precision_c:.4f}, Val Recall: {val_recall_c:.4f}, Val F1: {val_f1_c:.4f}\nEmotion: Val Accuracy: {val_accuracy_e:.4f} Val Precision: {val_precision_e:.4f}, Val Recall: {val_recall_e:.4f}, Val F1: {val_f1_e:.4f}")
                val_losses.append(val_epoch_loss)
                val_accuracy_list.append([val_accuracy_c, val_accuracy_e])
                val_precision_list.append([val_precision_c, val_precision_e])
                val_recall_list.append([val_recall_c, val_recall_e])
                val_f1_list.append([val_f1_c, val_f1_e])

                # Store best f1 across all folds
                if best_val_f1_c == None or val_f1_c > best_val_f1_c:
                    best_val_f1_c = val_f1_c
                if best_val_f1_e == None or val_f1_e > best_val_f1_e:
                    best_val_f1_e = val_f1_e

                print("\n>>>>>>>>>>>>>>EPOCH END<<<<<<<<<<<<<<<<")

            # Calculate mean of the validation metrics for this fold
            mean_accuracy = np.mean(val_accuracy_list, axis=0)
            mean_precision = np.mean(val_precision_list, axis=0)
            mean_recall = np.mean(val_recall_list, axis=0)
            mean_f1 = np.mean(val_f1_list,axis=0)

            print(f"\nFold %d" % (fold_id))
            print("Cause")
            print(f"Mean Accuracy: {mean_accuracy[0]:.4f}")
            print(f"Mean Precision: {mean_precision[0]:.4f}")
            print(f"Mean Recall: {mean_recall[0]:.4f}")
            print(f"Mean F1: {mean_f1[0]:.4f}")
            print("Emotion")
            print(f"Mean Accuracy: {mean_accuracy[1]:.4f}")
            print(f"Mean Precision: {mean_precision[1]:.4f}")
            print(f"Mean Recall: {mean_recall[1]:.4f}")
            print(f"Mean F1: {mean_f1[1]:.4f}")
            print("\n>>>>>>>>>>>>>>>>>>FOLD END<<<<<<<<<<<<<<<<<<<<<<<<")

            wandb.finish()

            cause_aprfb['acc'].append(mean_accuracy[0])
            cause_aprfb['p'].append(mean_precision[0])
            cause_aprfb['r'].append(mean_recall[0])
            cause_aprfb['f'].append(mean_f1[0])
            cause_aprfb['b'].append(best_val_f1[0])
            emotion_aprfb['acc'].append(mean_accuracy[1])
            emotion_aprfb['p'].append(mean_precision[1])
            emotion_aprfb['r'].append(mean_recall[1])
            emotion_aprfb['f'].append(mean_f1[1])
            emotion_aprfb['b'].append(best_val_f1[1])

        return cause_aprfb, emotion_aprfb

    def train(self, epoch):
        train_epoch_loss = 0.
        total_correct = [0., 0., 0.]
        total_samples = [0., 0.]
        with tqdm(total=len(self.train_loader)) as prog_bar:
            for step, batch in enumerate(self.train_loader, 1):#step=batch_idx, data=batch
                adj_b, convo_len_b, y_pairs_b, y_emotions_b, y_causes_b, y_mask_b, \
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b = batch

                batch_loss, correct_e, correct_c, correct_p, y_emotions_b_masked = self.update(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b)
                samples = len(y_emotions_b_masked)
                total_samples[0] += samples
                total_samples[1] += samples * samples
                total_correct[0] += correct_c
                total_correct[1] += correct_e
                total_correct[2] += correct_p
                train_epoch_loss += batch_loss
                prog_bar.set_description("Epoch: %d\tStep: %d\tTraining Loss: %0.4f" % (epoch+1, step, batch_loss))

                wandb.log({"epoch":epoch+1, "step_train_loss":batch_loss, "step_train_acc_c":correct_c/samples, "step_train_acc_e":correct_e/samples, "step_train_acc_p": correct_p/samples*samples})
                prog_bar.update()
        return train_epoch_loss / len(self.train_loader), total_correct, total_samples

    def update(self, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b):
        self.model.train()
        y_preds_e, y_preds_c = self.model(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b)
        y_causes_b = torch.tensor(y_causes_b, dtype=torch.float32).to(self.device)
        y_emotions_b = torch.tensor(y_emotions_b, dtype=torch.float32).to(self.device)
        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        preds_e = y_preds_e.masked_select(y_mask_b) # masked_select converts into 1d tensor
        preds_c = y_preds_c.masked_select(y_mask_b)
        y_causes_b = y_causes_b.masked_select(y_mask_b)
        y_emotions_b = y_emotions_b.masked_select(y_mask_b)
        binary_y_preds_e = (torch.sigmoid(preds_e) > self.threshold).float()
        binary_y_preds_c = (torch.sigmoid(preds_c) > self.threshold).float()

        loss_e = self.criterion(preds_e, y_emotions_b)
        loss_c = self.criterion(preds_c, y_causes_b)
        loss = loss_e + loss_c
        # loss_e = torch.sum(loss_e) / len(y_preds_e)
        # loss_c = torch.sum(loss_c) / len(y_preds_c)

        correct_c = (binary_y_preds_c == y_causes_b).sum().item()
        correct_e = (binary_y_preds_e == y_emotions_b).sum().item()
        # print(f"Correct = {correct}")

        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        return loss.item(), correct_e, correct_c, 0., y_emotions_b

    def evaluate(self, epoch):
        val_epoch_loss = 0.
        total_samples = [0, 0]
        tot_tp = [0., 0.]
        tot_fp = [0., 0.]
        tot_fn = [0., 0.]
        with tqdm(total=len(self.val_loader)) as prog_bar:
            with torch.no_grad():
                for step, batch in enumerate(self.val_loader, 1):# step=batch_idx, data=batch
                    adj_b, convo_len_b, y_pairs_b, y_emotions_b, y_causes_b, y_mask_b,\
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b = batch
                    batch_loss, tp, fp, fn, y_emotions_b_masked = self.update_val(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b)

                    samples = len(y_emotions_b_masked)
                    total_samples[0] += samples
                    total_samples[1] += samples * samples
                    tot_tp[0] += tp[0]
                    tot_fp[0] += fp[0]
                    tot_fn[0] += fn[0]
                    tot_tp[1] += tp[1]
                    tot_fp[1] += fp[1]
                    tot_fn[1] += fn[1]
                    val_epoch_loss += batch_loss
                    if (tp[0] + fp[0] + fn[0]) == 0:
                        val_accuracy_c = 0.0
                    else:
                        val_accuracy_c = (tp[0]) / (tp[0] + fp[0] + fn[0])
                    if (tp[0] + fp[0]) == 0:
                        val_precision_c = 0.0
                    else:
                        val_precision_c = (tp[0])/(tp[0] + fp[0])
                    if (tp[0] + fn[0]) == 0:
                        val_recall_c = 0.0
                    else:
                        val_recall_c = tp[0] / (tp[0] + fn[0])
                    if (val_precision_c + val_recall_c) == 0:
                        val_f1_c = 0
                    else:
                        val_f1_c = (2 * val_precision_c * val_recall_c) / (val_precision_c + val_recall_c)
                    if (tp[1] + fp[1] + fn[1]) == 0:
                        val_accuracy_e = 0.0
                    else:
                        val_accuracy_e = (tp[1]) / (tp[1] + fp[1] + fn[1])
                    if (tp[1] + fp[1]) == 0:
                        val_precision_e = 0.0
                    else:
                        val_precision_e = (tp[1])/(tp[1] + fp[1])
                    if (tp[1] + fn[1]) == 0:
                        val_recall_e = 0.0
                    else:
                        val_recall_e = tp[1] / (tp[1] + fn[1])
                    if (val_precision_e + val_recall_e) == 0:
                        val_f1_e = 0
                    else:
                        val_f1_e = (2 * val_precision_e * val_recall_e) / (val_precision_e + val_recall_e)

                    prog_bar.set_description("Epoch: %d\tStep: %d\tValidation Loss: %0.4f" % (epoch+1, step, batch_loss))
                    prog_bar.update()
                    wandb.log({"epoch":epoch+1, "step_val_loss":batch_loss, "step_val_acc_e":val_accuracy_e, "step_val_precision_e":val_precision_e, "step_val_recall_e":val_recall_e, "step_val_f1_e":val_f1_e, "step_val_acc_c":val_accuracy_c, "step_val_precision_c":val_precision_c, "step_val_recall_c":val_recall_c, "step_val_f1_c":val_f1_c})
        return val_epoch_loss / len(self.val_loader), tot_tp, tot_fp, tot_fn

    def update_val(self, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b):
        self.model.eval()
        y_preds_e, y_preds_c = self.model(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b)
        y_causes_b = torch.tensor(y_causes_b, dtype=torch.float32).to(self.device)
        y_emotions_b = torch.tensor(y_emotions_b, dtype=torch.float32).to(self.device)
        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        y_preds_e = y_preds_e.masked_select(y_mask_b)
        y_preds_c = y_preds_c.masked_select(y_mask_b)
        y_causes_b = y_causes_b.masked_select(y_mask_b)
        y_emotions_b = y_emotions_b.masked_select(y_mask_b)
        binary_y_preds_e = (y_preds_e > self.threshold).float()
        binary_y_preds_c = (y_preds_c > self.threshold).float()

        loss_e = self.criterion(y_preds_e, y_emotions_b)
        loss_c = self.criterion(y_preds_c, y_causes_b)
        loss = loss_e + loss_c
        # loss_e = torch.sum(loss_e) / len(y_preds_e)
        # loss_c = torch.sum(loss_c) / len(y_preds_c)

        tp, fp, fn = self.tp_fp_fn(binary_y_preds_e, binary_y_preds_c, y_emotions_b, y_causes_b)
        return loss.item(), tp, fp, fn, y_emotions_b

    def tp_fp_fn(self, predictions_e, predictions_c, labels_e, labels_c):
        print("predictions batch emotion")
        print(predictions_e)
        print("labels batch emotion")
        print(labels_e)
        print("predictions batch c")
        print(predictions_c)
        print("labels batch c")
        print(labels_c)

        predictions_e = predictions_e.to(torch.int)
        predictions_c = predictions_c.to(torch.int)
        labels_e = labels_e.to(torch.int)
        labels_c = labels_c.to(torch.int)

        tp = [torch.sum(predictions_c & labels_c), torch.sum(predictions_e & labels_e)]
        fp = [torch.sum(predictions_c & ~labels_c), torch.sum(predictions_e & ~labels_e)]
        fn = [torch.sum(~predictions_c & labels_c), torch.sum(~predictions_e & labels_e)]

        return tp, fp, fn

    # define functions for saving and loading models per fold

