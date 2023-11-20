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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from models import EmotionCausePairExtractorModel
from transformers import get_linear_schedule_with_warmup

class Wrapper():
    def __init__(self, args):
        self.k = args.kfold
        self.seed = args.seed
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.l2_reg = args.l2_reg
        self.device = args.device
        self.max_convo_len = args.max_convo_len
        self.batch_size = args.batch_size
        self.threshold_emo = args.threshold_emo
        self.threshold_cau = args.threshold_cau
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.threshold_pairs = args.threshold_pairs
        self.warmup_proportion = args.warmup_proportion

    def run(self, args):
        cause_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
        emotion_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
        pair_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
        for fold_id in range(1, self.k + 1):
            wandb.init(
                project="mecpe_task2",
                config={
                "epochs":args.num_epochs,
                "lr":args.lr,
                "batch_size":args.batch_size,
                "threshold_emo":args.threshold_emo,
                "threshold_cau":args.threshold_cau,
                "threshold_pairs":args.threshold_pairs,
                "max_convo_len":args.max_convo_len,
                },
                entity='arefa2001',
                name=f"fold{fold_id}",
                reinit=True,
            )
            print("\n\n>>>>>>>>>>>>>>>>>>FOLD %d<<<<<<<<<<<<<<<<<<<<<<" % (fold_id))
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
            best_val_f1_p = None

            # Model, loss fn and optimizer
            self.model = EmotionCausePairExtractorModel(args)
            self.model.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean') # apply reduction = 'none'?
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.num_update_steps = len(self.train_loader) // self.gradient_accumulation_steps * self.num_epochs
            self.warmup_steps = self.warmup_proportion * self.num_update_steps
            scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_update_steps)

            # Training and Validation loop
            for epoch in range(self.num_epochs):
                training_epoch_loss, total_correct, total_samples = self.train(epoch)
                train_accuracy_c = total_correct[0] / total_samples[0]
                train_accuracy_e = total_correct[1] / total_samples[0]
                train_accuracy_p = total_correct[2] / total_samples[1]

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Train Loss: {training_epoch_loss:.4f}, Train Accuracy Cause: {train_accuracy_c:.4f}, Train Accuracy Emotion: {train_accuracy_e:.4f}, Train Accuracy Pair: {train_accuracy_p:.4f}")
                train_losses.append(training_epoch_loss)
                train_accuracy_list.append([train_accuracy_c, train_accuracy_e, train_accuracy_p])

                # Evaluation
                val_epoch_loss, tp, fp, fn, y_preds_e, y_preds_c, y_preds_p, true_e, true_c, true_p = self.evaluate(epoch)
                # Cause
                val_accuracy_c = self.accuracy(tp[0], fp[0], fn[0])
                val_precision_c = self.precision(tp[0], fp[0])
                val_recall_c = self.recall(tp[0], fn[0])
                val_f1_c = self.f1_score(val_precision_c, val_recall_c)
                # Emotion
                val_accuracy_e = self.accuracy(tp[1], fp[1], fn[1])
                val_precision_e = self.precision(tp[1], fp[1])
                val_recall_e = self.recall(tp[1], fn[1])
                val_f1_e = self.f1_score(val_precision_e, val_recall_e)
                # Pairs
                val_accuracy_p = self.accuracy(tp[2], fp[2], fn[2])
                val_precision_p = self.precision(tp[2], fp[2])
                val_recall_p = self.recall(tp[2], fn[2])
                val_f1_p = self.f1_score(val_precision_p, val_recall_p)
                # AUROC
                auroc_e = roc_auc_score(true_e, y_preds_e)
                auroc_c = roc_auc_score(true_c, y_preds_c)
                auroc_p = roc_auc_score(true_p, y_preds_p)

                wandb.log({"auroc_e":auroc_e, "auroc_c":auroc_c, "auroc_p":auroc_p})

                # Plot ROC
                e_fpr, e_tpr, thresholds_e = roc_curve(true_e, y_preds_e)
                c_fpr, c_tpr, thresholds_c = roc_curve(true_c, y_preds_c)
                p_fpr, p_tpr, thresholds_p = roc_curve(true_p, y_preds_p)
                plt.figure(figsize=(8, 6))
                plt.plot(e_fpr, e_tpr, color='aqua', lw=2, label=f'ROC curve (Emotion, area = {auroc_e:.2f})')
                plt.plot(c_fpr, c_tpr, color='darkorange', lw=2, label=f'ROC curve (Cause, area = {auroc_c:.2f})')
                plt.plot(p_fpr, p_tpr, color='cornflowerblue', lw=2, label=f'ROC curve (Pair, area = {auroc_p:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.show()


                print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Val Loss: {val_epoch_loss:.4f}\nCause: Val Accuracy: {val_accuracy_c:.4f}, Val Precision: {val_precision_c:.4f}, Val Recall: {val_recall_c:.4f}, Val F1: {val_f1_c:.4f}\nEmotion: Val Accuracy: {val_accuracy_e:.4f} Val Precision: {val_precision_e:.4f}, Val Recall: {val_recall_e:.4f}, Val F1: {val_f1_e:.4f}\nPair: Val Accuracy: {val_accuracy_p:.4f}, Val Precision: {val_precision_p:.4f}, Val Recall: {val_recall_p:.4f}, Val F1: {val_f1_p:.4f}")
                val_losses.append(val_epoch_loss)
                val_accuracy_list.append([val_accuracy_c, val_accuracy_e, val_accuracy_p])
                val_precision_list.append([val_precision_c, val_precision_e, val_precision_p])
                val_recall_list.append([val_recall_c, val_recall_e, val_recall_p])
                val_f1_list.append([val_f1_c, val_f1_e, val_f1_p])

                # Store best f1 across all folds
                if best_val_f1_c == None or val_f1_c > best_val_f1_c:
                    best_val_f1_c = val_f1_c
                if best_val_f1_e == None or val_f1_e > best_val_f1_e:
                    best_val_f1_e = val_f1_e
                if best_val_f1_p == None or val_f1_p > best_val_f1_p:
                    best_val_f1_p = val_f1_p

                print("\n>>>>>>>>>>>>>EPOCH END<<<<<<<<<<<<<<<<")

            # Calculate mean of the validation metrics for this fold
            mean_accuracy = torch.mean(val_accuracy_list, axis=0)
            mean_precision = torch.mean(val_precision_list, axis=0)
            mean_recall = torch.mean(val_recall_list, axis=0)
            mean_f1 = torch.mean(val_f1_list,axis=0)

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
            print("Pair")
            print(f"Mean Accuracy: {mean_accuracy[2]:.4f}")
            print(f"Mean Precision: {mean_precision[2]:.4f}")
            print(f"Mean Recall: {mean_recall[2]:.4f}")
            print(f"Mean F1: {mean_f1[2]:.4f}")
            print("\n>>>>>>>>>>>>>>>>>>FOLD END<<<<<<<<<<<<<<<<<<<<<<")

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
            pair_aprfb['acc'].append(mean_accuracy[2])
            pair_aprfb['p'].append(mean_precision[2])
            pair_aprfb['r'].append(mean_recall[2])
            pair_aprfb['f'].append(mean_f1[2])
            pair_aprfb['b'].append(best_val_f1[2])

        return cause_aprfb, emotion_aprfb, pair_aprfb

    def train(self, epoch):
        train_epoch_loss = 0.
        total_correct = [0., 0., 0.]
        total_samples = [0., 0.]
        with tqdm(total=len(self.train_loader)) as prog_bar:
            for step, batch in enumerate(self.train_loader, 1):#step=batch_idx, data=batch
                adj_b, convo_len_b, y_pairs_b, y_emotions_b, y_causes_b, y_mask_b, \
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, \
                    pairs_labels_b, pairs_mask_b = batch

                batch_loss, correct_e, correct_c, correct_p, y_emotions_b_masked, pairs_labels_b_masked= self.update(step, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b, pairs_labels_b, pairs_mask_b)
                samples = len(y_emotions_b_masked)
                samples_pairs = len(pairs_labels_b_masked)
                total_samples[0] += samples
                total_samples[1] += samples_pairs
                total_correct[0] += correct_c
                total_correct[1] += correct_e
                total_correct[2] += correct_p
                train_epoch_loss += batch_loss
                prog_bar.set_description("Epoch: %d\tStep: %d\tTraining Loss: %0.4f" % (epoch+1, step, batch_loss))

                wandb.log({"epoch":epoch+1, "step_train_loss":batch_loss, "step_train_acc_c":correct_c/samples, "step_train_acc_e":correct_e/samples, "step_train_acc_p": correct_p/samples_pairs})
                prog_bar.update()
        return train_epoch_loss / len(self.train_loader), total_correct, total_samples

    def update(self, step, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b, pairs_labels_b, pairs_mask_b):
        self.model.train()
        y_preds_e, y_preds_c, y_preds_p = self.model(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_mask_b)
        y_causes_b = torch.tensor(y_causes_b, dtype=torch.float32).to(self.device)
        y_emotions_b = torch.tensor(y_emotions_b, dtype=torch.float32).to(self.device)
        pairs_labels_b = torch.tensor(pairs_labels_b, dtype=torch.float32).to(self.device)

        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        pairs_mask_b = torch.tensor(pairs_mask_b).bool().to(self.device)

        # using emo cau prediction
        # preds_p = torch.zeros(pairs_mask_b.shape).to(self.device)
        # for (i, j), pred, bi in zip(pairs_pos, y_preds_p, batch_idxs):
        #     preds_p[bi][i * len(y_emotions_b[0]) + j] = pred

        # print("pair labels b {}".format(pairs_labels_b.shape))
        # print("pair preds b {}".format(y_preds_p.shape))

        # using all pairs (rankcp)
        preds_e = y_preds_e.masked_select(y_mask_b) # masked_select converts into 1d tensor
        preds_c = y_preds_c.masked_select(y_mask_b)
        preds_p = y_preds_p.masked_select(pairs_mask_b)

        y_causes_b = y_causes_b.masked_select(y_mask_b)
        y_emotions_b = y_emotions_b.masked_select(y_mask_b)
        pairs_labels_b = pairs_labels_b.masked_select(pairs_mask_b)

        binary_y_preds_e = (torch.sigmoid(preds_e) > self.threshold_emo).float()
        binary_y_preds_c = (torch.sigmoid(preds_c) > self.threshold_cau).float()
        binary_y_preds_p = (torch.sigmoid(preds_p) > self.threshold_pairs).float()

        # print("binary preds e")
        # print(binary_y_preds_e)
        # print("labels e")
        # print(y_emotions_b)
        # print("binary preds c")
        # print(binary_y_preds_c)
        # print("labels c")
        # print(y_causes_b)
        # print("binary preds p")
        # print(binary_y_preds_p)
        # print("labels p")
        # print(pairs_labels_b)

        # print("pair labels b {}".format(pairs_labels_b.shape))
        # print("pair masks b {}".format(pairs_mask_b.shape))
        # print("pair preds b {}".format(preds_p.shape))

        loss_e = self.criterion(preds_e, y_emotions_b)
        loss_c = self.criterion(preds_c, y_causes_b)
        loss_p = self.pair_criterion(preds_p, pairs_labels_b)
        loss = loss_e + loss_c + loss_p
        loss = loss / self.gradient_accumulation_steps

        correct_c = (binary_y_preds_c == y_causes_b).sum().item()
        correct_e = (binary_y_preds_e == y_emotions_b).sum().item()
        correct_p = (binary_y_preds_p == pairs_labels_b).sum().item()

        loss.backward()
        if (step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()
        return loss.item(), correct_e, correct_c, correct_p, y_emotions_b, pairs_labels_b

    def evaluate(self, epoch):
        val_epoch_loss = 0.
        total_samples = [0, 0]
        tot_tp = [0., 0., 0.]
        tot_fp = [0., 0., 0.]
        tot_fn = [0., 0., 0.]
        y_preds_e, y_preds_c, y_preds_p = [], [], []
        true_e, true_c, true_p = [], [], []
        with tqdm(total=len(self.val_loader)) as prog_bar:
            with torch.no_grad():
                for step, batch in enumerate(self.val_loader, 1):# step=batch_idx, data=batch
                    adj_b, convo_len_b, y_pairs_b, y_emotions_b, y_causes_b, y_mask_b,\
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, \
                    pairs_labels_b, pairs_mask_b = batch
                    batch_loss, tp, fp, fn, y_emotions_b_masked, pairs_labels_b_masked , y_preds_e_step, y_preds_c_step, y_preds_p_step, true_e_step, true_c_step, true_p_step = self.update_val(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b, pairs_labels_b, pairs_mask_b)

                    samples = len(y_emotions_b_masked)
                    samples_pairs = len(pairs_labels_b_masked)
                    total_samples[0] += samples
                    total_samples[1] += samples_pairs
                    tot_tp[0] += tp[0]
                    tot_fp[0] += fp[0]
                    tot_fn[0] += fn[0]
                    tot_tp[1] += tp[1]
                    tot_fp[1] += fp[1]
                    tot_fn[1] += fn[1]
                    tot_tp[2] += tp[2]
                    tot_fp[2] += fp[2]
                    tot_fn[2] += fn[2]
                    val_epoch_loss += batch_loss

                    val_accuracy_c = self.accuracy(tp[0], fp[0], fn[0])
                    val_precision_c = self.precision(tp[0], fp[0])
                    val_recall_c = self.recall(tp[0], fn[0])
                    val_f1_c = self.f1_score(val_precision_c, val_recall_c)
                    val_accuracy_e = self.accuracy(tp[1], fp[1], fn[1])
                    val_precision_e = self.precision(tp[1], fp[1])
                    val_recall_e = self.recall(tp[1], fn[1])
                    val_f1_e = self.f1_score(val_precision_e, val_recall_e)
                    val_accuracy_p = self.accuracy(tp[2], fp[2], fn[2])
                    val_precision_p = self.precision(tp[2], fp[2])
                    val_recall_p = self.recall(tp[2], fn[2])
                    val_f1_p = self.f1_score(val_precision_p, val_recall_p)

                    # Add preds and labels for roc
                    y_preds_e.extend(y_preds_e_step.cpu().numpy())
                    y_preds_c.extend(y_preds_c_step.cpu().numpy())
                    y_preds_p.extend(y_preds_p_step.cpu().numpy())
                    true_e.extend(true_e_step.cpu().numpy())
                    true_c.extend(true_c_step.cpu().numpy())
                    true_p.extend(true_p_step.cpu().numpy())

                    prog_bar.set_description("Epoch: %d\tStep: %d\tValidation Loss: %0.4f" % (epoch+1, step, batch_loss))
                    prog_bar.update()

                    wandb.log({"epoch":epoch+1, "step_val_loss":batch_loss, "step_val_acc_e":val_accuracy_e, "step_val_precision_e":val_precision_e, "step_val_recall_e":val_recall_e, "step_val_f1_e":val_f1_e, "step_val_acc_c":val_accuracy_c, "step_val_precision_c":val_precision_c, "step_val_recall_c":val_recall_c, "step_val_f1_c":val_f1_c, "step_val_acc_p":val_accuracy_p, "step_val_precision_p": val_precision_p, "step_val_recall_p":val_recall_p, "step_val_f1_p":val_f1_p})
        return val_epoch_loss / len(self.val_loader), tot_tp, tot_fp, tot_fn, y_preds_e, y_preds_c, y_preds_p, true_e, true_c, true_p

    def update_val(self, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, y_pairs_b, pairs_labels_b, pairs_mask_b):
        self.model.eval()
        y_preds_e, y_preds_c, y_preds_p = self.model(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_mask_b)
        y_causes_b = torch.tensor(y_causes_b, dtype=torch.float32).to(self.device)
        y_emotions_b = torch.tensor(y_emotions_b, dtype=torch.float32).to(self.device)
        pairs_labels_b = torch.tensor(pairs_labels_b, dtype=torch.float32).to(self.device)

        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        pairs_mask_b = torch.tensor(pairs_mask_b).bool().to(self.device)

        # preds_p = torch.zeros(pairs_mask_b.shape).to(self.device)
        # for (i, j), pred, bi in zip(pairs_pos, y_preds_p, batch_idxs):
        #     preds_p[bi][i * len(y_emotions_b[0]) + j] = pred

        preds_e = y_preds_e.masked_select(y_mask_b)
        preds_c = y_preds_c.masked_select(y_mask_b)
        preds_p = y_preds_p.masked_select(pairs_mask_b)

        y_causes_b = y_causes_b.masked_select(y_mask_b)
        y_emotions_b = y_emotions_b.masked_select(y_mask_b)
        pairs_labels_b = pairs_labels_b.masked_select(pairs_mask_b)

        binary_y_preds_e = (torch.sigmoid(preds_e) > self.threshold_emo).float()
        binary_y_preds_c = (torch.sigmoid(preds_c) > self.threshold_cau).float()
        binary_y_preds_p = (torch.sigmoid(preds_p) > self.threshold_pairs).float()

        loss_p = self.pair_criterion(preds_p, pairs_labels_b)
        loss_e = self.criterion(preds_e, y_emotions_b)
        loss_c = self.criterion(preds_c, y_causes_b)
        loss = loss_e + loss_c + loss_p

        tp, fp, fn = self.tp_fp_fn(binary_y_preds_e, binary_y_preds_c, binary_y_preds_p, y_emotions_b, y_causes_b, pairs_labels_b)
        return loss.item(), tp, fp, fn, y_emotions_b, pairs_labels_b, preds_e, preds_c, preds_p, y_emotions_b, y_causes_b, pairs_labels_b

    def tp_fp_fn(self, predictions_e, predictions_c, predictions_p, labels_e, labels_c, labels_p):
        # print("predictions batch emotion")
        # print(predictions_e)
        # print("labels batch emotion")
        # print(labels_e)
        # print("predictions batch c")
        # print(predictions_c)
        # print("labels batch c")
        # print(labels_c)
        # print("predictions batch p")
        # print(predictions_p)
        # print("labels batch p")
        # print(labels_p)

        predictions_e = predictions_e.to(torch.int)
        predictions_c = predictions_c.to(torch.int)
        predictions_p = predictions_p.to(torch.int)
        labels_e = labels_e.to(torch.int)
        labels_c = labels_c.to(torch.int)
        labels_p = labels_p.to(torch.int)

        tp = [torch.sum(predictions_c & labels_c), torch.sum(predictions_e & labels_e), torch.sum(predictions_p & labels_p)]
        fp = [torch.sum(predictions_c & ~labels_c), torch.sum(predictions_e & ~labels_e), torch.sum(predictions_p & ~labels_p)]
        fn = [torch.sum(~predictions_c & labels_c), torch.sum(~predictions_e & labels_e), torch.sum(~predictions_p & labels_p)]

        return tp, fp, fn

    # define functions for saving and loading models per fold

    def pair_criterion(self, y_preds_p, pairs_labels_b):
        loss_pairs = self.criterion(y_preds_p, pairs_labels_b)
        # change

        return loss_pairs

    def accuracy(self, tp, fp, fn):
        if (tp + fp + fn) == 0:
            acc = 0.0
        else:
            acc = (tp) / (tp + fp + fn)
        return acc

    def precision(self, tp, fp):
        if (tp + fp) == 0:
            prec = 0.0
        else:
            prec = (tp)/(tp + fp)
        return prec

    def recall(self, tp, fn):
        if (tp + fn) == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)
        return rec

    def f1_score(self, precision, recall):
        if(precision + recall) == 0:
            return 0.0
        else:
            return (2 * precision * recall) / (precision + recall)

