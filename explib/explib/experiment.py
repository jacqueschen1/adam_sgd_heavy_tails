import time
import torch
import math
import os
import random
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from . import dataset, expmaker, logging, model, optim
from .util import get_grads, enable_running_stats, disable_running_stats
from accelerate import Accelerator


class Experiment:
    def __init__(
        self,
        exp_dict,
        workspace_dir,
        experiment_hash,
        debug,
        verbose,
        gpu,
        trained_norms,
    ):
        """Create an experiment"""
        # Get model, dataset, optimizers and other parameters
        self.seed = exp_dict["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.batch_size = exp_dict["batch_size"]
        self.max_epoch = exp_dict["max_epoch"]
        self.save_path = os.path.join(
            workspace_dir, exp_dict["dataset"], experiment_hash
        )
        self.logging = logging.init(
            workspace_dir, experiment_hash, exp_dict["dataset"], debug, verbose
        )
        self.model_args = exp_dict["model_args"] if "model_args" in exp_dict else None
        if self.model_args is None and exp_dict["dataset"] == "mnist":
            self.model_args = {}
            self.model_args["in_channels"] = 1
        self.device = gpu if torch.cuda.is_available() else "cpu"

        self.model_name = exp_dict["model"]

        self.full_batch = "full_batch" in exp_dict and exp_dict["full_batch"]
        self.drop_last = "drop_last" in exp_dict and exp_dict["drop_last"]

        # need separate training and validation datasets for BERT
        if self.model_name == "bert_base_pretrained":
            (
                self.train_dataloader,
                self.train_dataloader_for_eval,
                self.valid_dataloader,
                self.valid_dataset,
                self.valid_examples,
                self.train_dataset,
                self.train_examples,
            ) = dataset.init(
                exp_dict["dataset"],
                self.batch_size,
                workspace_dir,
                self.device,
                self.model_name,
                model_args=self.model_args,
                drop_last=self.drop_last,
                full_batch=self.full_batch,
            )
            optional_transformer_len = []
        else:
            (
                self.train_dataloader,
                self.valid_dataloader,
                *optional_transformer_len,
            ) = dataset.init(
                exp_dict["dataset"],
                self.batch_size,
                workspace_dir,
                self.device,
                self.model_name,
                model_args=self.model_args,
                drop_last=self.drop_last,
                full_batch=self.full_batch,
            )
        self.dataset_name = exp_dict["dataset"]
        if len(optional_transformer_len) > 0:
            self.model = model.init(
                exp_dict["model"],
                model_args=self.model_args,
                features_dim=optional_transformer_len[0],
            )
        else:
            if self.model_name != "bert_base_pretrained":
                features_dim = next(iter(self.train_dataloader))[0].shape[1]
            else:
                features_dim = 0
            self.model = model.init(
                exp_dict["model"],
                model_args=self.model_args,
                features_dim=features_dim,
            )
        self.model.to(self.device)

        self.optim = optim.init(
            exp_dict["opt"],
            self.model,
            len(self.train_dataloader),
        )
        self.optim_name = exp_dict["opt"]["name"]
        self.metrics = exp_dict["metrics"]
        self.loss_func = self.get_loss_function(exp_dict["loss_func"])
        self.init_noise_norm = (
            "init_noise_norm" in exp_dict and exp_dict["init_noise_norm"]
        )

        self.noise_norm_train = True
        self.log_every_step = (
            "log_every_step" in exp_dict and exp_dict["log_every_step"]
        )

        # Gradient accumulation for noise norm calculation
        if "accumulate_steps" in exp_dict:
            self.accumulate_steps = exp_dict["accumulate_steps"]
            self.grad_accumulate = True
        else:
            self.accumulate_steps = 1
            self.grad_accumulate = False

        self.trained_norms = trained_norms or (
            "trained_norms" in exp_dict and exp_dict["trained_norms"]
        )

        self.logging(exp_dict)
        self.logging({"hash": experiment_hash})
        self.logging({"device": self.device})

        if self.model_name == "bert_base_pretrained":
            self.accelerator = Accelerator()
            (
                self.model,
                self.optim,
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.model, self.optim, self.train_dataloader, self.valid_dataloader
            )
            self.train_dataloader_for_eval = self.accelerator.prepare(
                self.train_dataloader_for_eval
            )

    def run(self):
        """Run the experiment"""
        starting_epoch = 0

        r_start = time.time()
        if self.init_noise_norm or self.trained_norms:
            self.calculate_noise_norm(epoch=0)
        r_end = time.time()
        print(r_start - r_end, "norm calculation time")

        print("initial eval")
        r_start = time.time()
        self.eval()
        r_end = time.time()
        print(r_start - r_end, "eval time")

        model_dir = os.path.join(self.save_path, "model")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_file = os.path.join(model_dir, "model.pt")
        if not os.path.isfile(model_file):
            model_file = os.path.join(model_dir, "model_{}.pt".format(self.max_epoch))

        epoch_for_norms = [
            1,
            int(self.max_epoch * 0.1),
            int(self.max_epoch * 0.25),
            int(self.max_epoch * 0.5),
            int(self.max_epoch * 0.75),
        ]

        for epoch in range(starting_epoch, self.max_epoch):

            if epoch in epoch_for_norms and self.trained_norms:
                self.calculate_noise_norm(epoch=epoch)

            # Train Model
            epoch_begin_time = time.time()
            train_loss = self.train()
            epoch_end_time = time.time()
            epoch_training_time = epoch_end_time - epoch_begin_time

            if (
                self.optim_name == "SGD_Armijo" or self.optim_name == "Adam_Armijo"
            ) and not self.log_every_step:
                self.logging({"step_size": self.optim.state["step_size"]}, commit=False)

            if math.isnan(train_loss) or math.isinf(train_loss):
                if math.isnan(train_loss):
                    self.logging({"training_error": "nan"})
                else:
                    self.logging({"training_error": "inf"})
                break

            # Evaluation
            self.eval()
            self.logging(
                {
                    "epoch": epoch,
                    "average_training_loss": train_loss,
                    "epoch_training_time": epoch_training_time,
                }
            )

        if not os.path.isfile(model_file):
            with open(model_file, "wb") as f:
                torch.save(self.model.state_dict(), f)
            if self.trained_norms:
                self.calculate_noise_norm(epoch=self.max_epoch)

    def eval(self):
        """Evaluate model on Training Set and (if not full batch) on Validation Set"""
        train_metrics = {}
        valid_metrics = {}
        if self.model_name == "bert_base_pretrained":
            if not self.full_batch:
                metrics = self.calculate_metric(
                    self.valid_dataloader,
                    "bert_base_pretrained_metrics",
                    self.valid_dataset,
                    self.valid_examples,
                )
                valid_metrics["valid_exact_match"] = metrics["exact_match"]
                valid_metrics["valid_exact_f1"] = metrics["f1"]

            metrics = self.calculate_metric(
                self.train_dataloader_for_eval,
                "bert_base_pretrained_metrics",
                self.train_dataset,
                self.train_examples,
            )
            loss = self.eval_loss(self.train_dataloader)
            train_metrics["train_exact_match"] = metrics["exact_match"]
            train_metrics["train_exact_f1"] = metrics["f1"]
            train_metrics["training_loss"] = loss
        else:
            for metric in self.metrics:
                (
                    train_metrics["train_" + metric],
                    train_metrics["training_loss"],
                ) = self.calculate_metric(self.train_dataloader, metric)
                if not self.full_batch:
                    valid_metrics["valid_" + metric], _ = self.calculate_metric(
                        self.valid_dataloader, metric, no_loss=True
                    )

        self.logging(train_metrics, commit=False)
        if not self.full_batch:
            self.logging(valid_metrics, commit=False)

    def logLoss(self, predicted, actual):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(predicted, actual.long())

    def get_loss_function(self, function_name):
        criterion = None
        if function_name == "logloss":
            criterion = self.logLoss
        if function_name == "mse":
            criterion = torch.nn.MSELoss()

        return criterion

    def train(self):
        """Train for one epoch"""
        self.model.train()
        self.model.to(self.device)
        self.optim.zero_grad()

        if self.model_name == "bert_base_pretrained":
            return self.train_bert()

        epoch_loss = 0.0
        n = 0
        m = 0
        for (X, labels, *seq_len) in self.train_dataloader:
            X = X.to(self.device)

            if self.model_name == "transformer_encoder":
                closure_func = lambda: self.transformer_encoder_loss(
                    self.model, self.loss_func, X, labels, seq_len[0], self.device
                )
            elif self.model_name == "transformer_xl":
                closure_func = lambda: self.transformer_xl_loss(
                    self.model, self.loss_func, X, labels
                )
            else:
                labels = labels.to(self.device).float()
                closure_func = lambda: self.loss(self.model, self.loss_func, X, labels)

            if self.optim_name == "SGD_Armijo" or self.optim_name == "Adam_Armijo":
                logging = self.logging if self.log_every_step else None
                loss = self.optim.step(closure_func, logging=logging)
                n += 1
            else:
                loss = closure_func()
                if self.grad_accumulate:
                    loss = loss / self.accumulate_steps
                loss.backward()
                n += 1
                if not self.grad_accumulate or n % self.accumulate_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                    m += 1

            epoch_loss += loss.item()

            if self.full_batch:
                break

        epoch_loss = epoch_loss / m
        return epoch_loss

    def train_bert(self):
        n = 0
        m = 0
        epoch_loss = 0.0
        for X in self.train_dataloader:
            closure_func = lambda: self.model(**X).loss
            loss = closure_func()
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            self.accelerator.backward(loss)
            n += 1
            if not self.grad_accumulate or n % self.accumulate_steps == 0:
                self.optim.step()
                self.optim.zero_grad()
                m += 1
            epoch_loss += loss.item()

            if self.full_batch and m == 1:
                break
        epoch_loss = epoch_loss / m
        return epoch_loss

    def loss(self, model, loss_func, X, labels):
        y = model(X.float())
        return loss_func(y, labels)

    def transformer_xl_loss(self, model, loss_func, data, target):
        mems = tuple()
        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        return loss.float().mean().type_as(loss)

    def transformer_encoder_loss(self, model, loss_func, data, target, seq_len, device):
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
        output = model(data, src_mask)
        output_flat = output.view(-1, model.ntoken)
        return loss_func(output_flat, target.view(-1))

    @torch.no_grad()
    def evaluate_transformer_xl(self, dataloader, no_loss=False):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        ppl_loss = 0.0
        total_len = 0
        counter = 0
        m = 0
        mems = tuple()

        for batch, (data, target, seq_len) in enumerate(dataloader):
            ret = self.model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            ppl_loss += seq_len * loss.item()
            total_len += seq_len
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            total_loss += loss.item()
            counter += 1
            if not self.grad_accumulate or counter % self.accumulate_steps == 0:
                m += 1
            if self.full_batch:
                break
        if no_loss:
            epoch_loss = 0
        else:
            epoch_loss = total_loss / m

        return ppl_loss / total_len, epoch_loss

    @torch.no_grad()
    def eval_loss(self, dataloader, no_loss=False):
        """Run through the data once without training"""

        if self.model_name == "transformer_encoder":
            return model.transformer_encoder.evaluate(self, dataloader, no_loss=no_loss)
        if self.model_name == "transformer_xl":
            return self.evaluate_transformer_xl(dataloader, no_loss=no_loss)
        if self.model_name == "bert_base_pretrained":
            return model.bert_base_pretrained.eval_loss(self, self.model, dataloader)

        self.model.eval()
        self.model.to(self.device)
        epoch_loss = 0
        n = 0

        for (X, labels) in dataloader:

            X = X.to(self.device)
            labels = labels.to(self.device).float()

            y = self.model(X.float())
            loss = self.loss_func(y, labels)
            epoch_loss += loss.item()
            n += 1
            if self.full_batch:
                break

        epoch_loss = epoch_loss / n

        return epoch_loss

    @torch.no_grad()
    def calculate_metric(
        self, dataloader, metric, dataset=None, examples=None, no_loss=False
    ):
        """Calculate metrics for data in dataloader"""

        if metric == "accuracy":

            correct = torch.zeros(1).to(self.device)
            total_loss = 0
            n = 0
            m = 0
            counter = 0

            self.model.eval()
            self.model.to(self.device)

            for (X, labels) in dataloader:
                X = X.to(self.device)
                labels = labels.to(self.device).float()

                y = self.model(X)
                predicted = F.softmax(y, dim=1)
                _, predicted_labels = torch.max(predicted, 1)

                n += labels.size(0)
                correct += (predicted_labels == labels).sum()

                loss = self.loss(self.model, self.loss_func, X, labels)
                if self.grad_accumulate:
                    loss = loss / self.accumulate_steps
                total_loss += loss.item()
                counter += 1
                if not self.grad_accumulate or counter % self.accumulate_steps == 0:
                    m += 1
                if self.full_batch:
                    break

            if no_loss:
                epoch_loss = 0
            else:
                epoch_loss = total_loss / m

            return correct.item() / n, epoch_loss

        if metric == "mse":
            loss = self.eval_loss(dataloader)
            return loss, loss

        if metric == "ppl":
            try:
                ppl_loss, loss = self.eval_loss(dataloader, no_loss=no_loss)
                return math.exp(ppl_loss), loss
            except OverflowError:
                return float("inf"), float("inf")

        if metric == "bert_base_pretrained_metrics":
            return model.bert_base_pretrained.evaluate(
                self, self.model, dataloader, self.accelerator, dataset, examples
            )

    def save_noise_norm(self, noise_norms):
        torch.save(noise_norms, self.save_path + "/noise_norm_init.hist")

    def calculate_noise_norm(self, epoch=0):
        """Run through the data once without training, calculating noise norm"""
        if self.noise_norm_train:
            self.model.train()
        else:
            self.model.eval()

        self.model.apply(disable_running_stats)

        logs_path = os.path.join(self.save_path, "noise")
        Path(logs_path).mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        epoch_loss = 0
        n = 0
        grads = None
        self.optim.zero_grad()

        for (step, *X) in enumerate(self.train_dataloader):
            closure_func = self.get_closure_function_noise_norm(X)

            loss = closure_func()
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            epoch_loss += loss.item()

            if n % 1000 == 0:
                print("n", n)

            n += 1
            if self.model_name == "bert_base_pretrained":
                self.accelerator.backward(loss)
            else:
                loss.backward()

            if not self.grad_accumulate or (
                self.grad_accumulate and n % self.accumulate_steps == 0
            ):
                grad = get_grads(self.model).cpu()
                if grads is None:
                    grads = grad
                else:
                    grads = grads + grad
                self.optim.zero_grad()

        epoch_loss = epoch_loss / n

        torch.save(grads, self.save_path + "/noise/grad_{}_{}".format(n, epoch))
        self.calculate_norms(grads, n, epoch=epoch)
        self.model.apply(enable_running_stats)

    def calculate_norms(self, total_grads, n, epoch=0):
        print("calculate norm")
        n = n // self.accumulate_steps
        mean_grad = total_grads / n
        noise_norms = []

        m = 0
        self.optim.zero_grad()

        for (step, *X) in enumerate(self.train_dataloader):
            closure_func = self.get_closure_function_noise_norm(X)

            loss = closure_func()
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            if self.model_name == "bert_base_pretrained":
                self.accelerator.backward(loss)
            else:
                loss.backward()

            if m % 1000 == 0:
                print("m", m)
            m += 1

            if not self.grad_accumulate or (
                self.grad_accumulate and m % self.accumulate_steps == 0
            ):
                grad = get_grads(self.model).cpu()
                noise_norm = (grad - mean_grad).norm().item() ** 2
                noise_norms.append(noise_norm)
                self.optim.zero_grad()

        to_save = np.asarray(noise_norms)
        print(to_save.shape)
        np.save(
            self.save_path
            + "/noise/norm_{}_{}_{}_{}_{}_{}_{}".format(
                self.model_name,
                self.dataset_name,
                self.batch_size * self.accumulate_steps,
                self.seed,
                self.noise_norm_train,
                self.optim_name,
                epoch,
            ),
            to_save,
        )

    def get_closure_function_noise_norm(self, X):
        if self.model_name != "bert_base_pretrained":
            labels_seq_len = X[0][1:]
            X = X[0][0]
            X = X.to(self.device)

        if self.model_name == "transformer_encoder":
            labels, seq_len = labels_seq_len[0], labels_seq_len[1]
            closure_func = lambda: self.transformer_encoder_loss(
                self.model, self.loss_func, X, labels, seq_len, self.device
            )
        elif self.model_name == "transformer_xl":
            labels = labels_seq_len[0]
            closure_func = lambda: self.transformer_xl_loss(
                self.model, self.loss_func, X, labels
            )
        elif self.model_name == "bert_base_pretrained":
            closure_func = lambda: self.model(**X[0]).loss
        else:
            labels = labels_seq_len[0].to(self.device).float()
            closure_func = lambda: self.loss(self.model, self.loss_func, X, labels)

        return closure_func
