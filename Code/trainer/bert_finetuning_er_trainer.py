# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from os.path import join
from base import BaseTrainer
from utility import inf_loop, MetricTracker


class BERTERTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config,
                 data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        #self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = int(data_loader.sampler.__len__()/data_loader.batch_size/10)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (tcr_beta_tokenized,tcr_alpha_tokenized, antigen_tokenized, target) in enumerate(self.data_loader):

            tcr_beta_tokenized = {k: v.to(self.device) for k, v in tcr_beta_tokenized.items()}
            tcr_alpha_tokenized = {k: v.to(self.device) for k, v in tcr_alpha_tokenized.items()}
            antigen_tokenized = {k: v.to(self.device) for k, v in antigen_tokenized.items()}
 
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(tcr_beta_tokenized, tcr_alpha_tokenized, antigen_tokenized)
            # target shape: [batch_size, 1], output shape: [batch_size, 1]
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                if self.config["is_regression"]:
                    y_pred = output
                else:
                    y_pred = torch.sigmoid(output)
                
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    if met.__name__ != "roc_auc" and met.__name__ != 'recall'  and met.__name__ != "mse" and met.__name__ != 'get_spearman' and met.__name__ != 'get_pearson':
                        self.train_metrics.update(met.__name__, met(y_pred, y_true))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_'+k : v for k, v in val_log.items()}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        result_dict = {'y_true': [], 'y_pred': []}
        with torch.no_grad():
            for batch_idx, (tcr_beta_tokenized, tcr_alpha_tokenized, antigen_tokenized, target) in enumerate(self.valid_data_loader):
                tcr_beta_tokenized = {k: v.to(self.device) for k, v in tcr_beta_tokenized.items()}
                tcr_alpha_tokenized = {k: v.to(self.device) for k, v in tcr_alpha_tokenized.items()}
                antigen_tokenized = {k: v.to(self.device) for k, v in antigen_tokenized.items()}

                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(tcr_beta_tokenized, tcr_alpha_tokenized, antigen_tokenized)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
###
                if self.config["is_regression"]:
                    y_pred = output
                else:
                    y_pred = torch.sigmoid(output)
                
                
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                result_dict['y_pred'].append(y_pred)
                result_dict['y_true'].append(y_true)

        y_pred = np.concatenate(result_dict['y_pred'])
        y_true = np.concatenate(result_dict['y_true'])
        print(y_pred[:5], y_true[:5])
        for met in self.metric_fns:
            self.valid_metrics.update(met.__name__, met(y_pred, y_true))

        valid_metrics = self.valid_metrics.result()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return valid_metrics

    def test(self, tcr_tokenizer, antigen_tokenizer):
        self.model.eval()
        result_dict = {'tcr_beta': [],'tcr_alpha':[], 'antigen':[],
                       'y_true': [], 'y_pred': []}
        with torch.no_grad():
            for batch_idx, (tcr_beta_tokenized,tcr_alpha_tokenized, antigen_tokenized, target) in enumerate(self.test_data_loader):
                tcr_beta_tokenized = {k: v.to(self.device) for k, v in tcr_beta_tokenized.items()}
                tcr_alpha_tokenized = {k: v.to(self.device) for k, v in tcr_alpha_tokenized.items()}
                antigen_tokenized = {k: v.to(self.device) for k, v in antigen_tokenized.items()}
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(tcr_beta_tokenized, tcr_alpha_tokenized, antigen_tokenized)
                
                if self.config["is_regression"]:
                    y_pred = output
                else:
                    y_pred = torch.sigmoid(output)
                
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                result_dict['y_pred'].append(y_pred)
                result_dict['y_true'].append(y_true)

                tcr_beta = tcr_tokenizer.batch_decode(tcr_beta_tokenized['input_ids'],
                                                        skip_special_tokens=True)
                tcr_beta = [s.replace(" ", "") for s in tcr_beta]

                tcr_alpha = tcr_tokenizer.batch_decode(tcr_alpha_tokenized['input_ids'],
                                                        skip_special_tokens=True)
                tcr_alpha = [s.replace(" ", "") for s in tcr_alpha]

                antigen = antigen_tokenizer.batch_decode(antigen_tokenized['input_ids'],
                                                        skip_special_tokens=True)
                antigen = [s.replace(" ", "") for s in antigen]


                result_dict['tcr_beta'].append(tcr_beta)
                result_dict['tcr_alpha'].append(tcr_alpha)
                result_dict['antigen'].append(antigen)

        test_metrics = {}
        y_pred = np.concatenate(result_dict['y_pred'])
        y_true = np.concatenate(result_dict['y_true'])
        for met in self.metric_fns:
            test_metrics[met.__name__] = met(y_pred, y_true)

        test_df = pd.DataFrame({'tcr_beta': [v for l in result_dict['tcr_beta'] for v in l],
                                'tcr_alpha': [v for l in result_dict['tcr_alpha'] for v in l],
                                'antigen': [v for l in result_dict['antigen'] for v in l],
                                'y_true': list(y_true.flatten()),
                                'y_pred': list(y_pred.flatten())})
        test_df.to_csv(join(self.config.log_dir, 'test_result.csv'), index=False)

        return test_metrics

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
