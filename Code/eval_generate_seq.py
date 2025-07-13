#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
from os.path import join

import data.bert_finetuning_er_seq2seq_dataset as module_data
# import data.bert_finetuning_er_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.bert_binding as module_arch
from trainer.bert_finetuning_er_trainer import BERTERTrainer as Trainer
# from trainer.bert_finetuning_er_trainer_4_input import BERTERTrainer as Trainer
from parse_config import ConfigParser

import pandas as pd

def test(model ,data_loader, tcr_tokenizer, antigen_tokenizer, device,config):
    model.eval()
    result_dict = {'beta': [],
                    'alpha': [],
                    'antigen': [],
                    'y_pred': []}
    with torch.no_grad():
        for batch_idx, (tcr_beta_tokenized,tcr_alpha_tokenized, antigen_tokenized) in enumerate(data_loader):
            tcr_beta_tokenized = {k: v.to(device) for k, v in tcr_beta_tokenized.items()}
            tcr_alpha_tokenized = {k: v.to(device) for k, v in tcr_alpha_tokenized.items()}
            antigen_tokenized = {k: v.to(device) for k, v in antigen_tokenized.items()}
            
            output = model(tcr_beta_tokenized, tcr_alpha_tokenized, antigen_tokenized)
            
            #y_pred = output
            ##
            y_pred = torch.sigmoid(output)
            
            y_pred = y_pred.cpu().detach().numpy()
            result_dict['y_pred'].append(y_pred)

            beta = tcr_tokenizer.batch_decode(tcr_beta_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            beta = [s.replace(" ", "") for s in beta]

            alpha = tcr_tokenizer.batch_decode(tcr_alpha_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            alpha = [s.replace(" ", "") for s in alpha]


            antigen = antigen_tokenizer.batch_decode(antigen_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            antigen = [s.replace(" ", "") for s in antigen]


            result_dict['beta'].append(beta)
            result_dict['alpha'].append(alpha)
            result_dict['antigen'].append(antigen)


    y_pred = np.concatenate(result_dict['y_pred'])



    test_df = pd.DataFrame({'beta': [v for l in result_dict['beta'] for v in l],
                            'alpha': [v for l in result_dict['alpha'] for v in l],
                            'antigen': [v for l in result_dict['antigen'] for v in l],
                            'y_pred': list(y_pred.flatten())})
    test_df = test_df.sort_values(by='y_pred', ascending=False).reset_index(drop=True)
    test_df.to_csv(join(config.log_dir, 'test_result.csv'), index=False)

    return test_df




def main(config):
    logger = config.get_logger('eval_generation')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)

    tcr_tokenizer = data_loader.get_tcr_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()

    # build model architecture, then print to console

    model = config.init_obj('arch', module_arch)
    logger.info('Loading checkpint from {}'.format(
        config['discriminator_resume']))
    checkpoint = torch.load(config['discriminator_resume'])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.to("cuda")






    """Test."""
    logger = config.get_logger('test')
    #test_4_input(model, data_loader, tcr_tokenizer, antigen_tokenizer, "cuda", config)
    test(model=model, data_loader=data_loader, tcr_tokenizer=tcr_tokenizer, antigen_tokenizer=antigen_tokenizer, device="cuda", config=config)
    # load best checkpoint
    # resume = str(config.save_dir / 'model_best.pth')
    # logger.info('Loading checkpoint: {} ...'.format(resume))
    # checkpoint = torch.load(resume)
    # state_dict = checkpoint['state_dict']
    # model.load_state_dict(state_dict)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)