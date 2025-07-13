# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join, exists
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from bert_data_prepare.tokenizer import get_tokenizer
from base import BaseDataLoader
from bert_data_prepare.utility import is_valid_aaseq
from transformers import AutoTokenizer


class TCR_Antigen_Dataset_AnDan(BaseDataLoader):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       validation_split,
                       test_split,
                       num_workers,
                       data_dir, 
                       tcr_vocab_dir,
                       tcr_tokenizer_dir,
                       tokenizer_name='common',
                       #receptor_tokenizer_name='common',
                       token_length_list='2,3',
                       #receptor_token_length_list='2,3',
                       antigen_seq_name='antigen',
                       beta_seq_name ='TCR_Beta',
                       alpha_seq_name='TCR_Alpha',
                       label_name='Label',
                       test_tcrs=100,
                       #neg_ratio=1.0,
                       shuffle=True,
                       antigen_max_len=None,
                       beta_max_len=None,
                       alpha_max_len=None,):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.beta_seq_name = beta_seq_name
        self.alpha_seq_name = alpha_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.label_name = label_name

        self.test_tcrs = test_tcrs
        self.shuffle = shuffle
        self.beta_max_len = beta_max_len
        self.alpha_max_len = alpha_max_len
        self.antigen_max_len = antigen_max_len

        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()

        
        self.betaTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=tcr_vocab_dir,
                                              token_length_list=token_length_list)
        self.beta_tokenizer = self.betaTokenizer.get_bert_tokenizer(
            max_len=self.beta_max_len, 
            tokenizer_dir=tcr_tokenizer_dir)



        self.alphaTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=tcr_vocab_dir,
                                              token_length_list=token_length_list)
        self.alpha_tokenizer = self.alphaTokenizer.get_bert_tokenizer(
            max_len=self.alpha_max_len, 
            tokenizer_dir=tcr_tokenizer_dir)

        
        self.AntigenTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=tcr_vocab_dir,
                                               token_length_list=token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=tcr_tokenizer_dir)



        esm_dir = '/data/coding/TCRdesign/Code/esm2/esm2_150m/'
        
        self.antigen_tokenizer = AutoTokenizer.from_pretrained(esm_dir,cache_dir = "../esm2/esm2_150m/",max_len = self.antigen_max_len)


        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)


    def get_beta_tokenizer(self):
        return self.beta_tokenizer

    def get_alpha_tokenizer(self):
        return self.alpha_tokenizer

    def get_tcr_tokenizer(self):
        return self.beta_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer


    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        tp_dataset = TPDataset(
                                        beta_seqs = list(pair_df[self.beta_seq_name]),
                                        alpha_seqs = list(pair_df[self.alpha_seq_name]),
                                        antigen_seqs = list(pair_df[self.antigen_seq_name]),

                                        labels = list(pair_df[self.label_name]),
                                        tcr_split_fun = self.betaTokenizer.split,
                                        antigen_split_fun = self.AntigenTokenizer.split,
                                        tcr_tokenizer = self.beta_tokenizer,
                                        antigen_tokenizer = self.antigen_tokenizer,

                                        TCR_max_len = self.beta_max_len,
                                        antigen_max_len = self.antigen_max_len,

                                        logger = self.logger
                               )
        return tp_dataset

    def _split_dataset(self):
        # if exists(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv')):
        #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv'))
        #     self.logger.info(f'Loading created unseen tcrs for test with shape {test_pair_df.shape}')
        
        tcr_list = list(set(self.pair_df['tcr']))
        selected_tcr_index_list = self.rng.integers(len(tcr_list), size=self.test_tcrs)
        self.logger.info(f'Select {self.test_tcrs} from {len(tcr_list)} tcr')
        selected_tcrs = [tcr_list[i] for i in selected_tcr_index_list]
        test_pair_df = self.pair_df[self.pair_df['tcr'].isin(selected_tcrs)]
        #test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv'), index=False)

        selected_tcrs = list(set(test_pair_df['tcr']))
        train_valid_pair_df = self.pair_df[~self.pair_df['tcr'].isin(selected_tcrs)]
            
        self.logger.info(f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)

        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")

        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.tcr_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'tcr': 'tcr'}
        keep_columns = [column_map_dict[c] for c in [self.tcr_seq_name, self.receptor_seq_name]]
        
        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.tcr_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter

class TPDataset(Dataset):
    def __init__(self, beta_seqs,
                       alpha_seqs,
                       antigen_seqs,
                       labels,
                       tcr_split_fun,
                       antigen_split_fun,
                       tcr_tokenizer,
                       antigen_tokenizer,
                       TCR_max_len,
                       antigen_max_len,
                       logger):
        self.beta_seqs = beta_seqs
        self.alpha_seqs = alpha_seqs
        self.antigen_seqs = antigen_seqs
        self.labels = labels
        self.tcr_split_fun = tcr_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.tcr_tokenizer = tcr_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.TCR_max_len = TCR_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        beta, alpha, antigen= self.beta_seqs[i], self.alpha_seqs[i] , self.antigen_seqs[i]
        label = self.labels[i]
        beta_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(beta)),
                                                padding="max_length",
                                                max_length=self.TCR_max_len,
                                                truncation=True,
                                                return_tensors="pt")

        alpha_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(alpha)),
                                                padding="max_length",
                                                max_length=self.TCR_max_len,
                                                truncation=True,
                                                return_tensors="pt")

        antigen_tensor = self.antigen_tokenizer(antigen,
                                                  padding="max_length",
                                                  max_length=self.antigen_max_len,
                                                  truncation=True,
                                                  return_tensors="pt")

        

        label_tensor = torch.FloatTensor(np.atleast_1d(label))


        beta_tensor = {k: torch.squeeze(v) for k, v in beta_tensor.items()}
        alpha_tensor = {k: torch.squeeze(v) for k, v in alpha_tensor.items()}
        antigen_tensor = {k: torch.squeeze(v) for k,v in antigen_tensor.items()}

        return beta_tensor, alpha_tensor, antigen_tensor, label_tensor




    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)

