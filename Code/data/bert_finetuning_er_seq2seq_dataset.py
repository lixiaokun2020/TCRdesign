# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from os.path import join, exists
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from bert_data_prepare.tokenizer import get_tokenizer
from bert_data_prepare.utility import is_valid_aaseq


class Seq2SeqDataset(Dataset):
    def __init__(self, tcr_seqs,
                       antigen_seqs,
                       tcr_split_fun,
                       antigen_split_fun,
                       tcr_tokenizer,
                       antigen_tokenizer,
                       encoder_input,
                       TCR_max_len,
                       antigen_max_len,
                       logger):
        self.tcr_seqs = tcr_seqs
        self.antigen_seqs = antigen_seqs
        self.tcr_split_fun = tcr_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.tcr_tokenizer = tcr_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.encoder_input = encoder_input
        self.TCR_max_len = TCR_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

        self.logger.info(f"The input to the encoder is {encoder_input}")

    def __len__(self):
        return len(self.tcr_seqs)
        
    def __getitem__(self, i):
        tcr, antigen = self.tcr_seqs[i], self.antigen_seqs[i]

        input_data = {}
        tcr_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(tcr)),
                                                padding="max_length",
                                                max_length=self.TCR_max_len,
                                                truncation=True)
        antigen_tensor = self.antigen_tokenizer(self._insert_whitespace(self.antigen_split_fun(antigen)),
                                                  padding="max_length",
                                                  max_length=self.antigen_max_len,
                                                  truncation=True)

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        if self.encoder_input == 'tcr':
            input_data['input_ids'] = tcr_tensor['input_ids']
            input_data['attention_mask'] = tcr_tensor['attention_mask']
            input_data['labels'] = antigen_tensor.input_ids.copy()
            input_data['labels'] = [-100 if token == self.antigen_tokenizer.pad_token_id else token for token in input_data['labels']]
        
        elif self.encoder_input == 'antigen':
            input_data['input_ids'] = antigen_tensor['input_ids']
            input_data['attention_mask'] = antigen_tensor['attention_mask']
            input_data['labels'] = tcr_tensor['input_ids'].copy()
            input_data['labels'] = [-100 if token == self.tcr_tokenizer.pad_token_id else token for token in input_data['labels']]
        else:
            self.logger.info("Wrong encoder input!")
        
        input_data = {k: torch.tensor(v, dtype=torch.long) for k, v in input_data.items()}

        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized tcr: {tcr} -> {tcr_tensor['input_ids']}")
            self.logger.info(f"Example of tokenized antigen: {antigen} -> {antigen_tensor['input_ids']}")
            self.logger.info(f"Example of input_ids {input_data['input_ids']}")
            self.logger.info(f"Example of label: {input_data['labels']}")
            self._has_logged_example = True

        return input_data

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class TCRAntigenSeq2SeqDataset(object):
    def __init__(self, logger, 
                       seed,
                       data_dir, 
                    #    seq_dir,
                    #    neg_pair_save_dir, 
                    #    using_dataset, 
                       
                       tcr_vocab_dir,
                       antigen_vocab_dir,
                       tcr_tokenizer_dir,
                       antigen_tokenizer_dir,
                       tcr_tokenizer_name='common',
                       antigen_tokenizer_name='common',
                       tcr_token_length_list='2,3',
                       antigen_token_length_list='2,3',
                       TCR_max_len=None,
                       antigen_max_len=None,

                       valid_split=0.05,
                       TCR_seq_name='CDR3',
                       antigen_seq_name='Antigen',
                       encoder_input='Antigen',
                       shuffle=True):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        # self.seq_dir = seq_dir
        # self.neg_pair_save_dir = neg_pair_save_dir
        # self.using_dataset = list(using_dataset.split(','))
        self.TCR_seq_name = TCR_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.valid_split = valid_split
        self.encoder_input = encoder_input

        self.shuffle = shuffle
        self.TCR_max_len = TCR_max_len
        self.antigen_max_len = antigen_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()
        train_pair_df, valid_pair_df, test_pair_df = self._split_dataset()
        self.valid_pair_df = valid_pair_df
        self.test_pair_df = test_pair_df

        self.logger.info(f'Creating {TCR_seq_name} tokenizer...')
        self.TCRTokenizer = get_tokenizer(tokenizer_name=tcr_tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=tcr_vocab_dir,
                                              token_length_list=tcr_token_length_list)
        self.tcr_tokenizer = self.TCRTokenizer.get_bert_tokenizer(
            max_len=self.TCR_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        self.logger.info(f'Creating {antigen_seq_name} tokenizer...')
        self.AntigenTokenizer = get_tokenizer(tokenizer_name=antigen_tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=antigen_vocab_dir,
                                               token_length_list=antigen_token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=antigen_tokenizer_dir)

        self.train_dataset = self._get_dataset(pair_df=train_pair_df)
        self.valid_dataset = self._get_dataset(pair_df=valid_pair_df)
        self.test_dataset = self._get_dataset(pair_df=test_pair_df)

    def get_tcr_split_fn(self):
        return self.TCRTokenizer.split

    def get_antigen_split_fn(self):
        return self.AntigenTokenizer.split

    def get_valid_pair_df(self):
        return self.valid_pair_df

    def get_test_pair_df(self):
        return self.test_pair_df

    def get_train_dataset(self):
        return self.train_dataset

    def get_valid_dataset(self):
        return self.valid_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_tcr_tokenizer(self):
        return self.tcr_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer

    def _get_dataset(self, pair_df):
        er_dataset = Seq2SeqDataset(tcr_seqs=list(pair_df[self.TCR_seq_name]),
                                    antigen_seqs=list(pair_df[self.antigen_seq_name]),
                                    tcr_split_fun=self.TCRTokenizer.split,
                                    antigen_split_fun=self.AntigenTokenizer.split,
                                    tcr_tokenizer=self.tcr_tokenizer,
                                    antigen_tokenizer=self.antigen_tokenizer,
                                    encoder_input=self.encoder_input,
                                    TCR_max_len=self.TCR_max_len,
                                    antigen_max_len=self.antigen_max_len,
                                    logger=self.logger)
        return er_dataset

    def _split_dataset(self):
        train_pair_df , test_pair_df = train_test_split(self.pair_df , test_size=self.valid_split * 2, random_state=self.seed)
        valid_pair_df , test_pair_df = train_test_split(test_pair_df , test_size=0.5, random_state=self.seed)
        self.logger.info(f"{len(train_pair_df)} train and {len(valid_pair_df)} valid and {len(test_pair_df)} test.")

        return train_pair_df, valid_pair_df, test_pair_df

    def _create_pair(self):
        df = pd.read_csv(self.data_dir)
        return df



from base import BaseDataLoader
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
                       beta_seq_name='Beta',
                       alpha_seq_name='Alpha',
                       label_name='Label',
                       #receptor_seq_name='beta',
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

        
        self.BetaTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=tcr_vocab_dir,
                                              token_length_list=token_length_list)
        self.beta_tokenizer = self.BetaTokenizer.get_bert_tokenizer(
            max_len=self.beta_max_len, 
            tokenizer_dir=tcr_tokenizer_dir)



        self.AlphaTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=tcr_vocab_dir,
                                              token_length_list=token_length_list)
        self.alpha_tokenizer = self.AlphaTokenizer.get_bert_tokenizer(
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
                                        # labels = list(pair_df[self.label_name]),
                                        tcr_split_fun = self.BetaTokenizer.split,
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
        self.logger.info(f'Select {self.test_tcrs} from {len(tcr_list)} antibody')
        selected_antibodys = [tcr_list[i] for i in selected_tcr_index_list]
        test_pair_df = self.pair_df[self.pair_df['antibody'].isin(selected_antibodys)]
        #test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'), index=False)

        selected_antibodys = list(set(test_pair_df['antibody']))
        train_valid_pair_df = self.pair_df[~self.pair_df['antibody'].isin(selected_antibodys)]
            
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
        self.logger.info(f'Loading {self.TCR_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'antibody': 'antibody'}
        keep_columns = [column_map_dict[c] for c in [self.TCR_seq_name, self.receptor_seq_name]]
        
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

        column_rename_dict = {column_map_dict[c]: c for c in [self.TCR_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter

class TPDataset(Dataset):
    def __init__(self, beta_seqs,
                       alpha_seqs,
                       antigen_seqs,
                    #    labels,
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
        # self.labels = labels
        self.tcr_split_fun = tcr_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.tcr_tokenizer = tcr_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.TCR_max_len = TCR_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.beta_seqs)
        
    def __getitem__(self, i):
        beta,alpha,antigen = self.beta_seqs[i], self.alpha_seqs[i] , self.antigen_seqs[i]

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
        

        # label_tensor = torch.FloatTensor(np.atleast_1d(label))


        beta_tensor = {k: torch.squeeze(v) for k, v in beta_tensor.items()}
        alpha_tensor = {k: torch.squeeze(v) for k, v in alpha_tensor.items()}
        antigen_tensor = {k: torch.squeeze(v) for k,v in antigen_tensor.items()}
        return beta_tensor, alpha_tensor, antigen_tensor, #, label_tensor




    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)
