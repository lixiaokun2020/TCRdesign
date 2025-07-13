# -*- coding: utf-8 -*-

from transformers import EncoderDecoderModel

def get_EncoderDecoder_model(logger, 
                             TransformerVariant, 
                             TCRBert_dir,
                             AntigenBert_dir,
                             tcr_tokenizer,
                             antigen_tokenizer,
                             TCR_max_len,
                             antigen_max_len,
                             resume=None):
    if resume is not None:
        logger.info(f'Loading EncoderDecoder from {resume}')
        model = EncoderDecoderModel.from_pretrained(resume)
        model.config.decoder_start_token_id = tcr_tokenizer.cls_token_id
        model.config.eos_token_id = tcr_tokenizer.sep_token_id
        model.config.pad_token_id = tcr_tokenizer.pad_token_id
        model.config.vocab_size = tcr_tokenizer.vocab_size
        model.config.max_length = TCR_max_len
        return model
    
    """Load the bert model"""
    logger.info(f'Loading TCRBert from {TCRBert_dir}')
    logger.info(f'Loading AntigenBert from {AntigenBert_dir}')

    if TransformerVariant == 'TCR-Antigen':
        logger.info("Using TCRBert as encoder, AntigenBert as decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            TCRBert_dir, AntigenBert_dir)
        model.config.decoder_start_token_id = antigen_tokenizer.cls_token_id
        model.config.eos_token_id = antigen_tokenizer.sep_token_id
        model.config.pad_token_id = antigen_tokenizer.pad_token_id
        model.config.vocab_size = antigen_tokenizer.vocab_size
        model.config.max_length = antigen_max_len

    else:
        logger.info("Using AntigenBert as encoder, TCRBert as decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            AntigenBert_dir, TCRBert_dir)
        model.config.decoder_start_token_id = tcr_tokenizer.cls_token_id
        model.config.eos_token_id = tcr_tokenizer.sep_token_id
        model.config.pad_token_id = tcr_tokenizer.pad_token_id
        model.config.vocab_size = tcr_tokenizer.vocab_size
        model.config.max_length = TCR_max_len

    return model