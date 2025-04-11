import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tockenizers import Tokenizer
from tockenizer.models import Wordlevel
from tockenizer.trainers import WordlevelTrainer
from tockenizer.pre_tockenizers import Whitespace
from dataset import BilingualDataset,casual_mask
from model import build_transformer

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang] 

def get_or_build_tockenizer(config,ds,lang):
    tockenizer_path=Path(config["tockenizer_path"].format(lang))
    if not Path.exists(tockenizer_path):
        tockenizer=Tokenizer(Wordlevel(unk_token="[UNK]"))
        tockenizer.pre_tokenizer=Whitespace()
        trainer=WordlevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tockenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tockenizer.save(tockenizer_path)
    else:
        tockenizer=Tokenizer.from_file(tockenizer_path)
    return tockenizer

def get_ds(config):

    raw_ds=load_dataset('opus_books',f'{config["lang1"]}-{config["lang2"]}',split='train')

    tockenizer_src=get_or_build_tockenizer(config,raw_ds,config["lang1"])
    tockenizer_tgt=get_or_build_tockenizer(config,raw_ds,config["lang2"])

    train_ds_size=int(len(raw_ds)*0.9)
    valid_ds_size=len(raw_ds)-train_ds_size
    train_ds_raw, valid_ds_raw=random_split(raw_ds,[train_ds_size,valid_ds_size])

    train_ds=BilingualDataset(train_ds_raw,tockenizer_src,tockenizer_tgt,config["lang1"],config["lang2"],config["seq_len"])
    valid_ds=BilingualDataset(valid_ds_raw,tockenizer_src,tockenizer_tgt,config["lang1"],config["lang2"],config["seq_len"])

    max_len_src=0
    max_len_tgt=0
    
    for item in raw_ds:
        src_ids=tockenizer_src.encode(item["translation"][config["lang1"]]).ids
        tgt_ids=tockenizer_tgt.encode(item["translation"][config["lang2"]]).ids
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f"Max length src: {max_len_src}, Max length tgt: {max_len_tgt}")
    
    train_dataloader=DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_dataloader=DataLoader(valid_ds,batch_size=1,shuffle=True)

    return train_dataloader, val_dataloader, tockenizer_src, tockenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(vocab_src_len,vocab_tgt_len,config["seq_len"],config["seq_len"],config["d_model"])
    return model
    
