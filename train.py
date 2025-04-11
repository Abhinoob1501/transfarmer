import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import os
import warnings as Warning
from datasets import load_dataset
from tockenizers import Tokenizer
from tockenizer.models import Wordlevel
from tockenizer.trainers import WordlevelTrainer
from tockenizer.pre_tockenizers import Whitespace
from dataset import BilingualDataset,casual_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_weight_file_path,get_config
from tqdm import tqdm
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
    
def train_model(config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tockenizer_src, tockenizer_tgt=get_ds(config)
    model=get_model(config,tockenizer_src.get_vocab_size(),tockenizer_tgt.get_vocab_size()).to(device)

    writer=SummaryWriter(config["experiment_name"])

    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)

    initial_epoch=0
    global_step=0
    
    if config['preload']:
        model_filename=get_weight_file_path(config,config['preload'])
        print(f"Loading model from {model_filename}")
        state=torch.load(model_filename)   
        initial_epoch=state["epoch"]+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state["global_step"]
    
    loss_fn=nn.CrossEntropyLoss(ignore_index=tockenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator=tqdm(train_dataloader,desc=f" Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input=batch["encoder_input"].to(device)
            decoder_input=batch["decoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            decoder_mask=batch["decoder_mask"].to(device)

            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(decoder_input,encoder_output,encoder_mask,decoder_mask)
            proj_output=model.project(decoder_output)

            label=batch["label"].to(device)

            loss=loss_fn(proj_output.view(-1,tockenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({f" loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss",loss.item(),global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step+=1
        
    model_filename=get_weight_file_path(config,f"{epoch:02d}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }, model_filename)

if  __name__=="__main__":
    Warning.filterwarnings("ignore")
    config=get_config()
    train_model(config)

