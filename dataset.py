import torch
import torch.nn as nn
import torch.utils.data as Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds,tockenizer_src,tockenizer_tgt,src_lang,tgt_lang,seq_len)->None:
        super().__init__()
        self.ds = ds
        self.tockenizer_src = tockenizer_src
        self.tockenizer_tgt = tockenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor(tockenizer_tgt.token_to_id("[SOS]"),dtype=torch.int64)
        self.eos_token = torch.Tensor(tockenizer_tgt.token_to_id("[EOS]"),dtype=torch.int64)    
        self.pad_token = torch.Tensor(tockenizer_tgt.token_to_id("[PAD]"),dtype=torch.int64)
       
    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index:any)->any:
        src_target_pair=self.ds[index]
        src_text= src_target_pair['translation'][self.src_lang]
        tgt_text= src_target_pair['translation'][self.tgt_lang]

        enc_input_tockens=self.tockenizer_src.encode(src_text).ids
        dec_input_tockens=self.tockenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tockens=self.seq_len-len(enc_input_tockens)-2
        dec_num_padding_tockens=self.seq_len-len(dec_input_tockens)-1

        if enc_num_padding_tockens<0 or dec_num_padding_tockens<0:
            raise ValueError("Input sequence is too long.")
        

        encoder_input=torch.cat(
            [
                self.sos_token,
                torch.Tensor(enc_input_tockens,dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token]*enc_num_padding_tockens,dtype=torch.int64)
            ]
        )

        decoder_input=torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tockens,dtype=torch.int64),
                torch.Tensor([self.pad_token]*dec_num_padding_tockens,dtype=torch.int64)
            ]
        )

        label=torch.cat(
            [
                torch.Tensor(dec_input_tockens,dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token]*dec_num_padding_tockens,dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return{
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def casual_mask(size):
    mask=torch.triu(torch.ones(1,size, size), diagonal=1).type(torch.int)
    return mask==0