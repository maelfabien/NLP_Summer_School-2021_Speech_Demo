from speechbrain.pretrained import EncoderDecoderASR
import streamlit as st
import torch

modules = torch.nn.ModuleDict(None)

def load_asr_model():
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_model/")
    return asr_model
