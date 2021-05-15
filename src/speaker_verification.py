import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderClassifier
import streamlit as st
import torch

@st.cache()
def load_sv_model():
    """
    Loads pre-trained speaker verification model from speechbrain
    """
    verif_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return verif_model

def compare_emeddings(emb1, emb2):
    """
    To run speaker verification on embeddings directly
    """
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = similarity(emb1, emb2)
    return score

def get_embedding(path:str, verif_model):
    """
    Gets the embeddings on a WAV file
    """

    signal, fs = torchaudio.load(path)
    embeddings = verif_model.encode_batch(signal)
    return embeddings

def run_sv(path:str, speaker_model, verif_model, threshold = 0.85):
    """
    Extracts embeddings of a wav file and compares it (cosine similarity) with existing speaker model
    """
    test_emb = get_embedding(path, verif_model)
    score = compare_emeddings(test_emb, speaker_model)
    return score, score > threshold