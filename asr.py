from speechbrain.pretrained import EncoderDecoderASR
import streamlit as st

modules = torch.nn.ModuleDict(None)

def load_asr_model():
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech", savedir="pretrained_models/asr-crdnn-transformerlm-librispeech")
    return asr_model

def produce_transcript(path:str, model):
    return model.transcribe_file(path)

def encode_batch(wavs, wav_lens):
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        feats = modules.compute_features(wavs)
        feats = modules.normalize(feats, wav_lens)
        encoder_out = modules.asr_encoder(feats)
        return encoder_out

def transcribe_batch(self, wavs, wav_lens):

    with torch.no_grad():
        wav_lens = wav_lens.to("cpu")
        encoder_out = self.encode_batch(wavs, wav_lens)
        predicted_tokens, scores = modules.beam_searcher(
            encoder_out, wav_lens
        )
        predicted_words = [
            self.tokenizer.decode_ids(token_seq)
            for token_seq in predicted_tokens
        ]
    return predicted_words, predicted_tokens

def load_audio(path, savedir="."):

    source, fl = split_path(path)
    path = fetch(fl, source=source, savedir=savedir)
    signal, sr = torchaudio.load(path, channels_first=False)
    audio_normalizer = hparams.get("audio_normalizer", AudioNormalizer())

    return audio_normalizer(signal, sr)

def transcribe_file(path):

    waveform = load_audio(path)
    # Fake a batch:
    batch = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = self.transcribe_batch(
        batch, rel_length
    )
    return predicted_words[0]
