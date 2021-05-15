import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import pickle

from speaker_verification import *
from asr import *
from nlp import *

st.header("Mexican NLP Summer School")
st.subheader("Voice assistant demo")

page = st.sidebar.selectbox("Choose the page", ["Test SV", "Test ASR", "Enroll speaker", "Voice assistant"])

# Load models
verif_model = load_sv_model()
asr_model = load_asr_model()
topic_model = load_topic_model()

if page == "Enroll speaker":

    st.subheader("Speaker Enrollment")

    if st.button("Start Recording for 4 seconds"):
        with st.spinner("Recording..."):
            
            # Record user input
            myrecording = sd.rec(4*16000, samplerate=16000, channels=1)
            sd.wait()
            write("temp.wav", 16000, myrecording)

            # Get and save embedding
            emb = get_embedding("temp.wav", verif_model)
            pickle.dump(emb, "emb")

if page == "Voice Assistant":
    
    st.subheader("Voice Assistant")

    if st.button("Please give your command"):
        with st.spinner("Recording..."):
            
            # Record user input command
            myrecording = sd.rec(4*16000, samplerate=16000, channels=1)
            sd.wait()
            write("temp_command.wav", 16000, myrecording)

            # Compare with saved speaker model
            emb_spk_model = pickle.load("emb")
            score, decision = run_sv("temp_command.wav", speaker_model, verif_model, 0.85)
            st.write(score, decision)

            if decision:
                st.write("Transcript: ")

                # Transcript
                transcript = produce_transcript("temp_command.wav", asr_model)
                
                # Detect topic
                detect_topic(transcript, ["weather forecasts", "calendar and meetings"], "This question is about {}", topic_model)
                ## DO NLP THERE

            else:
                st.write("Access denied")
