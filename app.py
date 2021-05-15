import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import pickle

from src.speaker_verification import *
from src.asr import *
from src.nlp import *
from src.vad import *

st.header("Mexican NLP Summer School")
#st.subheader("Voice assistant demo")

page = st.sidebar.selectbox("Choose the page", ["Enroll speaker", "Voice assistant"])

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
        write("temp/temp.wav", 16000, myrecording)

        # Run VAD
        vad_file = run_vad("temp/temp.wav", "temp/temp.wav")

        # Get and save embedding
        emb = get_embedding("temp/temp.wav", verif_model)
        with open("emb", 'wb') as pickle_file:
            pickle.dump(emb, pickle_file)

        st.success("Enrollment complete!")

elif page == "Voice assistant":
    
    st.subheader("Voice Assistant")

    with open("temp/emb", "rb") as f:
        speaker_model = pickle.load(f)

    if st.button("Please give your command"):
        with st.spinner("Recording..."):
            
            # Record user input command
            myrecording = sd.rec(4*16000, samplerate=16000, channels=1)
            sd.wait()

        write("temp/temp_command.wav", 16000, myrecording)
        score, decision = run_sv("temp/temp_command.wav", speaker_model, verif_model, 0.1)

        if decision:
            st.success("Access granted, score %s"%int(score*100))

            # Transcript
            transcript = produce_transcript("temp/temp_command.wav", asr_model)
            st.write("Transcript: ", transcript)

            # Detect topic
            detect_topic(transcript, ["weather forecasts", "calendar and meetings"], "This question is about {}", topic_model)
            ## DO NLP THERE

        else:
            st.error("Access denied")
