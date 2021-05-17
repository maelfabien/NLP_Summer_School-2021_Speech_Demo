import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import pickle
import os
import numpy as np

from src.speaker_verification import *
from src.asr import *
from src.nlp import *
from src.vad import *
from src.weather import *

st.sidebar.image("pictures/ampln.png")
#st.header("Voice assistant demo")

page = st.sidebar.selectbox("Choose the page", ["Enroll speaker", "Voice assistant"])

# Load models
verif_model = load_sv_model()
asr_model = load_asr_model()
topic_model = load_topic_model()

if page == "Enroll speaker":

    st.header("Speaker Enrollment")

    if st.button("Start Recording for 4 seconds"):
        with st.spinner("Recording..."):
            
            # Record user input
            myrecording = sd.rec(4*16000, samplerate=16000, channels=1)
            sd.wait()

        os.system("rm temp/temp.wav")
        write("temp/temp.wav", 16000, myrecording)

        # Run VAD
        vad_file = run_vad("temp/temp.wav", "temp/temp.wav")

        # Get and save embedding
        emb = get_embedding("temp/temp.wav", verif_model)

        with open("temp/emb", 'wb') as pickle_file:
            pickle.dump(emb, pickle_file)

        st.success("Enrollment complete!")

elif page == "Voice assistant":
    
    st.header("Voice Assistant")

    with open("temp/emb", "rb") as f:
        speaker_model = pickle.load(f)

    if st.button("Please give your command"):
        with st.spinner("Recording..."):
            
            # Record user input command
            myrecording = sd.rec(4*16000, samplerate=16000, channels=1)
            sd.wait()

        os.system("rm temp/temp_command.wav")
        write("temp/temp_command.wav", 16000, myrecording)
        score, decision = run_sv("temp/temp_command.wav", speaker_model, verif_model, 0.1)

        if decision:
            st.success("Access granted with a score of %s"%int(score*100))

            # Transcript
            #transcript = produce_transcript("temp/temp_command.wav", asr_model)
            transcript = asr_model.transcribe_file("temp/temp_command.wav")
            st.write("Transcript: ", transcript)

            # Detect topic
            topic = detect_topic(transcript, ["weather forecasts", "calendar and meetings"], "This question is about {}", topic_model)
            
            ## DO NLP THERE

            if topic == "weather forecasts":
                city = detect_ner(transcript)
                current_temperature, weather_description = find_temperature(city)

                if weather_description != "None":
                    sentence = "The temperature is %s °C with %s"%(np.round(current_temperature, 1), weather_description)
                    st.write(sentence)
                else:
                    st.write("City not found")


        else:
            st.error("Access denied with a score of %s"%int(score*100))
