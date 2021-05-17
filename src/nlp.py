from transformers import pipeline
import streamlit as st
import spacy
import plotly.express as px
import spacy
import spacy_streamlit

model = "en_core_web_sm"
nlp = spacy.load(model)

@st.cache(allow_output_mutation=True)
def load_topic_model():
    classifier = pipeline("zero-shot-classification")
    return classifier
    
def detect_topic(text, topics, hypothesis_template, classifier):

    dict_res = classifier([text], topics, hypothesis_template=hypothesis_template, multi_class=True)
    fig1 = px.bar(y=dict_res['labels'][::-1], x=dict_res['scores'][::-1], labels = {'x':'', 'y':''}, width=500, height=250)
    st.plotly_chart(fig1)

    identified_topic = dict_res['labels'][::-1][-1]
    return identified_topic

def detect_ner(text):

    doc = nlp(text)

    to_return = []
    models = [model]
    visualizers = ["ner"]
    spacy_streamlit.visualize(models, text, visualizers)
    
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
