from transformers import pipeline
import streamlit as st
import spacy
import plotly.express as px

@st.cache(allow_output_mutation=True)
def load_topic_model():
    classifier = pipeline("zero-shot-classification")
    return classifier

@st.cache()
def detect_topic(text, topics, hypothesis_template, classifier):

    dict_res = classifier([row], topics, hypothesis_template=hypothesis_template, multi_class=True)
    fig1 = px.bar(y=dict_res['labels'][::-1], x=dict_res['scores'][::-1], labels = {'x':'', 'y':''}, width=500, height=250)
    st.plotly_chart(fig1)
    return dict_res
