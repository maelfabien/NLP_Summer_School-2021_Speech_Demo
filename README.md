# Mexican NLP Summer School - Demo

## An introduction to speech-based technologies for Natural Language Processing applications

This repository contains the code for the workshop given at the Mexican NLP Summer School. The link to the slides is [here](https://docs.google.com/presentation/d/1bXqvxy0KQnI3AhsncHj_26p1WdE-UKErplUBJ5BBANI/edit?usp=sharing).

### How to install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### How to run the app

```bash
streamlit run app.py
```

## What it does

After enrolling the speaker, it verifies the voice identity of the speaker, runs ASR transcripts, and identifies the topic of your query:

![](pictures/demo.png)
