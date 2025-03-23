#!/bin/bash
# Install dependencies listed in requirements.txt
pip install --no-cache-dir -r requirements.txt

# Download the spaCy model manually
python -m spacy download en_core_web_sm