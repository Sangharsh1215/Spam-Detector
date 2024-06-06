# SMS Spam Detector

This repository contains the implementation of an SMS Spam Detector using machine learning. The model is trained to classify SMS messages as either spam or ham (not spam).


## Introduction

The SMS Spam Detector is a machine learning project designed to filter out spam messages from SMS communications. It leverages natural language processing (NLP) techniques to analyze the content of messages and predict their classification.

## Dataset

The model is trained on the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The dataset contains a set of SMS labeled messages that have been collected for mobile phone spam research.

- **ham**: Legitimate (non-spam) messages
- **spam**: Unsolicited (spam) messages

## Usage

import joblib

model = joblib.load('spam_detector_model.pkl')

message = ["Congratulations! You've won a $1000 gift card. Click here to claim now."]
prediction = model.predict(message)

print("Spam" if prediction[0] == 1 else "Ham")


## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/Sangharsh1215/Spam-Detector.git
cd Spam-Detector
pip install -r requirements.txt

