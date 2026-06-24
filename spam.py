%%writefile spam.py

import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label","message"]
)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(
    data["message"]
)

y = data["label"]

model = MultinomialNB()

model.fit(X,y)

st.title("Spam Detection")

message = st.text_area(
    "Enter Message"
)

if st.button("Predict"):

    transform = vectorizer.transform(
        [message]
    )

    result = model.predict(
        transform
    )

    st.success(result[0])