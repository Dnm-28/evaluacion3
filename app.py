import streamlit as st 
import joblib

model = joblib.load('modelo_naive_bayes.pkl')
vectorizer = joblib.load('vectorizer.pkl')

user_input = st.text_area("Ingrese la noticia")

def predict_category(message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return prediction[0]

if st.button("Clasificar"):
    if user_input:
        category = predict_category(user_input)
        st.write(f'Categoría Predicha: {category}')
else:
        st.title("Ingrese un texto a analizar")

