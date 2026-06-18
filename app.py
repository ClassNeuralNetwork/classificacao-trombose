import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('modelo_trombose_com_class_weights.h5')

CLASSES = ["Trombose", "Não Trombose"]

INPUT_SIZE = (224, 224)

def processar(imagem: Image.Image) -> np.ndarray:
    imagem = imagem.convert('RGB')
    imagem = imagem.resize(INPUT_SIZE)
    arr = np.array(imagem, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def prever(imagem: Image.Image):
    entrada = processar(imagem)

    probabilidades = np.random.dirichlet(np.ones(len(CLASSES)))
    classe_pred = CLASSES[int(np.argmax(probabilidades))]
    confianca = float(np.max(probabilidades))
    return classe_pred, confianca, probabilidades

st.set_page_config(page_title="Classificação de Trombose", layout="centered")
st.title("Classificação de Trombose")
st.write("Faça upload de uma imagem para classificar se há presença de trombose.")

st.divider()
fonte = st.radio("Selecione a fonte da imagem:", options=["Upload de Arquivo", "URL da Imagem"], horizontal=True)

imagem_pil: Image.Image | None = None

if fonte == "Upload de Arquivo":
    arquivo = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])
    if arquivo is not None:
        imagem_pil = Image.open(arquivo)

else:
    foto = st.camera_input("Tire uma foto ou envie uma imagem")
    if foto is not None:
        imagem_pil = Image.open(foto)

if imagem_pil is not None:
    st.divider()
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Imagem Recebida")
        st.image(imagem_pil, use_container_width=True)
        
    with col_result:
        st.subheader("Resultado da Classificação")
        with st.spinner("Classificando..."):
            classe, confianca, probabilidades = prever(imagem_pil)
        
        st.success(f"Classe Predita: {classe}")
        st.metric("Confiança", f"{confianca * 100:.1f}%")

        st.divider()
        st.write("Probabilidades para cada classe:")
        for nome, prob in zip(CLASSES, probabilidades):
            st.progress(float(prob), text=f"{nome}: {prob * 100:.1f}%")
        
else:
    st.info("Aguardando o upload ou captura de uma imagem para classificação.")

st.divider()
st.caption("Template generico - adapte `preprocessar()`, `prever()` e `CLASSES`" "para seu modelo.")