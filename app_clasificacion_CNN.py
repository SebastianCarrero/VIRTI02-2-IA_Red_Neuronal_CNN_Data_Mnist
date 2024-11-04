import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Cargar el modelo entrenado y guardado
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_manusc_mejorado.h5')

model = load_model()

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28 píxeles
    image = np.array(image).reshape(28, 28, 1).astype('float32') / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Agregar un eje para el batch
    return image

# Título de la aplicación
st.title('Clasificación de Dígitos Manuscritos con CNN')

# Cargar la imagen
uploaded_file = st.file_uploader('Cargar una imagen de un dígito (0-9)', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Hacer la predicción
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Mostrar la predicción
    st.markdown(f'<h3>Predicción: <b>{predicted_digit}</b></h3>', unsafe_allow_html=True)

    # Visualizar las probabilidades de predicción para cada dígito con etiquetas y porcentajes
    st.subheader("Confianza de la predicción para cada dígito")
    fig, ax = plt.subplots()
    sns.barplot(x=list(range(10)), y=prediction[0], ax=ax, palette='viridis')
    ax.set_xlabel('Dígito')
    ax.set_ylabel('Probabilidad')

    # Añadir etiquetas encima de cada barra con el porcentaje
    for i, p in enumerate(prediction[0]):
        ax.text(i, p + 0.01, f'{p * 100:.2f}%', ha='center')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    # Mostrar imagen preprocesada en una sección separada
    st.subheader("Imagen Preprocesada")
    fig_preprocessed, ax_preprocessed = plt.subplots()
    ax_preprocessed.imshow(processed_image[0].reshape(28, 28), cmap='gray')  # Convertir a 28x28 para visualizar
    ax_preprocessed.axis('off')
    st.pyplot(fig_preprocessed)

