#
import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load models
@st.cache_resource
def load_models():

    model = load_model("models/model.keras")

    feature_extractor = load_model(
        "models/feature_extractor.keras"
    )

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, feature_extractor, tokenizer


# Generate caption
def generate_caption(
    image_path,
    model,
    feature_extractor,
    tokenizer
):

    # Image preprocessing
    img = load_img(
        image_path,
        target_size=(224, 224)
    )

    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Extract features
    features = feature_extractor.predict(
        img,
        verbose=0
    )

    # Generate caption
    text = "startseq"

    for _ in range(34):

        sequence = tokenizer.texts_to_sequences(
            [text]
        )[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=34
        )

        prediction = model.predict(
            [features, sequence],
            verbose=0
        )

        word_id = np.argmax(prediction)

        word = tokenizer.index_word.get(
            word_id
        )

        if word is None or word == "endseq":
            break

        text += " " + word

    return text.replace(
        "startseq",
        ""
    ).strip()


# Streamlit UI
st.title("🖼️ Image Caption Generator")

image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if image:

    st.image(
        image,
        width=500
    )

    if st.button("Generate Caption"):

        model, feature_extractor, tokenizer = load_models()

        with open(
            "uploaded_image.jpg",
            "wb"
        ) as f:
            f.write(image.getbuffer())

        with st.spinner("Generating caption..."):

            caption = generate_caption(
                "uploaded_image.jpg",
                model,
                feature_extractor,
                tokenizer
            )

     
        st.success("Generated Caption")
        st.write(caption)
        

