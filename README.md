Image Caption Generator using DenseNet201 and LSTM
1. Project Overview
This project generates natural-language captions for uploaded images
using a deep-learning pipeline. It combines DenseNet201 for
image-feature extraction, an LSTM network for word-by-word caption
generation, TensorFlow/Keras for model development, and Streamlit for
the web interface.

The project covers dataset preparation, image preprocessing, text
tokenization, feature extraction, model training, caption generation,
and a Streamlit application.
2. Project Workflow
1. Load the image-caption dataset from a CSV file.
2. Read and resize images to 224 × 224 pixels.
3. Preprocess captions by converting them to lowercase, removing
   one-character words, and adding startseq and endseq tokens.
4. Fit a Keras tokenizer and calculate the vocabulary size and maximum
   caption length.
5. Use pretrained DenseNet201 to extract image features.
6. Generate training examples with a custom Keras Sequence data
   generator. Each example contains image features and a partial
   caption; the target is the next word.
7. Train a model that combines image features, an embedding layer, and
   an LSTM.
8. Save the caption model, feature extractor, and tokenizer.
9. Use the saved artifacts to generate captions for new images.
10. Run the Streamlit application to upload an image and display its
    generated caption.
3. Model Architecture
3.1 DenseNet201 Image Feature Extractor
The project uses pretrained DenseNet201 to extract visual information
from an image. The notebook creates a feature-extraction model using the
output of DenseNet201's penultimate layer. The resulting feature vector
is passed to the caption-generation network.
3.2 LSTM Caption Generation Model
The caption model has two inputs:
- Image input: a 1,920-dimensional DenseNet201 feature vector,
  projected to 256 dimensions.
- Text input: a padded sequence of token IDs, passed through a
  256-dimensional embedding layer.
The image representation and word embeddings are concatenated and
processed by an LSTM with 256 units. Dropout layers are used for
regularization, followed by dense layers and a softmax output over the
vocabulary. The model is compiled with categorical cross-entropy loss
and the Adam optimizer.
4. Technologies Used
- Python
- TensorFlow and Keras
- DenseNet201
- LSTM
- NumPy and Pandas
- Matplotlib and Seaborn
- Pillow
- Streamlit
- Pickle
5. Requirements and Installation
Install the main packages used by the notebook and Streamlit
application:
pip install tensorflow numpy pandas matplotlib seaborn tqdm pillow streamlit
The notebook also uses Python standard-library modules such as os,
warnings, textwrap, and pickle. Use a Python version and
TensorFlow release compatible with your operating system and hardware.
6. Dataset
The training notebook expects:
- An image directory containing the image files.
- A CSV file named captions.txt.zip that can be read by
  pandas.read_csv.
- The CSV must contain the columns image and caption.
The original notebook uses local Windows paths. Update them to match
your machine. For example:
image_path = "path/to/Images"
data = pd.read_csv("path/to/captions.txt.zip")
Keep the image filenames in the CSV consistent with the files in the
image directory.
7. Training the Model
1. Clone or download this repository.
2. Place the image dataset and caption CSV in accessible locations.
3. Open test(2).ipynb in Jupyter Notebook, JupyterLab, or VS Code.
4. Update the dataset paths in the notebook.
5. Run the notebook cells in order to preprocess the data, extract
   image features, and train the model.
6. Save the trained caption model, tokenizer, and feature extractor.
7. Place the saved artifacts in the models/ directory using the
   filenames described below.
Training Configuration
The notebook's training configuration includes:
- Image size: 224 × 224
- Image feature dimension: 1,920
- Embedding dimension: 256
- LSTM units: 256
- Batch size: 64
- Maximum epochs: 50
- Loss: categorical cross-entropy
- Optimizer: Adam
- Callbacks: model checkpointing, early stopping, and learning-rate
  reduction on validation loss
8. Project Structure and Saved Models
The Streamlit application expects this structure:
project/
│
├── app.py
├── test(2).ipynb
├── README.md
│
└── models/
    ├── model.keras
    ├── feature_extractor.keras
    └── tokenizer.pkl
  File                                Purpose
  models/model.keras                Trained caption-generation model

  models/feature_extractor.keras    Saved DenseNet201-based image
feature extractor
models/tokenizer.pkl              Fitted tokenizer used to convert
words to token IDs
The notebook and Streamlit app must use matching artifact filenames and
formats. The app loads the feature extractor using Keras load_model(),
so feature_extractor.keras must be a saved Keras model, not a pickle
file.
9. Streamlit Web Application
The project includes a Streamlit interface that lets users upload a JPG,
JPEG, or PNG image and generate a caption.
9.1 How the Application Works
1. The app loads the trained caption model, feature extractor, and
   tokenizer. st.cache_resource caches the loaded resources.
2. The user uploads an image through st.file_uploader.
3. The uploaded image is displayed in the application.
4. When the user clicks Generate Caption, the uploaded image is
   written to uploaded_image.jpg.
5. The image is loaded and resized to 224 × 224.
6. Pixel values are converted to a NumPy array and divided by 255.0.
7. The feature extractor generates an image feature vector.
8. The tokenizer converts the current caption text into token IDs.
9. The sequence is padded to a maximum length of 34.
10. The caption model predicts the next word using argmax.
11. Word generation continues until endseq is predicted, no word is
    found, or the loop reaches 34 iterations.
12. The generated caption is displayed in Streamlit.
9.2 Complete Streamlit Application Code
Create a file named app.py in the project root and add this code:
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
9.3 Run the Streamlit Application
Open a terminal in the project root directory and run:
streamlit run app.py
Streamlit will display a local URL in the terminal, usually:
http://localhost:8501
Open that address in your browser, upload an image, and click Generate
Caption.
10. Important Notes and Troubleshooting
- Dataset paths: The notebook contains machine-specific Windows
  paths. Replace them before running the notebook on another computer.
- Model paths: The Streamlit code expects model.keras,
  feature_extractor.keras, and tokenizer.pkl inside the models/
  directory.
- Artifact format: load_model() expects a compatible saved Keras
  model. A pickle file containing a feature extractor cannot be loaded
  with load_model().
- Image preprocessing consistency: The app divides pixel values by
  255.0. Ensure this matches the preprocessing used during training.
  If DenseNet201's preprocess_input was used during training,
  inference should use the same preprocessing.
- Maximum caption length: The application uses maxlen=34 and a
  34-iteration generation loop. Keep this consistent with the maximum
  caption length used during training.
- Train/validation split: Review the split cell in the notebook.
  In the current notebook, nimages is referenced without being
  defined, and images is overwritten with an integer before it is
  sliced. The split logic therefore needs correction before that cell
  can run successfully.
- Temporary upload file: The app writes each upload to
  uploaded_image.jpg. For concurrent users or repeated uploads,
  consider using a unique temporary file or processing uploaded bytes
  directly.
- Model compatibility: Use compatible TensorFlow/Keras versions
  for training and inference.
11. Limitations
Generated captions depend on the dataset, vocabulary, training quality,
and visual features learned by the model. Captions may be incomplete or
inaccurate, particularly for images or objects that are poorly
represented in the training data.
12. Future Improvements
- Correct and validate the train/validation split.
- Add evaluation using captioning metrics such as BLEU or METEOR.
- Experiment with beam search instead of selecting only the
  highest-probability next word.
- Compare DenseNet201 with other pretrained image encoders.
- Improve upload handling for multiple concurrent users.
- Deploy the Streamlit application to a cloud hosting platform.
13. License
Add the license and dataset attribution applicable to your project
before public distribution.