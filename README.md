Image Caption Generator
A Streamlit-based application that generates captions for uploaded images using Deep Learning.
Features
- Upload images in JPG, JPEG, or PNG format.
- Extract image features using a trained feature extractor.
- Generate captions using an LSTM-based model.
- Display the uploaded image and generated caption.
Technologies Used
- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pickle
Project Structure
project/
│
├── app.py
└── models/
    ├── model.keras
    ├── feature_extractor.keras
    └── tokenizer.pkl


Installation
pip install streamlit tensorflow numpy pillow


Run Application
streamlit run app.py


Open http://localhost:8501 in your browser.
How It Works
1. Loads the trained model, feature extractor, and tokenizer.
2. Accepts an image uploaded by the user.
3. Resizes the image to 224 × 224 pixels.
4. Extracts image features.
5. Generates captions word by word.
6. Displays the generated caption.
Model Files
- model.keras: Caption generation model.
- feature_extractor.keras: Image feature extraction model.
- tokenizer.pkl: Text tokenizer used during training.
