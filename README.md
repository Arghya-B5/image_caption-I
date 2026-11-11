Image Caption Generator using DenseNet201 and LSTM
Overview
This project generates a natural-language caption for an input image
using a deep-learning image-captioning pipeline. It combines:
- DenseNet201 to extract visual features from images.
- LSTM to generate a caption one word at a time.
- TensorFlow/Keras to build and train the neural-network model.
- Tokenization and sequence padding to prepare text captions for
  training.
The workflow in the notebook includes image and caption loading, text
preprocessing, image-feature extraction, custom batch generation, model
training, saving model artifacts, and generating captions for new
images.
Project Workflow
1. Load the image-caption dataset from a CSV file.
2. Read and resize images to 224 × 224 pixels.
3. Preprocess captions by converting them to lowercase, removing
   one-character words, and adding startseq and endseq tokens.
4. Fit a Keras tokenizer and determine the vocabulary size and maximum
   caption length.
5. Use DenseNet201 to extract a feature vector for each image.
6. Generate training examples with a custom Keras Sequence data
   generator. Each example contains image features and a partial
   caption; the target is the next word.
7. Train an image-captioning model that combines image features, an
   embedding layer, and an LSTM.
8. Save the trained model, tokenizer, and feature extractor.
9. Load those artifacts to generate and display a caption for a new
   image.
Model Architecture
Image feature extractor
The notebook loads the pretrained DenseNet201 architecture and creates a
feature-extraction model using the output of its penultimate layer. The
resulting image feature vector is passed to the caption-generation
network.
Caption-generation network
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
Requirements
Install the main packages used in the notebook:
pip install tensorflow numpy pandas matplotlib seaborn tqdm pillow
The notebook also uses Python's standard-library modules, including
os, warnings, textwrap, and pickle.
Use a TensorFlow-compatible Python version and install a TensorFlow
release compatible with your operating system and hardware.

Dataset
The notebook expects:
- An image directory containing the image files.
- A CSV file named captions.txt.zip that can be read by
  pandas.read_csv.
- The CSV must contain the columns image and caption.
The original notebook uses local Windows paths. Update the paths in the
notebook to match your machine before running it. For example:
image_path = "path/to/Images"
data = pd.read_csv("path/to/captions.txt.zip")
Keep the image filenames in the CSV consistent with the files in the
image directory.
Run the Project
1. Clone or download this repository.
2. Place the image dataset and caption CSV in accessible locations.
3. Open test(2).ipynb in Jupyter Notebook, JupyterLab, or VS Code.
4. Update the dataset paths.
5. Run the notebook cells in order to preprocess the data, extract
   image features, and train the model.
6. The training cell saves the best model as model.keras.
7. The artifact-saving cell creates tokenizer.pkl and
   feature_extractor.pkl.
8. Set the path to a test image in the final prediction cell and run it
   to generate a caption.
Saved Artifacts
  File                                Purpose
  model.keras                       Trained caption-generation model

  tokenizer.pkl                     Fitted text tokenizer used to
convert words to token IDs
feature_extractor.pkl             DenseNet201-based image feature
extractor
Keep these files together when running the prediction code, or update
the paths in the prediction function.
Generating a Caption
The notebook defines generate_and_display_caption(...). It loads the
saved model and preprocessing artifacts, extracts features from the
supplied image, and predicts the caption sequentially until it generates
endseq or reaches the maximum sequence length.

Example:
caption = generate_and_display_caption(
    image_path="path/to/your/image.jpg",
    model_path="model.keras",
    tokenizer_path="tokenizer.pkl",
    feature_extractor_path="feature_extractor.pkl"
)

print(caption)
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
Important Notes
- Dataset paths: The notebook contains machine-specific Windows
  paths. Replace them before running the notebook elsewhere.
- Train/validation split: Review the split cell before training.
  In the current notebook, nimages is referenced without being
  defined, and images is overwritten with an integer before it is
  sliced. The split logic therefore needs correction before that cell
  can run successfully.
- Sequence length: The prediction function defaults to
  max_length=34. Use the same maximum caption length that was
  calculated during training; otherwise, update the prediction
  function accordingly.
- Input dimensions: The caption model is defined with a
  1,920-dimensional image input, matching the DenseNet201 feature
  representation used in the notebook. Keep the extractor and caption
  model consistent.
- Preprocessing consistency: Apply the same image and text
  preprocessing during training and inference.
Limitations
Generated captions depend on the dataset, vocabulary, training quality,
and the visual features learned by the model. A caption may be
incomplete or inaccurate, particularly for images or objects that are
poorly represented in the training data.
Future Improvements
- Correct and validate the train/validation split.
- Add evaluation using captioning metrics such as BLEU or METEOR.
- Experiment with beam search instead of selecting only the
  highest-probability next word.
- Compare DenseNet201 with other pretrained image encoders.
- Build a simple web interface for uploading an image and viewing its
  generated caption.
License
Add the license and dataset attribution applicable to your project
before public distribution.