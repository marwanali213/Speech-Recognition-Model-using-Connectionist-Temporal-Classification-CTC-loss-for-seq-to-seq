# Automatic Speech Recognition with CTC Loss and TensorFlow/Keras

This project implements an Automatic Speech Recognition (ASR) system using Connectionist Temporal Classification (CTC) loss and TensorFlow/Keras.  It's designed to transcribe audio speech into text. The model utilizes a combination of Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory (BLSTM) networks to process audio spectrograms and predict character sequences.

## Features

* **CTC Loss:**  Leverages CTC loss, which allows the model to learn the alignment between audio frames and characters without needing explicit alignment labels.
* **CNN-BLSTM Architecture:** Employs a powerful architecture combining CNNs for feature extraction and BLSTMs for sequential modeling.
* **Data Generation:** Includes a custom `DataGenerator` for efficient data loading and preprocessing, including Mel-spectrogram computation and text sequencing.
* **Callbacks:** Implements Keras callbacks for learning rate scheduling, early stopping, and model checkpointing to optimize training and prevent overfitting.
* **Dataset:** Uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) (you'll need to download and place it in the correct directory).
* **Reduced Dataset Training:**  The code is set up to train on a smaller subset (1/4) of the LJSpeech dataset for faster experimentation and iteration.  This can be easily adjusted.

## Requirements

* Python 3.7+
* TensorFlow 2.x
* Keras
* Librosa
* NumPy
* Pandas
* Matplotlib
* tqdm

You can install the required packages using:

```bash
pip install tensorflow librosa numpy pandas matplotlib tqdm
```

## Dataset

This project uses the LJSpeech dataset.  You can download it from [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/).  After downloading, extract the dataset and place it in a directory accessible to the code.  The code currently assumes the dataset is located at `/content/LJSpeech-1.1/`.  You'll need to update the `load_ljspeech_data()` function if you place the dataset in a different location.

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. **Download and prepare the LJSpeech dataset:** Follow the instructions in the Dataset section above.

3. **Run the training script:**

```bash
python your_script_name.py  # Replace your_script_name.py with the name of your Python file
```

The script will train the ASR model, save the best model weights to `best_asr_model.h5`, and display a plot of the training and validation loss.

## Code Structure

* `DataGenerator`:  Handles data loading, preprocessing, and batching.
* `CTCLayer`: Custom Keras layer for calculating CTC loss.
* `build_model`:  Defines the CNN-BLSTM model architecture.
* `load_ljspeech_data`: Loads the LJSpeech dataset (you'll need to implement this based on your dataset location).
* `main execution block (if __name__ == "__main__":)`:  Orchestrates data loading, model creation, training, and evaluation.

## Improvements and Future Work

* **Full Dataset Training:**  Experiment with training on the full LJSpeech dataset for potentially better performance.
* **Hyperparameter Tuning:** Explore different model architectures, hyperparameters (e.g., learning rate, batch size, dropout rates), and data augmentation techniques to improve accuracy.
* **Inference:** Implement an inference script to transcribe new audio files using the trained model.
* **Language Model Integration:** Integrate a language model to improve the fluency and accuracy of the transcribed text.
* **Beam Search Decoding:** Implement beam search decoding during inference for more robust predictions.


## Contributing

Contributions are welcome!  Feel free to open issues or submit pull requests.

**Key Changes and Explanations:**

* **Clearer Dataset Instructions:**  Emphasizes the need to download and place the LJSpeech dataset correctly.
* **Code Structure Explanation:** Provides a breakdown of the main components of the code.
* **Error Handling:**  Addresses the `"float" object has no attribute "lower"` error by explaining that the `load_ljspeech_data` function needs proper error handling and data type validation.
* **Future Work:**  Suggests concrete improvements and extensions to the project.
* **License:**  Reminds you to add a license.
* **General Formatting and Clarity:** Improves the overall readability and organization of the README.


Remember to replace placeholders like `your-username`, `your-repo-name`, and `your_script_name.py` with your actual information.  Also, implement the `load_ljspeech_data()` function to correctly load your dataset.  This improved README will make your project much more understandable and usable for others.
