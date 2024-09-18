# Deep Audio Classifier

## Project Overview

The **Deep Audio Classifier** project leverages deep learning techniques using **TensorFlow** to classify audio clips. The primary goal is to detect specific bird species (Capuchinbird) calls from audio recordings collected from a forest environment. This project preprocesses audio data into spectrograms and uses a convolutional neural network (CNN) to perform binary classification on the audio clips.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Performance Evaluation](#performance-evaluation)
7. [Prediction Pipeline](#prediction-pipeline)
8. [Exporting Results](#exporting-results)
9. [Instructions to Run](#instructions-to-run)
10. [Future Improvements](#future-improvements)

## Project Structure

```
.
├── data/                                # Folder containing audio clips
│   ├── Parsed_Capuchinbird_Clips         # Positive class audio clips (Capuchinbird)
│   ├── Parsed_Not_Capuchinbird_Clips     # Negative class audio clips (Other sounds)
│   └── Forest Recordings                 # Long-form audio recordings for prediction
├── results.csv                           # Output predictions with number of bird calls
├── deepaudioclassifier.py                # Main script containing the entire code
└── README.md                             # Project documentation
```

## Requirements

Install the required dependencies using the following:

```bash
pip install tensorflow==2.13.0 tensorflow-gpu==2.13.0 tensorflow-io==0.31.0 matplotlib==3.7.2
```

- **TensorFlow**: Framework for deep learning and neural networks.
- **TensorFlow I/O**: For loading audio data and converting formats.
- **Matplotlib**: For visualizing audio waveforms and spectrograms.

## Data Preprocessing

1. **Audio Loading**: The audio files are loaded using TensorFlow's `tf.io.read_file` and `tf.audio.decode_wav` functions. The audio is downsampled from 44.1kHz to 16kHz for efficient processing.
2. **Waveform Conversion**: Convert stereo to mono and ensure all audio clips are standardized to the same length (48,000 samples).
3. **Spectrogram Conversion**: The raw audio waveforms are transformed into spectrograms using Short-Time Fourier Transform (STFT), which is essential for the model to learn temporal patterns in audio.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow's **Keras API**. The key layers include:

- **Conv2D Layers**: Two convolutional layers with 16 filters and (3x3) kernels to extract features from spectrograms.
- **Flatten Layer**: Flattens the 2D output to 1D for fully connected layers.
- **Dense Layers**: Two fully connected layers, one with 128 units (ReLU activation) and one output layer with a single unit (sigmoid activation) for binary classification.

### Model Summary

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 1491, 257, 16)     160       
conv2d_1 (Conv2D)            (None, 1489, 255, 16)     2320      
flatten (Flatten)            (None, 6066240)           0         
dense (Dense)                (None, 128)               77647488  
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 77,649,137
Trainable params: 77,649,137
Non-trainable params: 0
_________________________________________________________________
```

## Training Process

1. **Data Pipeline**: TensorFlow Datasets are created by labeling the audio clips (`1` for Capuchinbird calls, `0` for other sounds) and batching them for training.
2. **Splitting**: The dataset is split into training (36 batches) and testing (15 batches) sets.
3. **Training**: The model is trained using the **Adam optimizer** and **binary cross-entropy loss** for 4 epochs. The key metrics tracked are **precision** and **recall**.

```python
model.compile(optimizer='Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.fit(train, epochs=4, validation_data=test)
```

## Performance Evaluation

- **Loss, Precision, Recall**: Plots are generated after training to visualize the model's performance on both training and validation sets. Precision and recall are critical for minimizing false negatives in detecting bird calls.

```python
plt.plot(hist.history['precision'])
plt.plot(hist.history['recall'])
```

## Prediction Pipeline

1. **Preprocessing MP3 Files**: Forest recordings in MP3 format are loaded and converted to WAV, then sliced into smaller windows (16k samples each).
2. **Spectrogram Conversion**: Each window is transformed into a spectrogram and passed through the trained model.
3. **Prediction**: The model outputs logits, which are converted into binary classes (1 for Capuchinbird call, 0 for no call).
4. **Grouping Detections**: Consecutive predictions are grouped to avoid redundant detections.

```python
yhat = [1 if prediction > 0.5 else 0 for prediction in model.predict(audio_slices)]
calls = tf.math.reduce_sum(yhat).numpy()
```

## Exporting Results

The model's predictions are exported to a CSV file that contains the number of Capuchinbird calls detected in each recording.

```python
with open('results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
```

## Instructions to Run

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/your-repo/deep-audio-classifier.git
   cd deep-audio-classifier
   ```

2. **Install Dependencies**:
   
   ```bash
   pip install tensorflow==2.13.0 tensorflow-gpu==2.13.0 tensorflow-io==0.31.0 matplotlib==3.7.2
   ```

3. **Run the Script**:
   The main script, `deepaudioclassifier.py`, contains all the steps from data preprocessing to model training and prediction. Ensure the data directory is set up correctly.
   
   ```bash
   python deepaudioclassifier.py
   ```

4. **Results**: After running the script, predictions will be saved in `results.csv`.

## Future Improvements

- **Model Tuning**: Experiment with different CNN architectures or hyperparameter tuning (e.g., kernel size, number of filters).
- **Longer Training**: Increase the number of epochs and explore early stopping or learning rate scheduling.
- **Additional Data**: Incorporate more diverse sound recordings to generalize the model for other bird species or sound categories.
- **Real-time Detection**: Implement real-time audio detection using streaming data.
