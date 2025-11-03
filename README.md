# Dynamic Live Sign Language Translator
This repository contains a real-time, two-handed, dynamic sign language recognition system. The application leverages computer vision and deep learning to capture, learn, and translate sign language gestures performed in front of a webcam into spoken words.

The entire pipeline, from data collection to live translation, is managed through a user-friendly graphical interface built with Tkinter.

## Features

*   **Live Translation**: Translates dynamic sign language gestures from a live webcam feed into text and speech.
*   **Two-Hand Tracking**: Utilizes MediaPipe to detect and track landmarks for both hands simultaneously.
*   **Data Collection**: An intuitive interface to record and label custom sign language gesture sequences.
*   **Model Training**: A built-in function to train an LSTM (Long Short-Term Memory) model on your custom-recorded gestures.
*   **Data Management**: Tools to list, review, play back, and delete recorded gesture data.
*   **Text-to-Speech (TTS)**: Automatically speaks the predicted labels for a complete translation experience.
*   **User-Friendly GUI**: A simple Tkinter-based GUI to access all functionalities, complete with a guide and keyboard shortcut reference.

## How It Works

The system operates through a multi-stage pipeline:

1.  **Hand Landmark Detection**: The application uses Google's MediaPipe Hands solution to detect 21 3D landmarks for each hand in the video frame.
2.  **Feature Extraction**: The 3D coordinates of the detected landmarks for both hands are normalized relative to the wrist's position. This creates a consistent feature vector (126 features total: 2 hands x 21 landmarks x 3 coordinates) for each frame, making the system robust to changes in hand position and scale.
3.  **Sequence Creation**: To understand dynamic gestures, the application collects landmark data over a fixed number of frames (`SEQUENCE_LENGTH = 30`). This sequence of feature vectors represents a single, complete gesture.
4.  **Model Training**: The collected sequences are used to train a TensorFlow/Keras-based LSTM neural network. LSTMs are ideal for learning patterns in sequential data, making them perfect for recognizing gestures that unfold over time. The trained model is saved as `.h5` and `.tflite` files.
5.  **Live Inference**: For live translation, the application continuously captures sequences from the webcam, feeds them into the trained TensorFlow Lite model, and predicts the corresponding label. A history buffer is used to stabilize predictions before announcing the result via TTS.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/kimmartelolives/dynamic-live-sign-language.git
    cd dynamic-live-sign-language
    ```
    
2.  **Create a Virtual Environment:**
    ```sh
    py -3.12 -m venv venv
    ```
    
3.  **Activate Virtual Environment:**
    ```sh
    venv\Scripts\Activate.ps1
    ```

4.  **Install the required Python libraries:**
    ```sh
    pip install opencv-python numpy mediapipe==0.10.21 pygame joblib tensorflow==2.18.0 scikit-learn gTTS
    ```
    *Note: The application uses Tkinter, which is typically included with standard Python installations.*

## Workflow

Follow these steps to collect data, train the model, and run the translator.

### Step 1: Launch the Application

Run the main script from your terminal to open the GUI:
```sh
python live-sign-language.py
```

### Step 2: Collect Gesture Data

1.  Click **Record Sequence**.
2.  When prompted, enter a name for the sign you want to record (e.g., "hello", "thank you").
3.  A camera window will open. Position yourself so both hands are visible.
4.  Press the **'c'** key to start recording a gesture. The system will capture a sequence of 30 frames.
5.  Repeat this process to record multiple samples for each label. It's recommended to have at least 10-15 samples per label for better accuracy.
6.  You can use **List Labels** to see what you've recorded or **Playback Labels & Sequences** to review your recordings.

### Step 3: Train the Model

1.  Once you have collected sufficient data for all your signs, click **Train & Convert Model**.
2.  A progress window will appear, showing the training epochs. This process may take several minutes depending on your hardware and the amount of data.
3.  After completion, the model will be saved as `sign_model_2hands.h5`, `sign_model_2hands.tflite`, and `label_encoder_2hands.joblib` in the project directory.

### Step 4: Run the Live Translator

1.  Click **Run Live Translator**.
2.  A new camera window will appear.
3.  Perform one of the signs you trained in front of the camera.
4.  The application will display the predicted sign and speak the word out loud.

## Keyboard Shortcuts

Ensure the relevant window (Recording, Playback, or Translator) is active by clicking on it before using shortcuts.

### During Recording

*   `c` : Start recording a new gesture sequence.
*   `q` : Quit the recording session.

### During Playback

*   `SPACE` : Pause or resume playback of the current sequence.
*   `n` : Skip to the next sequence.
*   `b` : Go back to the previous sequence.
*   `q` : Quit the playback session.

### Live Translator

*   `q` : Quit the translator.

### General

*   `F1` : Open the Guide window.
*   `F2` : Open the Keyboard Shortcuts window.

## File Structure

The application will create and use the following files and directories in the root folder:

*   `live-sign-language.py`: The main application script.
*   `dynamic_data/`: Directory where recorded gesture sequences (`.npy` files) are stored, organized into subfolders by label.
*   `tts_audio/`: Directory where generated audio files from Google Text-to-Speech are cached.
*   `sign_model_2hands.h5`: The saved Keras model after training.
*   `sign_model_2hands.tflite`: The converted TensorFlow Lite model used for real-time inference.
*   `label_encoder_2hands.joblib`: The saved file that maps text labels to numerical categories for the model.

## Dependencies

*   `opencv-python`
*   `numpy`
*   `mediapipe`
*   `pygame`
*   `joblib`
*   `tensorflow`
*   `scikit-learn`
*   `gTTS`
*   `tkinter`
