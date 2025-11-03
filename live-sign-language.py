import os
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import pygame
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import deque
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox, scrolledtext

def center_window(win):
    """Centers a Tkinter window on the screen."""
    win.update_idletasks() 
    width = win.winfo_width()
    height = win.winfo_height()
    if width == 1 and height == 1:  
        win.update()
        width = win.winfo_width()
        height = win.winfo_height()
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")


# ----------------------------
# Settings
# ----------------------------
DATA_DIR = "dynamic_data"
SEQUENCE_LENGTH = 30
FEATURES = 126  # 2 hands x 21 landmarks x 3 coords
MODEL_H5 = "sign_model_2hands.h5"
MODEL_TFLITE = "sign_model_2hands.tflite"
LABEL_ENCODER_FILE = "label_encoder_2hands.joblib"

# ----------------------------
# Mediapipe setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# TTS
# ----------------------------
pygame.mixer.init()
def speak_text(text):
    def thread_func(txt):
        try:
            os.makedirs("tts_audio", exist_ok=True)
            filename = os.path.join("tts_audio", f"{txt}.mp3")
            if os.path.exists(filename):
                os.remove(filename)
            from gtts import gTTS
            tts = gTTS(text=txt, lang='en')
            tts.save(filename)
            pygame.mixer.quit()
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"TTS Error: {e}")
    threading.Thread(target=thread_func, args=(text,), daemon=True).start()

# ----------------------------
# Data functions
# ----------------------------
def extract_landmarks(results):
    features = np.zeros(FEATURES, dtype=np.float32)
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks[:2]):
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            base_x, base_y, base_z = coords[:3]
            norm = [(coords[i]-base_x, coords[i+1]-base_y, coords[i+2]-base_z)
                    for i in range(0, len(coords), 3)]
            flat = np.array([val for triplet in norm for val in triplet], dtype=np.float32)
            if len(flat) < 63:
                flat = np.concatenate([flat, np.zeros(63 - len(flat), dtype=np.float32)])
            features[idx*63:(idx+1)*63] = flat
    return features

def get_labels():
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]

import tkinter as tk
from tkinter import messagebox, scrolledtext

def list_labels():
    labels = get_labels()
    if not labels:
        messagebox.showinfo("Labels", "No labels found yet.")
        return

 
    info = "Existing labels and sequence counts".center(50) + "\n" + f"Total Labels: {len(labels)}".center(50) + "\n\n"

    for i, l in enumerate(labels, start=1):
        path = os.path.join(DATA_DIR, l)
        count = len([f for f in os.listdir(path) if f.endswith(".npy")])
        info += f"{i}. {l}: {count} sequence{'s' if count != 1 else ''}\n"

 
    scroll_win = tk.Toplevel()
    scroll_win.title("Labels List")
    window_width = 400
    window_height = 300
    scroll_win.geometry(f"{window_width}x{window_height}")
    scroll_win.resizable(False, False)

  
    scroll_win.update_idletasks()
    screen_width = scroll_win.winfo_screenwidth()
    screen_height = scroll_win.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    scroll_win.geometry(f"{window_width}x{window_height}+{x}+{y}")

 
    text_box = scrolledtext.ScrolledText(scroll_win, wrap=tk.WORD, width=50, height=15)
    text_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    text_box.insert(tk.END, info)
    text_box.config(state=tk.DISABLED)  # Read-only

    
    close_btn = tk.Button(scroll_win, text="Close", command=scroll_win.destroy)
    close_btn.pack(pady=5)

# ----------------------------
# Recording sequences
# ----------------------------
def record_sequence(label):
    if not label:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    label_path = os.path.join(DATA_DIR, label)
    os.makedirs(label_path, exist_ok=True)
    sample_count = len([f for f in os.listdir(label_path) if f.endswith(".npy")])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    # messagebox.showinfo("Instructions", "Press 'C' to record a sequence, 'Q' to quit")

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands_detector_local:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector_local.process(rgb)

        
            if results.multi_hand_landmarks:
                for h in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

            height, width, _ = frame.shape

          
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Seq #{sample_count + 1}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, "Keys: [C]=Record | [Q]=Quit",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Dynamic Sign Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
               
                for i in range(3, 0, -1):
                    tmp = frame.copy()
                    cv2.putText(tmp, f"{i}", (250, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
                    cv2.imshow("Dynamic Sign Collector", tmp)
                    cv2.waitKey(1000)

                seq = []
                for i in range(SEQUENCE_LENGTH):
                    ret, f_seq = cap.read()
                    if not ret:
                        break
                    f_seq = cv2.flip(f_seq, 1)
                    rgb_seq = cv2.cvtColor(f_seq, cv2.COLOR_BGR2RGB)
                    r_seq = hands_detector_local.process(rgb_seq)
                    feat = extract_landmarks(r_seq)
                    seq.append(feat)

                  
                    if r_seq.multi_hand_landmarks:
                        for h in r_seq.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(f_seq, h, mp_hands.HAND_CONNECTIONS)

                    height, width, _ = f_seq.shape
                    
                    cv2.putText(f_seq, f"Label: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(f_seq, f"Recording: {i+1}/{SEQUENCE_LENGTH}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                    cv2.putText(f_seq, "Keys: [C]=Record | [Q]=Quit",
                                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                    cv2.imshow("Dynamic Sign Collector", f_seq)
                    cv2.waitKey(30)

                
                seq = np.array(seq)
                np.save(os.path.join(label_path, f"{label}_{sample_count}.npy"), seq)
                sample_count += 1
                messagebox.showinfo("Saved", f"Saved sequence #{sample_count} for label '{label}'")

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# Playback sequences
# ----------------------------
def play_all_sequences(label):
    if not label:
        return
    labels = get_labels()
    if label not in labels:
        messagebox.showinfo("Playback", f"No label exists with name '{label}'")
        return

    sel_path = os.path.join(DATA_DIR, label)
    sequences = sorted([f for f in os.listdir(sel_path) if f.endswith(".npy")])
    if not sequences:
        messagebox.showinfo("Playback", f"No sequences for label '{label}'")
        return

    width, height = 640, 480
    paused = False
    seq_index = 0

    while 0 <= seq_index < len(sequences):
        data = np.load(os.path.join(sel_path, sequences[seq_index]))
        frame_index = 0

        while True:
            frame_data = data[frame_index]
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            points = []

         
            for i in range(0, len(frame_data), 3):
                x, y = frame_data[i], frame_data[i + 1]
                x = int(np.clip((x + 0.5) * width, 0, width - 1))
                y = int(np.clip((y + 0.5) * height, 0, height - 1))
                points.append((x, y))

            for connection in mp_hands.HAND_CONNECTIONS:
                s, e = connection
                if s < len(points) and e < len(points):
                    cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

           
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Seq {seq_index + 1}/{len(sequences)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, "Keys: [SPACE]=Pause | [N]=Next | [B]=Back | [Q]=Quit",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Playback", frame)
            key = cv2.waitKey(50) & 0xFF

          
            if key == ord('q'):
                cv2.destroyWindow("Playback")
                return

            
            elif key == ord(' '):
                paused = not paused
                if paused:
                    cv2.putText(frame, "PAUSED", (width // 2 - 100, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow("Playback", frame)
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == ord(' '):
                            paused = False
                            break
                        elif k2 == ord('q'):
                            cv2.destroyWindow("Playback")
                            return

          
            elif key == ord('n'):
                seq_index += 1
                break

           
            elif key == ord('b'):
                seq_index -= 1
                if seq_index < 0:
                    seq_index = 0
                break

            frame_index += 1
            if frame_index >= len(data):
                
                seq_index += 1
                break

    cv2.destroyWindow("Playback")



# ----------------------------
# Train & Convert
# ----------------------------
# def train_model():
#     X, y = [], []
#     labels = get_labels()
#     for label in labels:
#         path = os.path.join(DATA_DIR,label)
#         for f in os.listdir(path):
#             if f.endswith(".npy"):
#                 data = np.load(os.path.join(path,f))
#                 if data.shape==(SEQUENCE_LENGTH,FEATURES):
#                     X.append(data)
#                     y.append(label)
#     if not X:
#         messagebox.showerror("Error","No data to train.")
#         return
#     X = np.array(X)
#     le = LabelEncoder()
#     y_enc = le.fit_transform(y)
#     y_cat = to_categorical(y_enc)
#     try:
#         X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_enc)
#     except:
#         X_train, X_test, y_train, y_test = X, X, y_cat, y_cat
#     model = Sequential([
#         LSTM(128,return_sequences=True,input_shape=(SEQUENCE_LENGTH,FEATURES)),
#         LSTM(64),
#         Dropout(0.3),
#         Dense(64,activation='relu'),
#         Dense(len(le.classes_),activation='softmax')
#     ])
#     model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#     model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=8)
#     model.save(MODEL_H5)
#     joblib.dump(le,LABEL_ENCODER_FILE)

#    
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.experimental_enable_resource_variables = True
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#     converter._experimental_lower_tensor_list_ops=False
#     tflite_model = converter.convert()
#     with open(MODEL_TFLITE,"wb") as f:
#         f.write(tflite_model)
#     messagebox.showinfo("Training","Model trained and saved.")

def train_model_with_loading():
    loading_win = tk.Toplevel()
    loading_win.title("Training Model")
    center_window(loading_win)

    

    tk.Label(loading_win, text="Training in progress...\nPlease wait.", padx=20, pady=10).pack()

    progress = ttk.Progressbar(loading_win, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=10)

    status_label = tk.Label(loading_win, text="Starting...", padx=10)
    status_label.pack(pady=5)

    cancel_flag = {"stop": False}

    def cancel_training():
        cancel_flag["stop"] = True
        status_label.config(text="Cancelling... Please wait.")

    tk.Button(loading_win, text="Cancel", command=cancel_training, width=15).pack(pady=10)

    loading_win.update()

    def train_thread():
        X, y = [], []
        labels = get_labels()
        for label in labels:
            path = os.path.join(DATA_DIR, label)
            for f in os.listdir(path):
                if f.endswith(".npy"):
                    data = np.load(os.path.join(path, f))
                    if data.shape == (SEQUENCE_LENGTH, FEATURES):
                        X.append(data)
                        y.append(label)
        if not X:
            messagebox.showerror("Error", "No data to train.")
            loading_win.destroy()
            return
        X = np.array(X)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        y_cat = to_categorical(y_enc)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_enc)
        except:
            X_train, X_test, y_train, y_test = X, X, y_cat, y_cat

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(len(le.classes_), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                progress['maximum'] = self.params['epochs']
            def on_epoch_end(self, epoch, logs=None):
                progress['value'] = epoch + 1
                status_label.config(text=f"Epoch {epoch+1}/{self.params['epochs']} - Loss: {logs['loss']:.4f}")
                loading_win.update()
                if cancel_flag["stop"]:
                    self.model.stop_training = True

        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=50,
                  batch_size=8,
                  callbacks=[ProgressCallback()])

        if cancel_flag["stop"]:
            messagebox.showinfo("Training", "Training cancelled.")
        else:
           
            model.save(MODEL_H5)
            joblib.dump(le, LABEL_ENCODER_FILE)

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.experimental_enable_resource_variables = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            with open(MODEL_TFLITE, "wb") as f:
                f.write(tflite_model)

            messagebox.showinfo("Training", "Model trained and saved.")

        loading_win.destroy()

    threading.Thread(target=train_thread, daemon=True).start()



# ----------------------------
# Delete Label
# ----------------------------
import tkinter as tk
from tkinter import messagebox
import os, shutil

def delete_labels():
    labels = get_labels()
    if not labels:
        messagebox.showinfo("Delete Labels", "No labels found.")
        return

    import shutil

    delete_window = tk.Toplevel()
    delete_window.title("Delete Labels")

  
    window_width, window_height = 420, 400
    delete_window.geometry(f"{window_width}x{window_height}")
    delete_window.resizable(False, False)
    delete_window.update_idletasks()
    screen_width = delete_window.winfo_screenwidth()
    screen_height = delete_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    delete_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    tk.Label(delete_window, text="Select labels to delete:", font=("Arial", 10, "bold")).pack(pady=5)
  

    frame_container = tk.Frame(delete_window)
    frame_container.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(frame_container)
    scrollbar = ttk.Scrollbar(frame_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(scrollable_window, width=canvas.winfo_width())

    scrollable_frame.bind("<Configure>", on_configure)


    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

  
    vars_dict = {}
    for i, label in enumerate(labels, start=1):
        count = len([f for f in os.listdir(os.path.join(DATA_DIR, label)) if f.endswith(".npy")])
        var = tk.BooleanVar()
        cb_text = f"{i}. {label} ({count} sequence{'s' if count != 1 else ''})"
        cb = tk.Checkbutton(scrollable_frame, text=cb_text, variable=var, anchor="w", justify="left")
        cb.pack(fill="x", padx=5, pady=2, anchor="w")
        vars_dict[label] = var

  
    def confirm_delete():
        selected_labels = [label for label, var in vars_dict.items() if var.get()]
        if not selected_labels:
            messagebox.showinfo("Delete Labels", "No labels selected.")
            return
        if messagebox.askyesno("Confirm Delete",
                               f"Are you sure you want to delete these labels?\n{', '.join(selected_labels)}"):
            for label in selected_labels:
                shutil.rmtree(os.path.join(DATA_DIR, label))
            messagebox.showinfo("Deleted", f"Deleted labels: {', '.join(selected_labels)}")
            delete_window.destroy()

    tk.Button(delete_window, text="Delete Selected", command=confirm_delete, width=20).pack(pady=10)




# ----------------------------
# Key Binds
# ----------------------------

def show_key_binds(parent):
    """Displays the list of keyboard shortcuts in a centered window."""
    keys_win = tk.Toplevel(parent)
    keys_win.title("Keyboard Shortcuts")

    guide_text = tk.Text(keys_win, wrap="word", width=60, height=22)
    guide_text.insert("1.0",
    """⌨️ KEY BINDS / SHORTCUTS

=============================
🎬 DURING RECORDING
=============================
▶️  c  → Start or continue recording a gesture sequence  
⏹️  q  → Stop recording and save the current sequence  

=============================
🎞️ DURING PLAYBACK (Play Label & Sequence)
=============================
▶️  SPACE  → Pause/Play current sequence  
⏩  n     → Skip to next sequence  
⏪  b      → Go back to previous sequence  
⏹️  q      → Stop playback  

=============================
🤖 LIVE TRANSLATOR
=============================
🎥  q → Quit translator window  

=============================
💡 GENERAL
=============================
📘  F1  → Open guide/help window
📖  F2  → Open key binds/shortcuts window
- All key binds are case-insensitive.
- Make sure the window is active (clicked) before pressing shortcuts.
- Press q anytime to safely close a recording or playback loop.
""")
    guide_text.config(state="disabled")
    guide_text.pack(padx=10, pady=10)

    tk.Button(keys_win, text="Close", command=keys_win.destroy).pack(pady=(0,10))

    center_window(keys_win)



# ----------------------------
# Guide
# ----------------------------

def show_guide(parent):
    """Displays the guide/help instructions in a centered window."""
    guide_win = tk.Toplevel(parent)
    guide_win.title("Guide")

   
    guide_text = tk.Text(guide_win, wrap="word", width=60, height=20)
    guide_text.insert("1.0", 
    """📘 GUIDE: 2-Hand Sign Language Pipeline

1️⃣ List Labels  
   → View all existing labels.

2️⃣ Record Sequence  
   → Record hand sign gestures for a specific label.

3️⃣ Play Sequences  
   → Replay recorded gesture sequences for a label.

4️⃣ Play All Labels Automatically  
   → Automatically play all gesture recordings for every label.

5️⃣ Delete Label(s)  
   → Remove one or more labels and their recordings.

6️⃣ Train & Convert Model  
   → Train the gesture recognition model using current data and export it.

7️⃣ Run Live Translator  
   → Run the live translator using webcam input.

💡 Tips:
- Make sure to record at least a few samples per label before training.
- You can retrain anytime after adding or removing labels.
- The “Train & Convert” step updates your model used in the Live Translator.
    """)
    guide_text.config(state="disabled")  
    guide_text.pack(padx=10, pady=10)

    tk.Button(guide_win, text="Close", command=guide_win.destroy).pack(pady=(0,10))

   
    center_window(guide_win)




# ----------------------------
# Play all sequences automatically for all labels
# ----------------------------
def play_all_labels_sequences():
    labels = get_labels()
    if not labels:
        messagebox.showinfo("Playback", "No labels found.")
        return

    width, height = 640, 480
    paused = False
    label_index = 0
    seq_index = 0

    while 0 <= label_index < len(labels):
        label = labels[label_index]
        sel_path = os.path.join(DATA_DIR, label)
        sequences = sorted([f for f in os.listdir(sel_path) if f.endswith(".npy")])

        if not sequences:
            label_index += 1
            continue

      
        seq_index = max(0, min(seq_index, len(sequences) - 1))

        data = np.load(os.path.join(sel_path, sequences[seq_index]))
        frame_index = 0

        while True:
            frame_data = data[frame_index]
            frame = np.zeros((height, width, 3), dtype=np.uint8)

         
            points = []
            for i in range(0, len(frame_data), 3):
                x, y = frame_data[i], frame_data[i + 1]
                x = int(np.clip((x + 0.5) * width, 0, width - 1))
                y = int(np.clip((y + 0.5) * height, 0, height - 1))
                points.append((x, y))

            for connection in mp_hands.HAND_CONNECTIONS:
                s, e = connection
                if s < len(points) and e < len(points):
                    cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Seq {seq_index+1}/{len(sequences)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, "Keys: [SPACE]=Pause | [N]=Next | [B]=Back | [Q]=Quit",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Playback All Labels & Sequences", frame)
            key = cv2.waitKey(50) & 0xFF

            
            if key == ord('q'):
                cv2.destroyWindow("Playback All Labels & Sequences")
                return

            
            elif key == ord(' '):
                paused = not paused
                if paused:
                    cv2.putText(frame, "PAUSED", (width // 2 - 100, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow("Playback All Labels", frame)
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == ord(' '):  
                            paused = False
                            break
                        elif k2 == ord('q'):
                            cv2.destroyWindow("Playback All Labels")
                            return

        
            elif key == ord('n'):
                seq_index += 1
                if seq_index >= len(sequences):
                    seq_index = 0
                    label_index += 1
                break  

           
            elif key == ord('b'):
                seq_index -= 1
                if seq_index < 0:
                    label_index -= 1
                    if label_index < 0:
                        
                        label_index = 0
                        seq_index = 0
                    else:
                        prev_label = labels[label_index]
                        prev_path = os.path.join(DATA_DIR, prev_label)
                        prev_sequences = sorted([f for f in os.listdir(prev_path) if f.endswith(".npy")])
                        if prev_sequences:
                            seq_index = len(prev_sequences) - 1
                        else:
                            seq_index = 0
                break  

            frame_index += 1
            if frame_index >= len(data):  
                
                seq_index += 1
                if seq_index >= len(sequences):
                    seq_index = 0
                    label_index += 1
                break  

        if label_index >= len(labels):
            break

        cv2.destroyWindow("Playback All Labels")

# ----------------------------
# Live Translator
# ----------------------------
def run_translator():
    if not os.path.exists(MODEL_TFLITE) or not os.path.exists(LABEL_ENCODER_FILE):
        messagebox.showerror("Error", "Train model first.")
        return

    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    le = joblib.load(LABEL_ENCODER_FILE)

    buffer = deque(maxlen=SEQUENCE_LENGTH)
    history = deque(maxlen=5)
    last_spoken = None
    last_time = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands_detector_local:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector_local.process(rgb)
            height, width, _ = frame.shape

            if results.multi_hand_landmarks:
                for h in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

                feat = extract_landmarks(results)
                buffer.append(feat)

                if len(buffer) == SEQUENCE_LENGTH and np.count_nonzero(buffer[-1]) > 0:
                    seq_input = np.expand_dims(np.array(buffer, dtype=np.float32), 0)
                    interpreter.set_tensor(input_details[0]['index'], seq_input)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])[0]
                    pred_idx = np.argmax(pred)
                    pred_label = le.inverse_transform([pred_idx])[0]

                    history.append(pred_label)
                    stable_pred = max(set(history), key=history.count)

            
                    cv2.putText(frame, f"Prediction: {stable_pred}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    now = time.time()
                    if stable_pred != last_spoken or now - last_time > 3:
                        speak_text(stable_pred)
                        last_spoken = stable_pred
                        last_time = now
            else:
                cv2.putText(frame, "No hands detected", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                buffer.append(np.zeros(FEATURES, dtype=np.float32))

            
            cv2.putText(frame, "Keys: [Q]=Quit",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1)

            cv2.imshow("Dynamic Sign Translator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

    

# ----------------------------
# GUI
# ----------------------------
def main_gui():
    root = tk.Tk()
    root.title("2-Hand Sign Language Pipeline")
    logo_img = tk.PhotoImage(file="sign-language.png") 
    root.iconphoto(True, logo_img)  
    root.bind("<F1>", lambda event: show_guide(root))
    root.bind("<F2>", lambda event: show_key_binds(root))

 
    menubar = tk.Menu(root)

   
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Guide", command=lambda: show_guide(root))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)
    menubar.add_cascade(label="File", menu=file_menu)

   
    keys_menu = tk.Menu(menubar, tearoff=0)
    keys_menu.add_command(label="View Key Binds", command=lambda: show_key_binds(root))
    menubar.add_cascade(label="Keys", menu=keys_menu)

    
    root.config(menu=menubar)
    
    tk.Button(root,text="List Labels",command=list_labels,width=30).pack(pady=5)
    tk.Button(root,text="Record Sequence",command=lambda:record_sequence(simpledialog.askstring("Label","Enter label name")),width=30).pack(pady=5)
    tk.Button(root,text="Playback Labels & Sequences",command=lambda:play_all_sequences(simpledialog.askstring("Label","Enter label name")),width=30).pack(pady=5)
    tk.Button(root,text="Playback ALL Labels & Sequences",command=play_all_labels_sequences,width=30).pack(pady=5)
    tk.Button(root, text="Delete Label(s)", command=delete_labels, width=30).pack(pady=5)
    # tk.Button(root,text="Train & Convert Model",command=train_model,width=30).pack(pady=5)
    tk.Button(root, text="Train & Convert Model", command=train_model_with_loading, width=30).pack(pady=5)
    tk.Button(root,text="Run Live Translator",command=run_translator,width=30).pack(pady=5)
    tk.Button(root,text="Quit",command=root.destroy,width=30).pack(pady=5)

    center_window(root)
    root.mainloop()

if __name__=="__main__":
    main_gui()
