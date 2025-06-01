import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------ Transformer Model ------------------

class EEGTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=2, num_layers=1, num_classes=7):
        super(EEGTransformer, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ------------------ Emotion Map ------------------

emotions_map = {
    0: ("Happy", "😄"),
    1: ("Surprisingly happy", "😢"),
    2: ("Joy", "😍"),
    3: ("Angry", "😠"),
    4: ("Stressed", "😣"),
    5: ("Relaxed", "😌"),
    6: ("Angrily stressed", "😱")
}

# ------------------ Load EEG Signal ------------------

def load_eeg_signal_from_file(file_path):
    try:
        # Read EEG CSV file with headers
        df = pd.read_csv(file_path, comment="#")
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric EEG data found in the file.")

        # Normalize each row, pad/truncate to 200 samples
        signals = []
        for _, row in numeric_df.iterrows():
            signal = row.values.astype(np.float32)
            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            # Pad or truncate to 200 samples
            if len(signal) > 200:
                signal = signal[:200]
            else:
                signal = np.pad(signal, (0, 200 - len(signal)))
            signals.append(signal)
        return signals
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load EEG data:\n{e}")
        return None

# ------------------ Train Model ------------------

def train_transformer_model():
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = [], []
    for label in emotions_map:
        for _ in range(20):
            sig = np.sin(np.linspace(0, 6.28, 200)) + np.random.normal(0, (label + 1) / 5.0, 200)
            X.append(sig)
            y.append(label)

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = EEGTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, X_tensor, y_tensor

print("Training model, please wait...")
model, X_train, y_train = train_transformer_model()
print("Model training complete.")

# ------------------ Prediction ------------------

def predict_emotion(signal):
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output = model(signal_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

# ------------------ Simulated LLM Response ------------------

def generate_llm_response(emotion_name):
    responses = {
        "Happy": "You seem happy! Keep up the positive vibes. 😊",
        "Sad": "Feeling sad is okay. Remember to take care of yourself and talk to someone you trust.",
        "Joy": "Joy is infectious! Spread the happiness around you.",
        "Angry": "It's normal to feel angry sometimes. Try some deep breathing to calm down.",
        "Stressed": "Stress can be tough. Take breaks and prioritize self-care.",
        "Relaxed": "You are relaxed. Great job maintaining your calmness.",
        "Fear": "Fear can be a signal to be cautious. Stay safe and grounded."
    }
    return responses.get(emotion_name, "Emotion detected. Keep being aware of your feelings!")

# ------------------ Globals for Navigation ------------------

loaded_signals = []
current_index = 0

# ------------------ Show Signal and Prediction ------------------

def show_current_signal():
    global loaded_signals, current_index

    if not loaded_signals:
        return

    signal = loaded_signals[current_index]
    pred = predict_emotion(signal)

    for widget in plot_frame.winfo_children():
        widget.destroy()
    for widget in cm_frame.winfo_children():
        widget.destroy()

    # Plot EEG Signal
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.plot(signal, color='blue')
    ax.set_title(f"EEG Signal (Line {current_index + 1})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Show prediction and emoji
    emotion_text = f"Predicted Emotion:\n{emotions_map[pred][0]} {emotions_map[pred][1]}"
    prediction_label.config(text=emotion_text, font=("Arial", 20), fg="blue")

    # Show LLM simulated response
    llm_response = generate_llm_response(emotions_map[pred][0])
    llm_response_label.config(text=llm_response, font=("Arial", 14), fg="green", wraplength=350, justify="left")

    # Show confusion matrix on training data
    with torch.no_grad():
        outputs = model(X_train)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    cm = confusion_matrix(y_train.cpu().numpy(), preds)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5), dpi=100)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[e[0] for e in emotions_map.values()],
        yticklabels=[e[0] for e in emotions_map.values()],
        ax=ax_cm
    )
    ax_cm.set_xlabel('Predicted', fontsize=12)
    ax_cm.set_ylabel('True', fontsize=12)
    ax_cm.set_title('Confusion Matrix (Training Data)', fontsize=14)
    fig_cm.tight_layout()
    canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
    canvas_cm.draw()
    canvas_cm.get_tk_widget().pack()

def show_next():
    global current_index, loaded_signals
    if loaded_signals and current_index < len(loaded_signals) - 1:
        current_index += 1
        show_current_signal()

def show_prev():
    global current_index, loaded_signals
    if loaded_signals and current_index > 0:
        current_index -= 1
        show_current_signal()

def jump_to_line():
    global current_index, loaded_signals
    val = jump_entry.get()
    if val.isdigit():
        idx = int(val) - 1  # Convert 1-based to 0-based index
        if 0 <= idx < len(loaded_signals):
            current_index = idx
            show_current_signal()
        else:
            messagebox.showwarning("Invalid Index", f"Enter a number between 1 and {len(loaded_signals)}")
    else:
        messagebox.showwarning("Invalid Input", "Please enter a valid line number.")

# ------------------ Upload Button ------------------

def upload_dataset():
    global loaded_signals, current_index
    file_path = filedialog.askopenfilename(
        title="Select EEG Dataset",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
    )
    if file_path:
        signals = load_eeg_signal_from_file(file_path)
        if signals is not None and len(signals) > 0:
            loaded_signals = signals
            current_index = 0
            show_current_signal()
        else:
            messagebox.showinfo("Info", "No valid EEG signals found in the file.")

# ------------------ GUI Setup ------------------

root = tk.Tk()
root.title("🎧 EEG Emotion Detector & AI Insight")
root.geometry("1400x1000")
root.configure(bg="#f9fafb")
root.resizable(False, False)

top_frame = tk.Frame(root, padx=15, pady=15, bg="#f9fafb")
top_frame.pack()

upload_btn = tk.Button(top_frame, text="Upload EEG Dataset", command=upload_dataset,
                       font=("Arial", 16), bg="#4a90e2", fg="white", padx=10, pady=6)
upload_btn.pack()

# Navigation frame with Prev, Next, Jump controls
nav_frame = tk.Frame(top_frame, bg="#f9fafb")
nav_frame.pack(pady=(10, 0))

prev_btn = tk.Button(nav_frame, text="Previous", command=show_prev,
                     font=("Arial", 12), bg="#4a90e2", fg="white", padx=8)
prev_btn.grid(row=0, column=0, padx=5)

next_btn = tk.Button(nav_frame, text="Next", command=show_next,
                     font=("Arial", 12), bg="#4a90e2", fg="white", padx=8)
next_btn.grid(row=0, column=1, padx=5)

jump_label = tk.Label(nav_frame, text="Jump to line:", bg="#f9fafb", font=("Arial", 12))
jump_label.grid(row=0, column=2, padx=(15, 5))

jump_entry = tk.Entry(nav_frame, width=5, font=("Arial", 12))
jump_entry.grid(row=0, column=3, padx=5)

jump_btn = tk.Button(nav_frame, text="Go", command=jump_to_line,
                     font=("Arial", 12), bg="#4a90e2", fg="white", padx=8)
jump_btn.grid(row=0, column=4, padx=5)

main_frame = tk.Frame(root, bg="#f9fafb")
main_frame.pack(pady=15, fill=tk.BOTH, expand=True)

plot_frame = tk.LabelFrame(main_frame, text="EEG Signal Plot", font=("Arial", 14, "bold"), padx=10, pady=10, bg="#e1eaff")
plot_frame.grid(row=0, column=0, padx=20, sticky="n")

right_frame = tk.Frame(main_frame, bg="#f9fafb")
right_frame.grid(row=0, column=1, sticky="n")

prediction_label = tk.Label(right_frame, text="Predicted Emotion:\nNone", font=("Arial", 20), fg="gray", bg="#f9fafb")
prediction_label.pack(pady=(0, 10))

llm_response_label = tk.Label(right_frame, text="", font=("Arial", 14), fg="green", wraplength=350, justify="left", bg="#f9fafb")
llm_response_label.pack()

cm_frame = tk.LabelFrame(root, text="Confusion Matrix (Training Data)", font=("Arial", 14, "bold"), padx=10, pady=10, bg="#ffe9e9")
cm_frame.pack(padx=15, pady=15, fill=tk.BOTH, expand=False)

root.mainloop()
