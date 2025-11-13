#builtin
from pathlib import Path
import random

#external
from PIL import Image
import numpy as np

#internal
from network import NeuralNetwork
from components.loss_function import CrossEntropy

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((64, 64))
    array = np.array(img, dtype=np.float32)   # 0–255
    normalize = array / 255.0                 # 0–1
    flattened_array = normalize.flatten().reshape(1, -1)
    return flattened_array

def build_train_image_info():
    digits = [0,1,2,3,4,5,6,7,8,9]
    image_info = []
    for c in range(1, 101):
        for b in range(1, 11):
            for x_idx, label in enumerate(digits, start=1):
                path = f"data/input_{c}_{b}_{x_idx}.jpg"
                image_info.append((path, label))
    return image_info

def iter_image_batches(image_info, batch_size):
    for i in range(0, len(image_info), batch_size):
        batch = image_info[i:i+batch_size]
        X_list, y_list = [], []
        for p, y in batch:
            p = Path(p)
            if not p.exists():
                continue
            X_list.append(load_image(p))
            y_list.append(y)
        if not X_list:
            continue
        X = np.vstack(X_list)            
        y = np.asarray(y_list, dtype=np.int64)
        yield X, y

def build_dataset(image_info):
    X_list, y_list = [], []
    for path, label in image_info:
        x_i = load_image(path)
        X_list.append(x_i)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y

def debug_label_distribution(image_info, name="dataset"):
    _, y = build_dataset(image_info)
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution for {name}:")
    print(dict(zip(unique, counts)))

def train_model(image_info, batch_size=32, num_epochs=5):
    first_path = Path(image_info[0][0])
    D = load_image(first_path).shape[1]

    loss_fn = CrossEntropy()
    net = NeuralNetwork(
        dimensions=[D, 256, 128, 10],
        learning_rate=0.05,
        loss_function=loss_fn,
    )

    for epoch in range(num_epochs):
        random.shuffle(image_info)  

        batches = 0
        for X_batch, y_batch in iter_image_batches(image_info, batch_size):
            net.train(X_batch, y_batch)
            batches += 1

        if (epoch + 1) % 1 == 0:
            sample = image_info[: min(512, len(image_info))]
            X_s, y_s = next(iter_image_batches(sample, len(sample)))
            probs = net.predict(X_s)
            loss = loss_fn.get_test_loss(y_s, probs)
            print(f"Epoch {epoch+1}/{num_epochs}  batches:{batches}  loss:{loss:.4f}")

    net.save_network("trained_network.json")
    return net

def test_model(image_info):
    X, y = build_dataset(image_info)
    loss_fn = CrossEntropy()
    net = NeuralNetwork.load_network("trained_network.json", loss_function=loss_fn)

    probs = net.predict(X)
    predicted_digits = np.argmax(probs, axis=1)

    accuracy = np.mean(predicted_digits == y)
    print("True labels:     ", y)
    print("Predicted labels:", predicted_digits)
    print(f"Accuracy: {accuracy:.3f}")

    return predicted_digits, accuracy

"""
def main():
    all_info = build_train_image_info() 
    random.shuffle(all_info)

    # --- tiny subset to overfit on ---
    tiny_info = all_info[:100]   # 100 images

    print("Trying to overfit tiny set...")
    train_model(tiny_info, batch_size=32, num_epochs=200)

    print("Performance on the SAME tiny set:")
    preds, acc = test_model(tiny_info)
    print("Tiny-set accuracy:", acc)"""

def main():
    all_info = build_train_image_info()
    random.shuffle(all_info)
    split = int(0.8 * len(all_info))
    train_image_info = all_info[:split]
    val_image_info   = all_info[split:]

    debug_label_distribution(train_image_info, "train")
    debug_label_distribution(val_image_info, "val")

    train_model(train_image_info, batch_size=32, num_epochs=30)
    print("Validation performance:")
    test_model(val_image_info)

if __name__ == "__main__":
    main()