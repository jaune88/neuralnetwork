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
    img = Image.open(path)
    array = np.array(img)
    normalize = array / 255.0
    flattened_array = normalize.flatten().reshape(1, -1)
    return flattened_array

def build_train_image_info():
    digits = [0,1,2,3,4,5,6,7,8,9]
    image_info = []
    for c in range(1, 51):
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

def train_model(image_info, batch_size=32, num_epochs=5):
    first_path = Path(image_info[0][0])
    D = load_image(first_path).shape[1]

    loss_fn = CrossEntropy()
    net = NeuralNetwork(
        dimensions=[D, 128, 10],
        learning_rate=0.01,
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
    
    return predicted_digits

def main():
    train_image_info = build_train_image_info()
    train_model(train_image_info, batch_size=32, num_epochs=5)

    test_image_info = [
        (Path(__file__).resolve().parent / "data/input_100_6_5.jpg", 4),
        (Path(__file__).resolve().parent / "data/input_100_3_2.jpg", 1),
    ]
    test_model(test_image_info)
    
    preds = test_model(test_image_info)
    print("Predicted digits:", preds)


if __name__ == "__main__":
    main()