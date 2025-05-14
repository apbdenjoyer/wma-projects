import os
from pathlib import Path
import numpy as np
from keras import layers, models
import tensorflow
from keras.src.utils import to_categorical
from tensorflow.keras.preprocessing import image
from PIL import Image
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2

IMAGE_SIZE = (128,128)
MODEL_PATH = 'models/wma_mp4.keras'

def get_labeled_faces(data_dir):
    label_map = {}

    X = []
    y = []

    current_label = 0
    for root, dirs, files in os.walk(data_dir):
        path_obj = Path(root)

        if path_obj == Path(data_dir):
            continue


        label_name = 'others' if path_obj.parent.name == 'others' else path_obj.name

        if label_name not in label_map:
            label_map[label_name] = current_label
            current_label += 1

        for file in files:
            file_path = os.path.join(root, file)
            img = Image.open(file_path).convert('RGB')
            img = img.resize(IMAGE_SIZE)
            img_array = np.asarray(img) / 255.0
            X.append(img_array)
            y.append(label_map[label_name])

    # inverted here for convenience earlier
    inv_label_map = {v: k for k, v in label_map.items()}
    return np.array(X), np.array(y), inv_label_map

def setup_model(layer_num: int = 3):
    # we're assuming 3 classes: man1, woman1, and others
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), padding='same'))

    for i in range(layer_num):
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()
    return model

def get_model():
    data_dir = "data"
    images, labels, label_map = get_labeled_faces(data_dir)

    train_images, test_images, train_labels, test_labels = train_test_split(images,labels,train_size=0.8, stratify=labels)

    print(label_map)

    print(f"Shape X_train:{train_images.shape}")
    print(f"Shape y_train:{train_labels.shape}")
    print(f"Shape X_test:{test_images.shape}")
    print(f"Shape y_test:{test_labels.shape}")

    train_labels = to_categorical(train_labels, num_classes=3)
    test_labels = to_categorical(test_labels, num_classes=3)

    model = None
    if os.path.exists(MODEL_PATH):
        print("Model found, loading...")
        model = models.load_model(MODEL_PATH)
    else:
        print("Model not found, creating a new one...")
        model = setup_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=3, batch_size=16)
        os.makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)

    train_model(label_map, model, test_images, test_labels)
    return label_map, model


def train_model(label_map, model, test_images,test_labels):
    pred_labels = model.predict(test_images)
    num_img = 100
    plt.figure(figsize=(15, 10))
    plt.suptitle("predicted | true", fontsize=20)
    for i in range(num_img):
        label = np.argmax(pred_labels[i])
        true_label = np.argmax(test_labels[i])
        color = 'green' if label == true_label else 'red'
        plt.subplot(10, 10, i + 1)
        plt.imshow(test_images[i])
        plt.title(f"{label_map[label]} | {label_map[true_label]}", color=color)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.5, hspace=2.0)
    plt.tight_layout(pad=2.0)
    plt.show()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc = ', test_acc)

list_haarcascade = ['haarcascade_frontalface_default.xml']
object_cascade = None
f = 105
n = 5
s = 50

def set_object():
    global object_cascade
    object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + list_haarcascade[0])
    current_feature = list_haarcascade[0].split('.')[0].split('haarcascade_')[1]

window_name = "webcam"
colors = {
    0: (0,255,0),
    1: (255,0,0),
    2:(0,0,255)
}

def main(label_map, model):
    global object_cascade, f, n, s, faces_map, window_name, colors

    cap = cv2.VideoCapture(0)
    set_object()

    cv2.namedWindow(window_name)

    ret, frame = cap.read()
    cv2.createTrackbar('scaleFactor', window_name, 101, 1000, fun)
    cv2.createTrackbar('minNeighbors', window_name, 1, 500, fun)
    cv2.createTrackbar('minSize', window_name, 1, 500, fun)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = object_cascade.detectMultiScale(gray, scaleFactor=f / 100, minNeighbors=n, minSize=(s, s))

        for (x, y, w, h) in faces:
            # finding faces
            face_region = frame[y:y + h, x:x + w]
            face_region_resized = cv2.resize(face_region, IMAGE_SIZE)
            face_region_resized = np.expand_dims(face_region_resized, axis=0) / 255.0

            # predicting
            pred_label = model.predict(face_region_resized)
            label = label_map[np.argmax(pred_label)]
            color = colors[np.argmax(pred_label)]

            # drawing
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == ord('n'):
            set_object()
        elif key == 27:
            cv2.destroyAllWindows()
            break

def fun(x):
    global f, n, s
    f = cv2.getTrackbarPos('scaleFactor', window_name)
    if f < 101:
        f = 101
    n = cv2.getTrackbarPos('minNeighbors', window_name)
    s = cv2.getTrackbarPos('minSize', window_name)

if __name__ == "__main__":
    label_map, model = get_model()
    main(label_map, model)