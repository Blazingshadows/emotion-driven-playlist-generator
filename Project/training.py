import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
dictionary = {}
c = 0

print("Available .npy files:")
for file in os.listdir():
    if file.endswith(".npy"):
        print(f"- {file}")

valid_npy_files = [f for f in os.listdir() if f.endswith(".npy") and not f.startswith("labels")]
if not valid_npy_files:
    print("No valid emotion .npy files found. Please make sure your data files exist.")
    exit()

print(f"Found {len(valid_npy_files)} valid emotion files to process.")

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        data = np.load(i)
        print(f"Loaded file: {i}, Shape: {data.shape}")
        
        if not is_init:
            is_init = True 
            X = data
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
            print(f"Initial X shape: {X.shape}")
        else:
            if X.shape[1] != data.shape[1]:
                print(f"Shape mismatch! File: {i}, Expected: {X.shape[1]}, Got: {data.shape[1]}")
                raise ValueError(f"Shape mismatch in file: {i}")
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

print(f"X shape before processing: {X.shape}")
if len(X.shape) == 1:
    X = X.reshape(-1, 1)
elif len(X.shape) > 2:
    X = X.reshape(X.shape[0], -1)  
print(f"X shape after processing: {X.shape}")


for i in range(y.shape[0]):
    y[i, 0] = dictionary[str(y[i, 0])]

y = np.array(y, dtype="int32")


y = to_categorical(y)
print(f"y shape: {y.shape}")


cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
X = X[cnt]
y = y[cnt]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"Input shape for model: {(X.shape[1],)}")
ip = Input(shape=(X.shape[1],))  

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)


model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])


model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


print("\n--- Detailed Classification Metrics ---\n")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")


print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=label,
           yticklabels=label)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")


model.save("model.h5")
np.save("labels.npy", np.array(label))