import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 1. Parameter definieren
IMG_SIZE = (224, 224)
BATCH_SIZE = 40
DATA_DIR = 'dataset'

print("Lade Datensatz...")

# 2. Trainings- und Validierungsdaten automatisch aus den Ordnern laden
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Erkannte Klassen: {class_names}")

# Performance-Optimierung für das Laden der Bilder
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Vortrainiertes Modell (MobileNetV2) als Basis laden (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Die Gewichte der Basis erst einmal einfrieren
base_model.trainable = False

# 4. Eigenes Netz aufbauen (Basis + Klassifikationsschicht für gut/schlecht)
model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)), # Normalisierung auf [-1, 1]
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax') # Softmax für die Klassenwahrscheinlichkeiten
])

# 5. Modell kompilieren
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# --- 6. Training starten ---
EPOCHS = 40  # oder wie viele Epochen du nutzen möchtest
print("\n--- Training startet ---")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --- 7. Modell abspeichern ---
model.save('pcb_quality_model_tf.keras')
print("Modell erfolgreich als 'pcb_quality_model_tf.keras' gespeichert!")
print("\n--- Training beendet! ---")


# --- 8. Trainingskurven plotten und speichern ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')

plt.title('Trainings- und Validierungsverlauf des PCB-Klassifikators')
plt.xlabel('Epochen')
plt.ylabel('Wert (Loss / Accuracy)')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('Trainingskurven.png', dpi=300, bbox_inches='tight')
print("Trainingskurven erfolgreich als 'Trainingskurven.png' gespeichert!")
plt.show()


# --- 9. Normalisierte Konfusionsmatrix erstellen und speichern ---
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Wahre Labels und Vorhersagen generieren
y = np.asarray(list(val_dataset.unbatch().map(lambda bild, label: label)))
yhat = model.predict(val_dataset)
yhatnum = yhat.argmax(axis=1)

# In Klassennamen ('Bad_Platine', 'Good_Platine') übersetzen
y_str = np.array(class_names)[y]
yhat_str = np.array(class_names)[yhatnum]

# Plotten der normalisierten Konfusionsmatrix
fig, ax = plt.subplots(figsize=(6, 6))

ConfusionMatrixDisplay.from_predictions(
    y_str, yhat_str,
    labels=class_names,
    normalize='true',       # Prozentuale Normalisierung
    cmap=plt.cm.Blues,
    ax=ax,
    values_format='.2f'     # Zeigt Werte von 0.00 bis 1.00
)

plt.title("Normalized Confusion Matrix (PCB-Klassifikation)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('Normalized_Konfusionsmatrix.png', dpi=300, bbox_inches='tight')
print("Normalisierte Konfusionsmatrix erfolgreich als 'Normalized_Konfusionsmatrix.png' gespeichert!")
plt.show()

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# 1. Detaillierter Klassifikationsbericht als Text (gibt Precision, Recall & F1 für jede Klasse aus)
print("\n--- Classification Report ---")
report = classification_report(y_str, yhat_str, target_names=class_names)
print(report)

# average='weighted' oder 'macro' sorgt dafür, dass bei mehreren Klassen gemittelt wird
prec = precision_score(y_str, yhat_str, average='weighted')
rec = recall_score(y_str, yhat_str, average='weighted')
f1 = f1_score(y_str, yhat_str, average='weighted')

print(f"Gesamt Precision: {prec:.4f}")
print(f"Gesamt Recall:    {rec:.4f}")
print(f"Gesamt F1-Score:  {f1:.4f}")