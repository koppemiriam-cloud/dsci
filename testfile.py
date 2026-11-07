import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# 1. Das gespeicherte Modell laden
model = tf.keras.models.load_model('pcb_quality_model_tf.keras')

# 2. Pfad zu deinem Bild (passe den Namen an, falls es in einem Unterordner liegt)
img_path = 'IMG_0100.png'

# 3. Bild laden und auf die Standard-Größe von MobileNetV2 skalieren (z.B. 224x224)
# Hinweis: Nutze exakt die Bildgröße, die du auch beim Training verwendet hast!
img_height, img_width = 224, 224
image = load_img(img_path, target_size=(img_height, img_width))

# 4. In ein NumPy-Array umwandeln und die Batch-Dimension hinzufügen
input_arr = img_to_array(image)
input_arr = np.array([input_arr])  # Shape wird zu (1, 224, 224, 3)

# 5. Vorhersage berechnen
predictions = model.predict(input_arr)

# 6. Ergebnis interpretieren
# (Je nach Ordnerstruktur sortiert TensorFlow die Klassen alphabetisch: z.B. ['Bad', 'Good'])
class_names = ['Bad', 'Good']
predicted_index = predictions.argmax(axis=1)[0]
confidence = float(np.max(predictions)) * 100

print(f"\n--- Vorhersage für {img_path} ---")
print(f"Ergebnis: {class_names[predicted_index]} (Sicherheit: {confidence:.2f}%)")
print(f"Rohwerte der Wahrscheinlichkeiten: {predictions}")