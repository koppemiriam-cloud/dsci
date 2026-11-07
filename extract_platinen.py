"""Platinen lokalisieren und mit transparentem Hintergrund in Originalauflösung speichern."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_closing, binary_fill_holes

import object_extraction as extraction

# Frühere Kaggle-Pfade (lokal nicht benötigt):
# INPUT_PATH = Path('/kaggle/input/datasets/miriamkoppe/platinen')
# OUTPUT_PATH = Path('/kaggle/working/Ausschnitte')
# EXTRACTION_MODULE_PATH = Path('/kaggle/input/datasets/toedtlifhsg/object-extraction')


@dataclass(frozen=True)
class Config:
    """Erkennung für eine einzelne Platine auf hellem Hintergrund pro Foto."""

    input_path: Path = Path(__file__).resolve().parent / 'Platinen'
    output_path: Path = Path(__file__).resolve().parent / 'Ausschnitte'
    image_extension: str = 'png'
    quick_test: bool = False
    sample_size: int = 4
    random_seed: int = 0
    show_preview: bool = True
    detection_max_side: int = 1600
    value_threshold: float = -190.0  # Vordergrund dunkler als 190 (HSV, 0..255)
    closing_radius: int = 5  # Pixel der verkleinerten Erkennungskopie
    crop_padding: int = 5  # Pixel im Originalbild
    min_object_fraction: float = 0.01


def find_images(config: Config) -> list[Path]:
    if not config.input_path.is_dir():
        raise FileNotFoundError(f'Bildordner nicht gefunden: {config.input_path}')
    images = sorted(
        path for path in config.input_path.glob(f'*.{config.image_extension}')
        if path.is_file()
    )
    if not images:
        raise FileNotFoundError(
            f'Keine .{config.image_extension}-Bilder in {config.input_path} gefunden.'
        )
    return images


def select_images(images: list[Path], config: Config) -> list[Path]:
    if not config.quick_test:
        return images
    if config.sample_size < 1:
        raise ValueError('Die Stichprobengröße muss mindestens 1 sein.')
    return random.Random(config.random_seed).sample(
        images, min(config.sample_size, len(images))
    )


def preview_image(image: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int],
                  crop: np.ndarray, title: str) -> None:
    top, left, bottom, right = bounds
    figure, axes = plt.subplots(1, 3, figsize=(15, 7))
    axes[0].imshow(image)
    axes[0].add_patch(Rectangle(
        (left, top), right - left, bottom - top,
        fill=False, edgecolor='red', linewidth=2,
    ))
    axes[0].set_title('Recognized Area')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Maske')
    axes[2].imshow(crop)
    axes[2].set_title('Freigestellte Platine (RGBA)')
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def detect_board(
    image: np.ndarray,
    config: Config
) -> tuple[np.ndarray, tuple[int, int, int, int]]:

    height, width = image.shape[:2]

    # Bild für die Erkennung verkleinern
    scale = min(
        1.0,
        config.detection_max_side / max(height, width)
    )

    detection_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale))
    )

    detection_image = cv2.resize(
        image,
        detection_size,
        interpolation=cv2.INTER_AREA
    )

    # RGB in HSV umwandeln
    hsv = cv2.cvtColor(
        detection_image,
        cv2.COLOR_RGB2HSV
    )

    # Präzisere HSV-Grenzwerte (verhindert, dass der weiße Hintergrund/Schatten fälschlicherweise erfasst wird)
    lower_green = np.array([25, 25, 20])
    upper_green = np.array([95, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Moderater Kernel, um Lücken zu schliessen, ohne den Hintergrund zu berühren
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Alle Konturen finden
    contours, _ = cv2.findContours(
        green_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "Keine grüne Platinenfläche erkannt."
        )

    # Grösste Kontur wählen (die eigentliche Platine)
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 50:
        raise ValueError(
            "Die erkannte Platinenfläche ist zu klein."
        )

    # Maske exakt entlang der Platinen-Kontur zeichnen
    mask_small = np.zeros(
        detection_image.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        mask_small,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Innere Löcher (schwarze Gehäuse, Chips) füllen, damit sie im Crop erhalten bleiben
    mask_small = binary_fill_holes(mask_small).astype(np.uint8) * 255

    # Äussere Platinenmaske auf Originalauflösung skalieren
    board_mask = cv2.resize(
        mask_small,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    ) > 0

    # Begrenzungsrahmen anhand der Platine bestimmen
    rows, cols = np.where(board_mask)

    if len(rows) == 0:
        raise ValueError("Keine Platine erkannt.")

    padding = config.crop_padding

    top = max(0, int(rows.min()) - padding)
    left = max(0, int(cols.min()) - padding)

    bottom = min(height, int(rows.max()) + 1 + padding)
    right = min(width, int(cols.max()) + 1 + padding)

    bounds = (top, left, bottom, right)

    # Die saubere Platinenmaske als finale Anzeige-/Speichermaske nutzen
    display_mask = board_mask

    return display_mask, bounds

def process_image(image_path: Path, config: Config, show_preview: bool = False) -> Path:
    # EXIF-Ausrichtung berücksichtigen, damit Hochformat nicht seitlich gespeichert wird.
    with Image.open(image_path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    mask, bounds = detect_board(image, config)
    top, left, bottom, right = bounds
    crop = np.dstack((image[top:bottom, left:right],
                      mask[top:bottom, left:right].astype(np.uint8) * 255))
    output_path = config.output_path / f'{image_path.stem}_crop.png'
    Image.fromarray(crop).save(output_path)
    Image.fromarray(crop).save(config.output_path / f'{image_path.stem}_freigestellt.png')
    print(f'{image_path.name}: {image.shape[1]}×{image.shape[0]} → '
          f'{crop.shape[1]}×{crop.shape[0]} Pixel; {output_path}')
    if show_preview:
        preview_image(image, mask, bounds, crop, image_path.name)
    return output_path


def main(config: Config | None = None) -> None:
    config = config if config is not None else Config()

    images = select_images(find_images(config), config)
    config.output_path.mkdir(parents=True, exist_ok=True)

    print(f"Verarbeite {len(images)} Bilder.")

    erfolgreich = 0
    fehlgeschlagen = 0

    for index, image_path in enumerate(images):
        print(f"\nBild {index + 1}/{len(images)}: {image_path.name}")

        try:
            process_image(
                image_path,
                config,
                show_preview=(index == 0)
            )
            erfolgreich += 1

        except Exception as error:
            fehlgeschlagen += 1
            print(f"FEHLER bei {image_path.name}: {error}")

    print("\n--- Zusammenfassung ---")
    print(f"Erfolgreich verarbeitet: {erfolgreich}")
    print(f"Fehlgeschlagen: {fehlgeschlagen}")
    print(f"Gespeicherte Ausschnitte: {config.output_path}")


if __name__ == '__main__':
    main()
