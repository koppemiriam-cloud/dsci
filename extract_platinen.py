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
    green_hsv_lower: tuple[int, int, int] = (25, 25, 20)
    green_hsv_upper: tuple[int, int, int] = (95, 255, 255)
    closing_radius: int = 5  # Pixel der verkleinerten Erkennungskopie
    crop_padding: int = 5  # Pixel im Originalbild
    min_object_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.detection_max_side < 1:
            raise ValueError('detection_max_side muss mindestens 1 sein.')
        if self.closing_radius < 0 or self.crop_padding < 0:
            raise ValueError('closing_radius und crop_padding dürfen nicht negativ sein.')
        if not 0 < self.min_object_fraction <= 1:
            raise ValueError('min_object_fraction muss im Bereich (0, 1] liegen.')
        if len(self.green_hsv_lower) != 3 or len(self.green_hsv_upper) != 3:
            raise ValueError('HSV-Grenzen müssen jeweils drei Werte enthalten.')
        for lower, upper, maximum in zip(self.green_hsv_lower, self.green_hsv_upper,
                                         (179, 255, 255)):
            if not 0 <= lower <= upper <= maximum:
                raise ValueError('Ungültige HSV-Grenzen (H: 0..179, S/V: 0..255).')


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
    axes[0].set_title('Erkannte Platine')
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

    """Grüne Aussenkontur suchen; alle eingeschlossenen Bauteile erhalten.

    Die Maske füllt absichtlich auch Bohrungen innerhalb der Platine. Dunkle
    Pixel ausserhalb der Kontur werden nicht ergänzt: Dort liegen oft Schatten.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Erwarte ein RGB-Bild mit drei Kanälen und dtype uint8.')
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError('Das Bild darf nicht leer sein.')

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

    green_mask = cv2.inRange(
        hsv, np.array(config.green_hsv_lower, dtype=np.uint8),
        np.array(config.green_hsv_upper, dtype=np.uint8),
    )

    # Rand auffüllen, damit Closing auch bei angeschnittenen Platinen stabil ist.
    radius = config.closing_radius
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * radius + 1, 2 * radius + 1))
        padded = cv2.copyMakeBorder(green_mask, radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=0)
        green_mask = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
        green_mask = green_mask[radius:-radius, radius:-radius]

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

    if cv2.contourArea(largest_contour) < config.min_object_fraction * green_mask.size:
        raise ValueError(
            "Die erkannte Platinenfläche ist zu klein."
        )

    # Konturkoordinaten ins Original übertragen und dort füllen. So entstehen
    # keine grossen Pixelblöcke durch Nearest-Neighbour-Skalierung der Maske.
    contour = largest_contour.astype(np.float64)
    contour[:, :, 0] = (contour[:, :, 0] + 0.5) * width / detection_size[0] - 0.5
    contour[:, :, 1] = (contour[:, :, 1] + 0.5) * height / detection_size[1] - 0.5
    contour = np.rint(contour).astype(np.int32)
    board_mask = np.zeros((height, width), dtype=np.uint8)
    # FILLED erhält auch schwarze Chips und helle Aufdrucke innerhalb der Kontur.
    cv2.drawContours(board_mask, [contour], -1, 1, thickness=cv2.FILLED)
    board_mask = board_mask.astype(bool)

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

    return board_mask, bounds

def process_image(image_path: Path, config: Config, show_preview: bool = False) -> Path:
    # EXIF-Ausrichtung berücksichtigen, damit Hochformat nicht seitlich gespeichert wird.
    with Image.open(image_path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    mask, bounds = detect_board(image, config)
    top, left, bottom, right = bounds
    crop = np.dstack((image[top:bottom, left:right],
                      mask[top:bottom, left:right].astype(np.uint8) * 255))
    output_path = config.output_path / f'{image_path.stem}_crop.png'
    config.output_path.mkdir(parents=True, exist_ok=True)
    result = Image.fromarray(crop)
    result.save(output_path)
    result.save(config.output_path / f'{image_path.stem}_freigestellt.png')
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
                show_preview=(config.show_preview and index == 0)
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
