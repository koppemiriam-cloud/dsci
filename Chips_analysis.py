"""Die N schwarzen IC-Gehäuse (grösste schwarze Flächen) aus freigestellten Platinen ausschneiden.

Vorgehen pro Bild:
1. Platinenform aus dem Alphakanal bestimmen, Randzone (Schatten!) ausschliessen.
2. Dunkle Pixel suchen, Flächen bilden (Erosion/Dilatation via object_extraction.create_mask).
3. Jede Fläche nach Anzahl schwarzer Pixel bewerten, die N besten behalten.
4. Aus dem Originalbild ausschneiden und speichern.

Diagnose: Für das Vorschaubild gibt es einen 2x2-Plot (Original mit Kandidaten, Maske der
schwarzen Pixel, Flächen mit Score, Value-Kanal mit Schwellwertlinie) und eine Kandidatentabelle
in der Konsole.

Voraussetzung: *_freigestellt.png in 'Ausschnitte'. object_extraction.py im selben Ordner (unverändert).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import warnings

# Harmlose skimage-Warnung aus object_extraction.py (square) ausblenden
warnings.filterwarnings('ignore', category=FutureWarning, module='object_extraction')

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.measure import label, regionprops

import object_extraction as extraction


@dataclass(frozen=True)
class Config:
    input_path: Path = Path(__file__).resolve().parent / 'Ausschnitte'
    output_path: Path = Path(__file__).resolve().parent / 'Gehaeuse'
    input_pattern: str = '*_freigestellt.png'
    expected_count: int = 7             # Anzahl Gehäuse, die behalten werden
    quick_test: bool = False
    sample_size: int = 4
    random_seed: int = 0
    show_preview: bool = True
    # Dateinamen, für die der Diagnose-Plot gezeigt wird (leer = nur das erste Bild),
    # z.B. ('bad_IMG_0125_freigestellt.png',)
    preview_names: tuple[str, ...] = ()
    detection_max_side: int = 1600      # Erkennung auf verkleinerter Kopie, Ausschnitt aus Original
    value_threshold: int = -70          # negativ: Value < 70 gilt als schwarz
    erosion_size: int = 3               # entfernt Einzelpixel (Pixel der verkleinerten Kopie)
    dilation_size: int = 9              # verbindet Gehäuse mit hellem Aufdruck zu einer Fläche
    edge_margin_fraction: float = 0.03  # Randzone der Platine (Anteil der kleineren Platinenseite)
    # Mindestdichte: Anteil schwarzer Pixel in der Bounding-Box ihres schwarzen Kerns.
    # ICs sind kompakte, dichte Flächen; verstreute dunkle Lötstellen/Pads sind es nicht.
    min_density: float = 0.6
    separation_ratio: float = 2.0       # Warnung, wenn Rang N nicht >= ratio * Rang N+1 ist
    crop_padding: int = 10              # Pixel im Originalbild um jedes Gehäuse
    table_rows: int = 12                # Zeilen der Kandidatentabelle in der Konsole


@dataclass
class Candidate:
    score: int                              # Anzahl schwarzer Pixel (verkleinerte Kopie)
    box: tuple[int, int, int, int]          # (top, left, bottom, right) im Originalbild, mit Padding
    box_small: tuple[int, int, int, int]    # (minr, minc, maxr, maxc) in der verkleinerten Kopie
    mean_v: float                           # mittlerer Value (0..255) der Fläche
    density: float = 1.0                    # schwarze Pixel / Fläche der Bounding-Box des schwarzen Kerns
    eligible: bool = True                   # Dichte >= min_density
    selected: bool = False


@dataclass
class Detection:
    candidates: list[Candidate]             # alle Flächen, absteigend nach Score
    raw_black: np.ndarray                   # schwarze Pixel (verkleinerte Kopie)
    morphed: np.ndarray                     # Maske nach Erosion/Dilatation
    v_small: np.ndarray                     # Value-Kanal (verkleinerte Kopie)
    valid_small: np.ndarray                 # erlaubter Bereich ohne Rand
    expected_count: int

    @property
    def selected(self) -> list[Candidate]:
        # oben nach unten, dann links nach rechts
        return sorted((c for c in self.candidates if c.selected), key=lambda c: c.box)

    @property
    def runner_up_score(self) -> int:
        rejected = [c.score for c in self.candidates if c.eligible and not c.selected]
        return max(rejected) if rejected else 0


def find_images(config: Config) -> list[Path]:
    if not config.input_path.is_dir():
        raise FileNotFoundError(f'Eingabeordner nicht gefunden: {config.input_path}')
    images = sorted(p for p in config.input_path.glob(config.input_pattern) if p.is_file())
    if not images:
        raise FileNotFoundError(f'Keine Dateien {config.input_pattern} in {config.input_path} gefunden.')
    return images


def select_images(images: list[Path], config: Config) -> list[Path]:
    if not config.quick_test:
        return images
    return random.Random(config.random_seed).sample(images, min(config.sample_size, len(images)))


def load_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """RGB-uint8-Bild (transparente Pixel auf Weiss) und boolesche Platinenmaske (Alpha > 127)."""
    with Image.open(image_path) as source:
        rgba = source.convert('RGBA')
    board_mask = np.asarray(rgba)[:, :, 3] > 127
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(background, rgba).convert('RGB'))
    return rgb, board_mask


def detect_housings(image: np.ndarray, board_mask: np.ndarray, config: Config) -> Detection:
    height, width = image.shape[:2]
    scale = min(1.0, config.detection_max_side / max(height, width))
    if scale < 1.0:
        size = (round(width * scale), round(height * scale))
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        board_small = cv2.resize(board_mask.astype(np.uint8) * 255, size,
                                 interpolation=cv2.INTER_AREA) > 127
    else:
        small = image
        board_small = board_mask

    rows, cols = np.where(board_small)
    if len(rows) == 0:
        raise ValueError('Keine Platine (Alphakanal) gefunden.')

    # Randzone: nur Pixel, die weiter als margin_px vom Platinenrand (bzw. Bildrand) entfernt sind
    board_size = min(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    margin_px = config.edge_margin_fraction * board_size
    padded = np.pad(board_small.astype(np.uint8) * 255, 1)  # Bildrand zählt ebenfalls als Rand
    valid = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1] > margin_px

    # Ungültige Bereiche weiss -> können nie schwarzes Gehäuse werden
    small_valid = np.where(valid[:, :, None], small, 255).astype(np.uint8)
    v_small = small_valid.max(axis=2)  # Value-Kanal in HSV = Maximum von R, G, B

    raw_black = extraction.generate_mask_with_hsv_threshold(
        small_valid, value_threshold=config.value_threshold)
    morphed, _ = extraction.create_mask(
        small_valid,
        fraction_of_rows_to_remove=0.0,
        fraction_of_cols_to_remove=0.0,
        value_threshold=config.value_threshold,
        erosion_size=config.erosion_size,
        dilation_size=config.dilation_size,
        value_to_fill=255,
    )

    # Jede Fläche nach Anzahl schwarzer Pixel bewerten
    candidates: list[Candidate] = []
    for region in regionprops(label(morphed)):
        minr, minc, maxr, maxc = region.bbox
        black_in_region = raw_black[minr:maxr, minc:maxc] & region.image
        score = int(np.count_nonzero(black_in_region))
        if score == 0:
            continue
        ys, xs = np.nonzero(black_in_region)
        core_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        density = score / core_area
        mean_v = float(v_small[minr:maxr, minc:maxc][region.image].mean())
        box = (
            max(0, int(minr / scale) - config.crop_padding),
            max(0, int(minc / scale) - config.crop_padding),
            min(height, int(np.ceil(maxr / scale)) + config.crop_padding),
            min(width, int(np.ceil(maxc / scale)) + config.crop_padding),
        )
        candidates.append(Candidate(score=score, box=box, box_small=region.bbox, mean_v=mean_v,
                                    density=density, eligible=density >= config.min_density))

    candidates.sort(key=lambda c: c.score, reverse=True)
    eligible = [c for c in candidates if c.eligible]
    for candidate in eligible[:config.expected_count]:
        candidate.selected = True

    return Detection(candidates=candidates, raw_black=raw_black, morphed=morphed,
                     v_small=v_small, valid_small=valid, expected_count=config.expected_count)


def print_candidates(detection: Detection, config: Config, name: str) -> None:
    print(f'\nKandidaten für {name} (nach Score sortiert):')
    print(f'{"Rang":>4} {"Score":>7} {"Breite×Höhe (Original)":>24} {"mittl. Value":>13} {"Dichte":>7}  Auswahl')
    for rank, c in enumerate(detection.candidates[:config.table_rows]):
        top, left, bottom, right = c.box
        if c.selected:
            status = 'ja'
        elif not c.eligible:
            status = f'nein (Dichte < {config.min_density})'
        else:
            status = 'nein'
        print(f'{rank:>4} {c.score:>7} {right - left:>10}×{bottom - top:<13} {c.mean_v:>13.0f} '
              f'{c.density:>7.2f}  {status}')


def preview_image(image: np.ndarray, detection: Detection, config: Config, title: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 16))
    ax_img, ax_mask, ax_cand, ax_v = axes.ravel()

    # 1) Original mit Kandidaten (rot = gewählt, blau = abgelehnt)
    ax_img.imshow(image)
    for rank, c in enumerate(detection.candidates[:config.expected_count + 8]):
        top, left, bottom, right = c.box
        colour = 'red' if c.selected else ('deepskyblue' if c.eligible else 'magenta')
        ax_img.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                   edgecolor=colour, linewidth=2,
                                   linestyle='-' if c.selected else '--'))
        ax_img.text(left, top, f'#{rank} ({c.score}, d={c.density:.2f})', color='yellow',
                    fontsize=9, va='bottom',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
    ax_img.set_title(f'{title}\nrot = gewählt, blau = abgelehnt, magenta = zu locker '
                     f'(Rang, Score, Dichte)')

    # 2) Maske: weiss = schwarze Pixel, dunkelblau = ausgeschlossene Randzone, schwarz = hell
    mask_rgb = np.zeros((*detection.raw_black.shape, 3), dtype=np.uint8)
    mask_rgb[~detection.valid_small] = (0, 0, 110)
    mask_rgb[detection.raw_black] = (255, 255, 255)
    ax_mask.imshow(mask_rgb)
    ax_mask.set_title('Maske: weiss = schwarze Pixel, blau = Randzone (ignoriert)')

    # 3) Flächen nach Erosion/Dilatation mit Score
    ax_cand.imshow(detection.morphed, cmap='gray', vmin=0, vmax=2)
    for rank, c in enumerate(detection.candidates[:config.expected_count + 8]):
        minr, minc, maxr, maxc = c.box_small
        colour = 'lime' if c.selected else ('orange' if c.eligible else 'magenta')
        ax_cand.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False,
                                    edgecolor=colour, linewidth=2))
        ax_cand.text(minc, minr, f'#{rank} ({c.score})', color=colour, fontsize=9, va='bottom')
    ax_cand.set_title('Flächen nach Erosion/Dilatation: grün = gewählt, orange = abgelehnt, '
                      'magenta = zu locker')

    # 4) Value-Kanal mit Schwellwertlinie
    shown = ax_v.imshow(detection.v_small, cmap='magma', vmin=0, vmax=255)
    ax_v.contour(detection.v_small, levels=[abs(config.value_threshold)],
                 colors='cyan', linewidths=0.6)
    figure.colorbar(shown, ax=ax_v, fraction=0.046, pad=0.02)
    ax_v.set_title(f'Value-Kanal (0..255), cyan = Schwellwert {abs(config.value_threshold)}')

    for axis in axes.ravel():
        axis.set_axis_off()
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def process_image(image_path: Path, config: Config, show_preview: bool = False) -> bool:
    image, board_mask = load_image(image_path)
    detection = detect_housings(image, board_mask, config)
    selected = detection.selected

    base = image_path.stem.removesuffix('_freigestellt')
    for i, c in enumerate(selected):
        top, left, bottom, right = c.box
        out = config.output_path / f'{base}_gehaeuse_{i}.png'
        Image.fromarray(image[top:bottom, left:right]).save(out)

    problems = []
    if len(selected) < config.expected_count:
        problems.append(f'nur {len(selected)} statt {config.expected_count} Flächen gefunden')
    elif min(c.score for c in selected) < config.separation_ratio * detection.runner_up_score:
        problems.append(f'unklare Trennung: kleinste Auswahl {min(c.score for c in selected)} Pixel, '
                        f'beste abgelehnte Fläche {detection.runner_up_score} Pixel')

    message = f'{image_path.name}: {len(selected)} Gehäuse gespeichert'
    if problems:
        message += '  <-- ACHTUNG: ' + '; '.join(problems)
    print(message)

    if show_preview:
        print_candidates(detection, config, image_path.name)
        preview_image(image, detection, config, image_path.name)
    return not problems


def main(config: Config | None = None) -> None:
    config = config if config is not None else Config()

    images = select_images(find_images(config), config)
    config.output_path.mkdir(parents=True, exist_ok=True)
    print(f'Verarbeite {len(images)} Bilder.')

    ok, unsicher, fehler = 0, 0, 0
    for index, image_path in enumerate(images):
        if config.preview_names:
            show = image_path.name in config.preview_names
        else:
            show = index == 0
        try:
            clean = process_image(image_path, config,
                                  show_preview=config.show_preview and show)
            if clean:
                ok += 1
            else:
                unsicher += 1
        except Exception as error:
            fehler += 1
            print(f'FEHLER bei {image_path.name}: {error}')

    print('\n--- Zusammenfassung ---')
    print(f'Eindeutig: {ok}')
    print(f'Mit Warnung (bitte prüfen): {unsicher}')
    print(f'Fehlgeschlagen: {fehler}')
    print(f'Gespeichert in: {config.output_path}')


if __name__ == '__main__':
    main()