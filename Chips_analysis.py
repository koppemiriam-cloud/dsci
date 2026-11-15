"""IC-Gehäuse mit PCB-YOLO erkennen und aus freigestellten Platinen ausschneiden.

Start: python Chips_analysis.py --quick-test --no-preview
Abhängigkeiten: python -m pip install -r requirements-chips.txt
Modellquelle: https://huggingface.co/Arshia82sbn/pcb-yolov8s-detection
Beim ersten Start werden Modellgewichte heruntergeladen, Bilder bleiben lokal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    input_path: Path = ROOT / 'Ausschnitte'
    # Separater Ordner verhindert Vermischung mit alten Schwellwert-Ergebnissen.
    output_path: Path = ROOT / 'Gehaeuse_YOLO'
    model_path: Path | None = None
    confidence: float = 0.25
    iou_threshold: float = 0.45
    tile_size: int = 1280
    tile_overlap: float = 0.25
    image_size: int = 640
    crop_padding: int = 15
    expected_count: int = 7  # nur Diagnose; keine erzwungene Auswahl von sieben Objekten
    quick_test: bool = False
    show_preview: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1 or not 0 < self.iou_threshold <= 1:
            raise ValueError('Konfidenz und IoU müssen in (0, 1] liegen.')
        if not 0 <= self.tile_overlap < 1:
            raise ValueError('tile_overlap muss in [0, 1) liegen.')
        if self.tile_size < 1 or self.image_size < 1 or self.crop_padding < 0:
            raise ValueError('Ungültige Bildgrösse oder negativer Ausschnittrand.')


@dataclass(frozen=True)
class Candidate:
    box: tuple[float, float, float, float]  # links, oben, rechts, unten, ohne Padding
    confidence: float


def tile_starts(length: int, size: int, overlap: float) -> list[int]:
    last = max(0, length - size)
    step = max(1, round(size * (1 - overlap)))
    return sorted(set(range(0, last + 1, step)) | {last})


def intersection_over_union(a: Candidate, b: Candidate) -> float:
    x1, y1 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    x2, y2 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])
    area_b = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicates(candidates: list[Candidate], threshold: float) -> list[Candidate]:
    kept = []
    for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        if all(intersection_over_union(candidate, other) < threshold for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda c: (c.box[1], c.box[0]))


def load_model(config: Config):
    from ultralytics import YOLO
    weights = config.model_path
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = Path(hf_hub_download(
            repo_id='Arshia82sbn/pcb-yolov8s-detection', filename='best.pt',
            revision='7eb710b', local_dir=ROOT / 'models' / 'pcb-yolov8s',
        ))
    elif not weights.is_file():
        raise FileNotFoundError(f'Modell nicht gefunden: {weights}')
    model = YOLO(str(weights))
    names = model.names
    classes = names.items() if isinstance(names, dict) else enumerate(names)
    ic_ids = [int(i) for i, name in classes if str(name).strip().lower() == 'ic']
    if not ic_ids:
        raise ValueError(f'Das Modell hat keine Klasse ic: {names}')
    return model, ic_ids


def detect_housings(image: Image.Image, model, ic_ids: list[int], config: Config) -> list[Candidate]:
    width, height = image.size
    # Ganzbild liefert Kontext; überlappende Kacheln erhalten kleine Gehäuse.
    windows = {(0, 0, width, height)}
    for top in tile_starts(height, config.tile_size, config.tile_overlap):
        for left in tile_starts(width, config.tile_size, config.tile_overlap):
            windows.add((left, top, min(width, left + config.tile_size),
                         min(height, top + config.tile_size)))
    candidates = []
    for left, top, right, bottom in sorted(windows):
        results = model.predict(
            source=image.crop((left, top, right, bottom)),  # PIL: RGB, kein BGR-Tausch
            classes=ic_ids, conf=config.confidence, iou=config.iou_threshold,
            imgsz=config.image_size, verbose=False,
        )
        for result in results:
            if result.boxes is None:
                continue
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = box
                # An inneren Kachelgrenzen abgeschnittene Boxen nicht exportieren.
                # Überlappende Kacheln bzw. Ganzbild liefern vollständige Kandidaten.
                if ((left > 0 and x1 <= 2) or (top > 0 and y1 <= 2)
                        or (right < width and x2 >= right - left - 2)
                        or (bottom < height and y2 >= bottom - top - 2)):
                    continue
                bounds = (max(0, left + x1), max(0, top + y1),
                          min(width, left + x2), min(height, top + y2))
                if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                    candidates.append(Candidate(bounds, confidence))
    return suppress_duplicates(candidates, config.iou_threshold)


def process_image(path: Path, model, ic_ids: list[int], config: Config,
                  show_preview: bool = False) -> None:
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
    # RGB-Werte unter transparenten Pixeln können noch Tisch/Schatten enthalten.
    # Für das Modell explizit weiss hinterlegen, beim Export Alpha erhalten.
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, rgba).convert('RGB')
    candidates = detect_housings(image, model, ic_ids, config)
    # Pro Bild eigene Ausgabe, damit wiederholte Läufe keine alten Gehäuse übriglassen.
    folder = config.output_path / path.name
    folder.mkdir(parents=True, exist_ok=True)
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    records = []
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.box
        padding = config.crop_padding
        crop_box = (max(0, math.floor(x1) - padding), max(0, math.floor(y1) - padding),
                    min(image.width, math.ceil(x2) + padding),
                    min(image.height, math.ceil(y2) + padding))
        filename = f'gehaeuse_{index:03d}.png'
        rgba.crop(crop_box).save(folder / filename)
        draw.rectangle(candidate.box, outline='red', width=4)
        draw.text((x1, max(0, y1 - 14)), f'{index}: IC {candidate.confidence:.2f}', fill='red')
        records.append(dict(file=filename, confidence=candidate.confidence,
                            detection_box=candidate.box, crop_box=crop_box))
    overview.save(folder / 'uebersicht.jpg')
    (folder / 'erkennung.json').write_text(json.dumps(
        dict(source=str(path), model='pcb-yolov8s' if config.model_path is None
             else str(config.model_path), confidence=config.confidence,
             detections=records), indent=2), encoding='utf-8')
    current = {r['file'] for r in records}
    for old in folder.glob('gehaeuse_[0-9][0-9][0-9].png'):
        if old.name not in current:
            old.unlink()
    message = f'{path.name}: {len(candidates)} IC-Gehäuse → {folder}'
    if len(candidates) != config.expected_count:
        message += f' (erwartet: {config.expected_count}; bitte Übersicht prüfen)'
    print(message)
    if show_preview:
        overview.show()


def main(config: Config | None = None) -> None:
    config = config or Config()
    if not config.input_path.is_dir():
        raise FileNotFoundError(f'Bildordner nicht gefunden: {config.input_path}')
    images = sorted(p for p in config.input_path.iterdir()
                    if p.is_file() and p.name.endswith('_freigestellt.png'))
    if not images:
        raise FileNotFoundError(f'Keine *_freigestellt.png in {config.input_path}; zuerst extract_platinen.py starten.')
    if config.quick_test:
        images = images[:1]
    model, ic_ids = load_model(config)
    failures = 0
    for index, path in enumerate(images):
        try:
            process_image(path, model, ic_ids, config, config.show_preview and index == 0)
        except Exception as error:
            failures += 1
            print(f'FEHLER bei {path.name}: {error}')
    if failures:
        raise SystemExit(f'{failures} von {len(images)} Bildern fehlgeschlagen.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick-test', action='store_true', help='Nur erstes Bild verarbeiten')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--model', type=Path, help='Eigene PCB-Modellgewichte statt Download')
    parser.add_argument('--input', type=Path, default=ROOT / 'Ausschnitte')
    parser.add_argument('--output', type=Path, default=ROOT / 'Gehaeuse_YOLO')
    args = parser.parse_args()
    main(Config(input_path=args.input, output_path=args.output, model_path=args.model,
                confidence=args.confidence, quick_test=args.quick_test,
                show_preview=not args.no_preview))
