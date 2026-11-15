IC-Gehäuse erkennen
==================

Die Erkennung verwendet das spezialisierte Modell
https://huggingface.co/Arshia82sbn/pcb-yolov8s-detection
(Revision 7eb710b, best.pt). Es läuft lokal. Beim ersten Start ist Internet
für den Download nötig. Eigene Gewichte lassen sich mit --model angeben.

    source .venv/bin/activate
    python -m pip install -r requirements-chips.txt
    python Chips_analysis.py --quick-test

Alle Bilder:

    python Chips_analysis.py --no-preview

Eingabe: Ausschnitte/*_freigestellt.png. Zuerst python extract_platinen.py
ausführen. Transparente Bildbereiche werden für das Modell weiss hinterlegt;
exportierte Gehäuse behalten die Transparenz. *_crop.png wird nicht doppelt verarbeitet.
Ausgabe: Gehaeuse_YOLO/<Originaldateiname>/ mit PNG-Ausschnitten,
uebersicht.jpg und erkennung.json (Koordinaten und Konfidenz).
Wiederholte Läufe ersetzen Ergebnisse desselben Bildes in diesem Ordner.

Das Modell prüft Ganzbild und überlappende Kacheln und entfernt doppelte
Treffer. Nur die Modellklasse ic wird gespeichert. Die erwartete Anzahl 7
ist eine Warnschwelle, kein Auswahlzwang. Die Übersichten bitte kontrollieren:
Die Eignung eines fremdtrainierten Modells für diese Fotos muss geprüft werden.

Bei fehlenden Treffern zunächst die Übersicht ansehen, dann gegebenenfalls:

    python Chips_analysis.py --quick-test --confidence 0.15

Ein niedrigerer Wert kann zusätzliche Fehlalarme erzeugen. Kachelgrösse,
Überlappung und Ausschnittrand stehen in Config in Chips_analysis.py.

Tests ohne Modelldownload:

    python -m unittest discover -s tests -v
