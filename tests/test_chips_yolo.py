import unittest
from pathlib import Path
import tempfile
from PIL import Image
from Chips_analysis import Candidate, Config, tile_starts, suppress_duplicates, detect_housings, process_image


class Tensor:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return self.values


class FakeModel:
    def predict(self, source, **kwargs):
        assert source.mode == 'RGB'
        assert kwargs['classes'] == [10]
        boxes = type('Boxes', (), {'xyxy': Tensor([[10.5, 20, 60, 70]]),
                                  'conf': Tensor([0.9])})()
        return [type('Result', (), {'boxes': boxes})()]


class DetectionTests(unittest.TestCase):
    def test_tiles_cover_last_edge(self):
        self.assertEqual(tile_starts(200, 100, .25), [0, 75, 100])
        self.assertEqual(tile_starts(50, 100, .25), [0])

    def test_nms_keeps_distinct_ics(self):
        a = Candidate((10, 10, 50, 50), .9)
        duplicate = Candidate((11, 11, 51, 51), .8)
        other = Candidate((70, 10, 100, 50), .7)
        self.assertEqual(suppress_duplicates([duplicate, other, a], .45), [a, other])

    def test_original_coordinates_and_output_cleanup(self):
        image = Image.new('RGB', (100, 100), 'green')
        cfg = Config(tile_size=200)
        result = detect_housings(image, FakeModel(), [10], cfg)
        self.assertEqual(result[0].box, (10.5, 20, 60, 70))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'board.png'
            rgba = image.convert('RGBA')
            rgba.putpixel((0, 5), (90, 50, 10, 0))
            rgba.save(source)
            cfg = Config(output_path=root / 'out', tile_size=200, show_preview=False)
            folder = cfg.output_path / source.name
            folder.mkdir(parents=True)
            (folder / 'gehaeuse_009.png').write_bytes(b'old')
            process_image(source, FakeModel(), [10], cfg)
            self.assertFalse((folder / 'gehaeuse_009.png').exists())
            with Image.open(folder / 'gehaeuse_000.png') as crop:
                self.assertEqual(crop.size, (75, 80))
                self.assertEqual(crop.mode, 'RGBA')
                self.assertEqual(crop.getpixel((0, 0))[3], 0)
            self.assertTrue((folder / 'erkennung.json').exists())


if __name__ == '__main__':
    unittest.main()
