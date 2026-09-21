import unittest

import numpy as np
from PIL import Image, ImageDraw

from image_extraction.descriptors import SpatialDescriptor


class DescriptorTests(unittest.TestCase):
    def scene(self, *, x=30, shape="rectangle", size=(128, 80), offset=0):
        image = Image.new("RGB", size, (180 + offset,) * 3)
        draw = ImageDraw.Draw(image)
        box = (x - 12, 22, x + 12, 58)
        getattr(draw, shape)(box, fill=(60 + offset,) * 3)
        self.addCleanup(image.close)
        return image

    def test_fixed_shape_dtype_unit_norm_and_determinism(self):
        descriptor = SpatialDescriptor()
        image = self.scene()
        before = image.tobytes()
        vector = descriptor.extract(image)
        self.assertEqual(vector.shape, (576,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        np.testing.assert_array_equal(vector, descriptor.extract(image))
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(descriptor.metadata()["dimension"], 576)

    def test_matching_layout_beats_same_shape_at_different_position(self):
        descriptor = SpatialDescriptor()
        query = descriptor.extract(self.scene())
        positive = descriptor.extract(self.scene(shape="ellipse"))
        negative = descriptor.extract(self.scene(x=98))
        self.assertGreater(float(query @ positive), float(query @ negative))

    def test_global_brightness_offset_does_not_change_layout(self):
        descriptor = SpatialDescriptor()
        original = descriptor.extract(self.scene())
        brighter = descriptor.extract(self.scene(offset=20))
        self.assertGreater(float(original @ brighter), 0.999)

    def test_uniform_and_tiny_images_are_finite(self):
        descriptor = SpatialDescriptor()
        for size in ((1, 1), (1, 30), (30, 1), (4, 4), (900, 10)):
            with Image.new("L", size, 127) as image:
                vector = descriptor.extract(image)
                np.testing.assert_array_equal(vector, np.zeros(descriptor.dimension))
        for size in ((1, 30), (30, 1)):
            with Image.new("L", size, 0) as image:
                image.putpixel((0, 0), 255)
                vector = descriptor.extract(image)
                self.assertTrue(np.isfinite(vector).all())
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)

    def test_exif_orientation_matches_displayed_geometry(self):
        descriptor = SpatialDescriptor()
        image = self.scene()
        rotated = image.transpose(Image.Transpose.ROTATE_90)
        self.addCleanup(rotated.close)
        rotated.getexif()[274] = 6
        np.testing.assert_allclose(descriptor.extract(image), descriptor.extract(rotated), atol=1e-6)

    def test_invalid_configuration(self):
        for kwargs in ({"grid_size": 0}, {"max_side": 0}, {"orientation_bins": 1.5},
                       {"blur_radius": -1}, {"blur_radius": float("nan")},
                       {"intensity_weight": 1.1}, {"intensity_weight": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpatialDescriptor(**kwargs)


if __name__ == "__main__":
    unittest.main()
