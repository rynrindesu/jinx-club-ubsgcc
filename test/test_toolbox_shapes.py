import base64
from io import BytesIO
from math import cos, radians, sin
import unittest

from PIL import Image, ImageDraw

from app.phase1.toolbox.shapes import classify_shape_image


class ShapeClassifierTests(unittest.TestCase):
    def classify(self, image: Image.Image) -> str:
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        return classify_shape_image(base64.b64encode(encoded.getvalue()).decode())

    def image_with_polygon(self, points: list[tuple[float, float]]) -> Image.Image:
        image = Image.new("RGBA", (160, 160), "white")
        ImageDraw.Draw(image).polygon(points, fill="black")
        return image

    def rotated(self, points: list[tuple[float, float]], angle: float) -> list[tuple[float, float]]:
        angle_radians = radians(angle)
        return [
            (
                80 + x * cos(angle_radians) - y * sin(angle_radians),
                80 + x * sin(angle_radians) + y * cos(angle_radians),
            )
            for x, y in points
        ]

    def test_classifies_rotated_rectangle_that_old_corner_count_misclassifies(self):
        image = self.image_with_polygon(
            self.rotated([(-38, -24), (38, -24), (38, 24), (-38, 24)], 4.36)
        )

        self.assertEqual(self.classify(image), "rectangle")

    def test_classifies_rotated_triangle(self):
        image = self.image_with_polygon(
            self.rotated([(0, -25), (-34, 20), (14, 40)], 75.41)
        )

        self.assertEqual(self.classify(image), "triangle")

    def test_classifies_circle(self):
        image = Image.new("RGBA", (160, 160), "white")
        ImageDraw.Draw(image).ellipse((35, 35, 125, 125), fill="black")

        self.assertEqual(self.classify(image), "circle")

    def test_ignores_a_small_disconnected_foreground_speck(self):
        image = self.image_with_polygon(
            [(40, 30), (120, 30), (120, 110), (40, 110)]
        )
        ImageDraw.Draw(image).rectangle((145, 145, 146, 146), fill="black")

        self.assertEqual(self.classify(image), "rectangle")


if __name__ == "__main__":
    unittest.main()
