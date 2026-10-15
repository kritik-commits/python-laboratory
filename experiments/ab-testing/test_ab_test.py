import unittest

from ab_test import two_proportion_z_test


class TestTwoProportionZTest(unittest.TestCase):

    def test_conversion_rates(self):
        result = two_proportion_z_test(120, 1000, 145, 1000)

        self.assertAlmostEqual(result["control_rate"], 0.12)
        self.assertAlmostEqual(result["variant_rate"], 0.145)

    def test_variant_has_positive_lift(self):
        result = two_proportion_z_test(120, 1000, 145, 1000)

        self.assertGreater(result["absolute_lift"], 0)

    def test_invalid_visitor_count(self):
        with self.assertRaises(ValueError):
            two_proportion_z_test(10, 0, 15, 100)

    def test_invalid_conversion_count(self):
        with self.assertRaises(ValueError):
            two_proportion_z_test(101, 100, 15, 100)


if __name__ == "__main__":
    unittest.main()