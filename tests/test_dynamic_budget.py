"""Unit tests for calculate_dynamic_thinking_budget in scripts/gemini_client.py."""

import unittest

from scripts.gemini_client import calculate_dynamic_thinking_budget


class TestDynamicThinkingBudget(unittest.TestCase):
    def test_short_video_budget(self):
        """Verify budget for a short video (15 sentences, 90 seconds) is between 1024 and 1500."""
        budget = calculate_dynamic_thinking_budget(num_sentences=15, video_duration_seconds=90.0)
        self.assertGreaterEqual(budget, 1024)
        self.assertLessEqual(budget, 1500)
        self.assertEqual(budget, 1325)

    def test_medium_video_budget(self):
        """Verify budget for an 8.4-minute video (62 sentences, 504 seconds) is between 3800 and 4200."""
        budget = calculate_dynamic_thinking_budget(num_sentences=62, video_duration_seconds=504.0)
        self.assertGreaterEqual(budget, 3800)
        self.assertLessEqual(budget, 4200)
        self.assertEqual(budget, 3910)

    def test_long_video_budget_clamped_at_ceiling(self):
        """Verify budget for a long video (200 sentences, 1800 seconds) is clamped at or below 5120."""
        budget_1800s = calculate_dynamic_thinking_budget(num_sentences=200, video_duration_seconds=1800.0)
        self.assertLessEqual(budget_1800s, 5120)
        self.assertEqual(budget_1800s, 4830)

        budget_600s = calculate_dynamic_thinking_budget(num_sentences=200, video_duration_seconds=600.0)
        self.assertEqual(budget_600s, 5120)

    def test_very_small_input_clamped_at_floor(self):
        """Verify minimum budget floor is 1024."""
        budget = calculate_dynamic_thinking_budget(num_sentences=2, video_duration_seconds=15.0)
        self.assertEqual(budget, 1024)


if __name__ == "__main__":
    unittest.main()
