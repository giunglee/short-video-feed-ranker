"""Observable EDA arithmetic and support rules on small public fixtures."""

import unittest

import numpy as np
import pandas as pd

from feed_ranker.experiments.eda_review import cramers_v, cross_statistics, scan_crosses


class EdaReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        # One repeated user spans two periods; counts are events, not users.
        self.cells = pd.DataFrame(
            [
                {"user_id": u, "a": a, "b": b, "n": n, "total": n * mean, "period": period}
                for period in ["through July 20", "after July 20"]
                for u, a, b, n, mean in [
                    (1, "x", "p", 2, 1.0),
                    (2, "x", "q", 1, 2.0),
                    (3, "y", "p", 1, 3.0),
                    (4, "y", "q", 2, 4.0),
                ]
            ]
        )

    def test_aggregates_match_expanded_event_means(self) -> None:
        stats = cross_statistics(self.cells, "a", "b")
        overall = 2.5
        cell = stats.loc[stats.a.eq("x") & stats.b.eq("p")].iloc[0]
        self.assertEqual(cell.events, 4)
        self.assertEqual(cell.users, 1)
        self.assertAlmostEqual(cell["mean"], 1.0)
        mean_a = 4 / 3
        mean_b = 5 / 3
        self.assertAlmostEqual(cell.interaction_lift, overall / (mean_a * mean_b))
        self.assertAlmostEqual(cell.additive_departure, 1 - mean_a - mean_b + overall)

    def test_user_support_prevents_repeat_events_passing_screen(self) -> None:
        result = scan_crosses(self.cells, ["a", "b"], min_events=1, min_users=2)
        self.assertFalse(result.supported.any())
        self.assertFalse(result.period_supported.any())
        result = scan_crosses(self.cells, ["a", "b"], min_events=1, min_users=1)
        self.assertTrue(result.period_supported.all())

    def test_redundancy_and_zero_reference(self) -> None:
        a = pd.Series(["x", "x", "y", "y"] * 10)
        self.assertAlmostEqual(cramers_v(a, a), 1.0)
        self.assertEqual(cramers_v(a, pd.Series(["z"] * len(a))), 0.0)
        self.cells["total"] = 0.0
        result = cross_statistics(self.cells, "a", "b")
        self.assertTrue(result.interaction_lift.isna().all())
        self.assertTrue(np.isfinite(result.additive_departure).all())


class DailyDurationTest(unittest.TestCase):
    def test_event_counts_duration_mix_and_undated_exclusion(self) -> None:
        from feed_ranker.experiments.eda_review import daily_duration_summary

        frame = pd.DataFrame(
            {
                "date": [20200705, 20200705, 20200706, np.nan],
                "user_id": [1, 1, 2, 3],
                "video_id": [1, 1, 2, 3],
                "watch_ratio": [1.0, 3.0, 4.0, 9.0],
                "video_duration": [1000.0, 1000.0, 8000.0, 8000.0],
            }
        )
        summary, shares = daily_duration_summary({"big train": frame})
        self.assertEqual(summary.events.sum(), 3)
        first = summary.loc[summary.date.eq(20200705)].iloc[0]
        self.assertEqual(first.events, 2)
        self.assertEqual(first.users, 1)
        self.assertEqual(first.label_mean, 2.0)
        self.assertEqual(first.duration_mean_ms, 1000.0)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0)
        # Reference uses all train bucket labels, including the undated train row.
        self.assertEqual(summary.duration_mix_reference.tolist(), [2.0, 6.5])

    def test_unobserved_calendar_day_has_zero_count_and_undefined_mean(self) -> None:
        from feed_ranker.experiments.eda_review import daily_duration_summary

        frame = pd.DataFrame(
            {
                "date": [20200705, 20200707],
                "user_id": [1, 2],
                "video_id": [1, 2],
                "watch_ratio": [1.0, 2.0],
                "video_duration": [1000.0, 8000.0],
            }
        )
        summary, shares = daily_duration_summary({"big train": frame})
        gap = summary.loc[summary.date.eq(20200706)].iloc[0]
        self.assertEqual(gap.events, 0)
        self.assertTrue(pd.isna(gap.label_mean))
        self.assertTrue(pd.isna(gap.duration_mix_reference))
        self.assertTrue(shares.loc[20200706].isna().all())


class ExtremeEventsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "video_id": [1, 1, 2, 3],
                "date": [20200705, 20200705, 20200707, np.nan],
                "timestamp": [100, 100, 200, np.nan],
                "watch_ratio": [5.0, 5.0, 1.0, 0.0],
                "video_duration": [1000.0, 1000.0, 2000.0, 2000.0],
                "play_duration": [5000.0, 5000.0, 2000.0, 0.0],
            }
        )

    def test_boundary_loss_shares_and_logging_signature(self) -> None:
        from feed_ranker.experiments.eda_extremes import logging_checks, tail_audit

        frames = {"big train": self.frame}
        summary = tail_audit(frames)
        self.assertEqual(summary.events.tolist(), [2, 2])
        self.assertAlmostEqual(summary.absolute_error_share.sum(), 1.0)
        self.assertAlmostEqual(summary.squared_error_share.sum(), 1.0)
        audit = logging_checks(frames).iloc[0]
        self.assertEqual(audit.repeated_event_signature_beyond_first, 1)
        self.assertEqual(audit.missing_timestamp, 1)
        self.assertEqual(audit.ratio_absolute_discrepancy_max, 0.0)

    def test_entity_counts_concentration_and_calendar_gaps(self) -> None:
        from feed_ranker.experiments.eda_extremes import (
            concentration_summary,
            daily_tail_counts,
            entity_statistics,
        )

        users = entity_statistics(self.frame, "user_id")
        self.assertEqual(users.loc[1, "partners"], 1)
        self.assertEqual(users.loc[1, "tail_fraction"], 1.0)
        self.assertEqual(users.loc[2, "tail_fraction"], 0.0)
        summary = concentration_summary({"big train": {"user_id": users}}).iloc[0]
        self.assertEqual(summary.top_1pct_tail_share, 1.0)
        self.assertEqual(summary.same_entities_all_event_share, 0.5)
        days = daily_tail_counts({"big train": self.frame}).set_index("date")
        self.assertEqual(days.events.sum(), 3)
        self.assertEqual(days.loc[20200706, "events"], 0)
        self.assertTrue(pd.isna(days.loc[20200706, "tail_fraction"]))
        with self.assertRaisesRegex(ValueError, "entity must"):
            entity_statistics(self.frame, "date")


class DuplicateSensitivityTest(unittest.TestCase):
    def test_all_labels_deduplicated_but_unknown_timestamps_preserved(self) -> None:
        from feed_ranker.experiments.eda_extremes import duplicate_sensitivity

        frame = pd.DataFrame(
            {
                "user_id": [1] * 6,
                "video_id": [2] * 6,
                "timestamp": [1, 1, 2, 2, np.nan, np.nan],
                "play_duration": [6000, 6000, 1000, 1000, 6000, 6000],
                "video_duration": [1000] * 6,
                "watch_ratio": [6.0, 6.0, 1.0, 1.0, 6.0, 6.0],
            }
        )
        original = frame.copy(deep=True)
        result = duplicate_sensitivity({"big train": frame}).iloc[0]
        self.assertEqual(result.removed_events, 2)
        self.assertEqual(result.removed_tail_events, 1)
        self.assertEqual(result.fraction_tail_removed, 0.25)
        self.assertEqual(result.fraction_common_removed, 0.5)
        self.assertEqual(result.deduplicated_tail_fraction, 0.75)
        pd.testing.assert_frame_equal(frame, original)
