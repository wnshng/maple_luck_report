from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.analytics.summary_events import build_analysis_summary_properties, build_analysis_summary_signature
from src.config import is_local_state_persistence_enabled


class PrivacyGuardsTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_local_state_persistence_disabled_by_default(self) -> None:
        self.assertFalse(is_local_state_persistence_enabled())

    @patch.dict(os.environ, {"ENABLE_LOCAL_STATE_PERSISTENCE": "true", "STREAMLIT_RUNTIME": "streamlit_cloud"}, clear=True)
    def test_local_state_persistence_disabled_on_streamlit_cloud(self) -> None:
        self.assertFalse(is_local_state_persistence_enabled())

    def test_analysis_summary_properties_only_use_aggregates(self) -> None:
        context = {
            "controls": {"start_date": date(2024, 5, 11), "end_date": date(2026, 5, 10)},
            "selected_character": {
                "character_name": "비공개",
                "ocid": "secret-ocid",
                "character_class": "엔젤릭버스터",
                "world_name": "스카니아",
                "character_level": 286,
            },
            "cube_df": pd.DataFrame({"dummy": [1, 2]}),
            "potential_df": pd.DataFrame({"dummy": [1]}),
            "starforce_df": pd.DataFrame({"is_destroyed": [False, True, False]}),
            "effective_df": pd.DataFrame({"is_grade_up": [True, False, False], "is_major_option": [True, True, False], "is_effective_option": [True, False, False]}),
            "cube_summary": {"major_rate": 2 / 3, "effective_rate": 1 / 3},
            "star_summary": {"success_rate": 0.5},
            "cube_by_day_of_month": pd.DataFrame({"day_of_month": [13, 9], "attempts": [12, 8], "effective_option_rate": [0.4, 0.9]}),
            "cube_by_hour": pd.DataFrame({"hour_label": ["20시", "9시"], "attempts": [12, 5], "effective_option_rate": [0.5, 0.7]}),
            "cube_by_weekday": pd.DataFrame({"weekday_kr": ["금요일"], "attempts": [14], "effective_option_rate": [0.45]}),
            "cube_by_type": pd.DataFrame({"cube_type": ["블랙 큐브"], "attempts": [15], "effective_option_rate": [0.48]}),
            "star_by_day_of_month": pd.DataFrame({"day_of_month": [18], "attempts": [11], "success_rate": [0.6]}),
            "star_by_hour": pd.DataFrame({"hour_label": ["21시"], "attempts": [13], "success_rate": [0.7]}),
            "star_by_weekday": pd.DataFrame({"weekday_kr": ["토요일"], "attempts": [10], "success_rate": [0.65]}),
            "star_by_transition": pd.DataFrame({"transition_label": ["16→17성"], "attempts": [12], "success_rate": [0.62]}),
        }

        properties = build_analysis_summary_properties(context)
        assert properties is not None

        self.assertEqual(properties["date_range_days"], 730)
        self.assertEqual(properties["total_record_count"], 6)
        self.assertEqual(properties["cube_attempts"], 2)
        self.assertEqual(properties["potential_attempts"], 1)
        self.assertEqual(properties["starforce_attempts"], 3)
        self.assertEqual(properties["best_cube_day_of_month"], "13일")
        self.assertEqual(properties["best_cube_hour"], "20시")
        self.assertEqual(properties["best_cube_weekday"], "금요일")
        self.assertEqual(properties["best_cube_type"], "블랙 큐브")
        self.assertEqual(properties["best_starforce_transition"], "16→17성")
        self.assertEqual(properties["character_class"], "엔젤릭버스터")
        self.assertEqual(properties["world_name"], "스카니아")
        self.assertEqual(properties["character_level_bucket"], "280-289")
        self.assertNotIn("character_name", properties)
        self.assertNotIn("ocid", properties)

    def test_analysis_summary_signature_uses_aggregate_identity(self) -> None:
        signature = build_analysis_summary_signature(
            character_class="엔젤릭버스터",
            world_name="스카니아",
            last_query_range="2024-05-11 ~ 2026-05-10",
            cube_attempts=120,
            potential_attempts=80,
            starforce_attempts=30,
        )
        self.assertIn("엔젤릭버스터", signature)
        self.assertIn("스카니아", signature)
        self.assertNotIn("비공개", signature)
        self.assertNotIn("secret-ocid", signature)


if __name__ == "__main__":
    unittest.main()
