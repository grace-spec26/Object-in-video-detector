import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

from refinement_helpers import (  # noqa: E402
    append_refinement_point,
    clear_all_refinement_points,
    count_frame_points,
    empty_frame_points,
    merge_frame_point_lists,
    pop_refinement_point,
    remove_nearest_refinement_point,
)


class RefinementHelpersTest(unittest.TestCase):
    def test_append_refinement_point_adds_labeled_point_without_mutating_input(self):
        original = empty_frame_points(3)

        updated = append_refinement_point(original, frame_index=1, x=10.5, y=20.25, label=0)

        self.assertEqual(original, [[], [], []])
        self.assertEqual(updated[1], [(10.5, 20.25, 1, 0)])

    def test_remove_nearest_refinement_point_removes_only_close_point_on_selected_frame(self):
        points = [
            [(100.0, 100.0, 0, 1)],
            [(10.0, 10.0, 1, 1), (30.0, 30.0, 1, 0)],
        ]

        updated, removed = remove_nearest_refinement_point(
            points,
            frame_index=1,
            x=11.0,
            y=10.0,
            max_distance=5.0,
        )

        self.assertTrue(removed)
        self.assertEqual(updated[0], points[0])
        self.assertEqual(updated[1], [(30.0, 30.0, 1, 0)])

    def test_remove_nearest_refinement_point_keeps_far_points(self):
        points = [[(100.0, 100.0, 0, 1)]]

        updated, removed = remove_nearest_refinement_point(
            points,
            frame_index=0,
            x=10.0,
            y=10.0,
            max_distance=5.0,
        )

        self.assertFalse(removed)
        self.assertEqual(updated, points)

    def test_pop_refinement_point_removes_last_point_on_selected_frame(self):
        points = [[(1.0, 1.0, 0, 1), (2.0, 2.0, 0, 0)]]

        updated, removed = pop_refinement_point(points, frame_index=0)

        self.assertTrue(removed)
        self.assertEqual(updated, [[(1.0, 1.0, 0, 1)]])

    def test_merge_frame_point_lists_combines_original_and_refinement_prompts(self):
        base = [[(1.0, 2.0, 0, 1)], []]
        refinements = [[], [(3.0, 4.0, 1, 0)]]

        merged = merge_frame_point_lists(base, refinements)

        self.assertEqual(merged, [[(1.0, 2.0, 0, 1)], [(3.0, 4.0, 1, 0)]])
        self.assertEqual(count_frame_points(merged), 2)

    def test_clear_all_refinement_points_preserves_frame_count(self):
        points = [[(1.0, 1.0, 0, 1)], [], [(2.0, 2.0, 2, 0)]]

        self.assertEqual(clear_all_refinement_points(points), [[], [], []])


if __name__ == "__main__":
    unittest.main()
