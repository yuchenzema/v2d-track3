import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).parent/'video_to_data/reconstruction/modules/v2d_sam2/lib/bidirectional_prompt_plan.py'
spec=importlib.util.spec_from_file_location('sam2_direction_plan',path);plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)


class BidirectionalPlanTests(unittest.TestCase):
    def test_late_object_prompt_covers_earlier_frames(self):
        passes=plan.propagation_passes([171]);self.assertEqual(passes,[(True,171),(False,171)])
        covered=set()
        for reverse,start in passes:covered.update(range(start,-1,-1) if reverse else range(start,343))
        self.assertEqual(covered,set(range(343)))

    def test_other_object_early_prompt_does_not_erase_late_object_reverse_pass(self):
        self.assertEqual(plan.propagation_passes([0]),[(False,0)])
        self.assertEqual(plan.propagation_passes([171])[0],(True,171))

    def test_nearest_supported_direction_wins_between_prompts(self):
        self.assertLess(plan.prompt_distance(170,[0,171],True),plan.prompt_distance(170,[0,171],False))
        self.assertLess(plan.prompt_distance(5,[0,171],False),plan.prompt_distance(5,[0,171],True))

    def test_invalid_prompt_schedule_is_rejected(self):
        for frames in [[],[-1]]:
            with self.assertRaises(ValueError):plan.propagation_passes(frames)


if __name__=='__main__':unittest.main()
