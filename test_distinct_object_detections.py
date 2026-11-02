import unittest

from public_validation_pipeline import validate_distinct_detection_prompts


def detection(body,frame,box):
    return dict(object_id=body,frame_index=frame,box=dict(zip(('x0','y0','x1','y1'),box)))


class DistinctObjectDetectionTests(unittest.TestCase):
    def test_actual_public11_duplicate_cup_detections_require_review(self):
        prompts=[detection(1,68,[467.07318115234375,396.83447265625,570.1990356445312,544.5191040039062]),
                 detection(2,68,[466.75689697265625,396.60064697265625,572.5639038085938,546.2261962890625])]
        with self.assertRaisesRegex(RuntimeError,'source-image prompt review needed'):
            validate_distinct_detection_prompts(prompts)

    def test_separate_visible_cups_are_valid(self):
        validate_distinct_detection_prompts([detection(1,0,[612,420,705,556]),detection(2,0,[450,438,550,584])])

    def test_stacked_cups_can_have_partially_overlapping_boxes(self):
        validate_distinct_detection_prompts([detection(1,136,[441,425,533,510]),detection(2,136,[450,489,537,583])])

    def test_boxes_from_different_source_frames_are_not_compared(self):
        validate_distinct_detection_prompts([detection(1,0,[440,420,540,580]),detection(2,68,[440,420,540,580])])

    def test_point_prompts_do_not_enter_box_validation(self):
        validate_distinct_detection_prompts([dict(object_id=1,frame_index=0,points=[dict(x=1,y=2)]),
                                            dict(object_id=2,frame_index=0,points=[dict(x=3,y=4)])])
