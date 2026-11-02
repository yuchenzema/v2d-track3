import unittest
from public_validation_pipeline import distinct_board_pair


def box(x0,y0,x1,y1,confidence=.9):
    return dict(box=dict(x0=x0,y0=y0,x1=x1,y1=y1),confidence=confidence)


class DistinctBoardTests(unittest.TestCase):
    def test_shifted_overlapping_boxes_are_not_two_boards(self):
        self.assertIsNone(distinct_board_pair([box(394,380,718,679),box(394,279,721,679)]))

    def test_a_board_nested_in_a_table_box_is_rejected_even_with_small_iou(self):
        self.assertIsNone(distinct_board_pair([box(0,0,1000,1000),box(300,300,400,400)]))

    def test_two_separate_boards_are_selected_and_ordered(self):
        left=box(100,100,180,250,.8);right=box(280,100,360,250,.85)
        self.assertEqual(distinct_board_pair([right,left,box(0,0,1000,1000,.99)]),[left,right])


if __name__=='__main__':unittest.main()
