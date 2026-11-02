import unittest
import numpy as np
from global_hand_identity import decide_observations


class HandIdentityObservationTests(unittest.TestCase):
    def fixture(self,ambiguous_strong=False):
        rows=[];votes=np.zeros((2,2));counts=np.zeros(2,int)
        for frame in range(240):
            for track in range(2):
                strong=frame<10 or frame>=230
                side=(1-track if ambiguous_strong and frame>=230 else track)
                score=.99 if strong else .501
                rows.append(dict(frame=frame,track=track,label=['Left','Right'][side],confidence=score))
                votes[track,side]+=score;votes[track,1-side]+=1-score;counts[track]+=1
        return votes,counts,rows

    def test_strong_separated_anchors_survive_many_uncertain_grasp_labels(self):
        permutation,confidence,evidence=decide_observations(*self.fixture())
        self.assertEqual(permutation,[0,1]);self.assertGreater(min(confidence),.98)
        self.assertEqual(evidence['minimum_observations_per_track'],10)
        self.assertEqual(evidence['minimum_supported_windows_per_track'],2)

    def test_conflicting_strong_identity_evidence_remains_blocked(self):
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):
            decide_observations(*self.fixture(ambiguous_strong=True))

    def test_one_strong_burst_cannot_certify_a_whole_track(self):
        votes,counts,rows=self.fixture()
        for row in rows:
            if row['frame']>=230:row['frame']=row['frame']-230
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):
            decide_observations(votes,counts,rows)


if __name__=='__main__':unittest.main()
