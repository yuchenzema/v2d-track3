import unittest
from quality_progress_counts import counts


class QualityAccountingTests(unittest.TestCase):
    def test_repair_keeps_completed_work_and_the_previous_selected_model(self):
        progress=dict(episodes={'16':dict(status='running',run='new'), '28':dict(status='running',run='first')},
            followup_history=[dict(episode=16,previous=dict(status='completed',run='old',selected=True))])
        result=counts(progress,dict(episodes={'16':dict(run='old')}))
        self.assertEqual(result,dict(reviewed=[16],selected=[16],followups=[16]))
        progress['episodes']['16'].update(status='completed',selected=False)
        result=counts(progress,dict(episodes={'16':dict(run='old')}))
        self.assertEqual(result,dict(reviewed=[16],selected=[16],followups=[]))
