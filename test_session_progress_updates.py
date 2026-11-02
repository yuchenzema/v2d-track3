import unittest
from session_progress_updates import delivery_due


class UpdateDeliveryTests(unittest.TestCase):
    def test_stopped_or_changed_scope_cannot_keep_sending_old_prompts(self):
        config={'enabled':True,'episodes':[19,20]}
        self.assertFalse(delivery_due(config,{'status':'stopped','episodes':[19,20]},1000)[0])
        self.assertFalse(delivery_due(config,{'status':'active','episodes':[32]},1000)[0])

    def test_minimum_ten_minute_spacing(self):
        config={'enabled':True,'episodes':[19,20],'last_queued_timestamp':1000}
        scope={'status':'active','episodes':[19,20]}
        self.assertFalse(delivery_due(config,scope,1599)[0])
        self.assertTrue(delivery_due(config,scope,1600)[0])

    def test_final_report_is_only_queued_once(self):
        config={'enabled':True,'episodes':[19,20],'final_report_queued':True}
        self.assertFalse(delivery_due(config,{'status':'active','episodes':[19,20]},2000)[0])
