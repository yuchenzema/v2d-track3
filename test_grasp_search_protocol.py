import unittest
from grasp_search_protocol import search_seed


class SearchSeedTests(unittest.TestCase):
    def test_replaying_the_same_attempt_is_reproducible(self):
        self.assertEqual(search_seed(20,'attempt-a'),search_seed(20,'attempt-a'))

    def test_retry_labels_do_not_replay_the_same_random_search(self):
        seeds={search_seed(20,f'physicalhold20v4-blue-cup-reference-{attempt}') for attempt in range(100)}
        self.assertEqual(len(seeds),100)
        self.assertNotEqual(search_seed(20,'attempt-a'),search_seed(30,'attempt-a'))
