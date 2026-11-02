import unittest

from public_validation_pipeline import public_validation_order


class PublicValidationCoverageOrderTests(unittest.TestCase):
    def test_two_unattempted_public_cases_precede_retries_without_changing_roster(self):
        episodes=[43,42,11,24,23,26,0,2,18,21,22,31,12,13,27,39,40,41,44,45]
        results={str(ep):dict(source_revision='') for ep in episodes if ep not in [44,45]}
        result=public_validation_order(episodes,results,{})
        self.assertEqual(result,[44,45]+episodes[:-2])
        self.assertEqual(len(result),20)
        self.assertEqual(set(result),set(episodes))

    def test_fresh_source_revision_precedes_previously_attempted_cases(self):
        results={'24':dict(source_revision='_review9'),'11':dict(source_revision='')}
        self.assertEqual(public_validation_order([24,11],results,{'24':'_review9','11':'_cups1'}),[11,24])

    def test_existing_repair_priority_is_preserved_when_coverage_is_complete(self):
        episodes=[43,42,11]
        results={str(ep):dict(source_revision='') for ep in episodes}
        self.assertEqual(public_validation_order(episodes,results,{}),episodes)

    def test_results_outside_the_authorised_roster_do_not_expand_it(self):
        self.assertEqual(public_validation_order([0,2],{'0':dict(source_revision=''),'99':dict(source_revision='')},{}),[2,0])
