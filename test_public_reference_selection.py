import unittest

from public_reference_selection import latest_cad_attempt, select_attempt


def attempt(round, auc, retained_auc=None, rpe=1., mp=1.):
    metrics = {'AUC': auc, 'SP-SR': 1., 'MP-SR': mp, 'RPE_cm': rpe, 'MPPE_cm': 1.}
    return dict(repair_round=round, refined_metrics=metrics,
                best_reference_metrics={**metrics, 'AUC': auc if retained_auc is None else retained_auc},
                refined_run=f'cad_{round}', selected_reference_prediction=f'reference_{round}')


class ReferenceSelectionTests(unittest.TestCase):
    def test_high_raw_auc_does_not_replace_better_retained_trajectory(self):
        old=attempt(0,.85); new=attempt(1,.95,retained_auc=.80)
        selected=select_attempt(old,new)
        self.assertEqual(selected['best_reference_metrics']['AUC'],.85)
        self.assertEqual(selected['selected_reference_prediction'],'reference_0')
        self.assertEqual(latest_cad_attempt(selected)['refined_run'],'cad_1')
        self.assertEqual(old['repair_round'],0) # caller input/history not mutated

    def test_raw_auc_cannot_hide_retained_reference_improvement(self):
        old=attempt(0,.99,retained_auc=.80); new=attempt(1,.90)
        self.assertEqual(select_attempt(old,new)['selected_reference_prediction'],'reference_1')

    def test_position_or_success_regression_rejects_higher_auc(self):
        for kwargs in [dict(rpe=2.),dict(mp=0.)]:
            with self.subTest(kwargs=kwargs):
                selected=select_attempt(attempt(0,.85),attempt(1,.95,**kwargs))
                self.assertEqual(selected['selected_reference_prediction'],'reference_0')

    def test_nonfinite_metrics_do_not_replace_a_valid_reference(self):
        self.assertEqual(select_attempt(attempt(0,.85),attempt(1,.95,rpe=float('nan')))['selected_reference_prediction'],'reference_0')

    def test_latest_rejected_cad_remains_visible_across_retries(self):
        selected=select_attempt(attempt(0,.85),attempt(1,.75))
        selected=select_attempt(selected,attempt(2,.70))
        self.assertEqual(selected['selected_reference_prediction'],'reference_0')
        self.assertEqual(latest_cad_attempt(selected)['refined_run'],'cad_2')
        self.assertEqual(len(selected['rejected_refinements']),2)
