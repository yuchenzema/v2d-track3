"""Keep public reference selection separate from the source-only tracker."""
import copy
import math


HIGHER_IS_BETTER = ('AUC', 'SP-SR', 'MP-SR')
LOWER_IS_BETTER = ('MPPE_cm', 'RPE_cm')


def retained_metrics(attempt):
    return attempt.get('best_reference_metrics', attempt.get('refined_metrics', {}))


def maintains_reference(candidate, previous):
    """Require every measured metric of the actual retained reference to hold."""
    new, old = retained_metrics(candidate), retained_metrics(previous)
    for key in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        values = new.get(key), old.get(key)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return False
        if key in HIGHER_IS_BETTER and values[0] < values[1] - 1e-9:
            return False
        if key in LOWER_IS_BETTER and values[0] > values[1] + 1e-9:
            return False
    return True


def select_attempt(previous, candidate):
    """Never replace a better retained trajectory with a rejected refinement."""
    candidate = copy.deepcopy(candidate)
    if not previous:
        return candidate
    previous = copy.deepcopy(previous)
    candidate['adopted_over_previous'] = maintains_reference(candidate, previous)
    if candidate['adopted_over_previous']:
        candidate['previous_attempt'] = previous
        return candidate
    candidate['selection_reason'] = 'Reference regression rejected; previous retained trajectory and metrics preserved'
    previous.setdefault('rejected_refinements', []).append(candidate)
    previous['repair_round'] = candidate['repair_round']
    return previous


def latest_cad_attempt(result):
    """Return the newest inspected CAD, independently of the selected reference."""
    attempts = [result]
    for rejected in result.get('rejected_refinements', []):
        attempts.append(latest_cad_attempt(rejected))
    # A retained entry also carries the retry counter. A rejected attempt with
    # the same round is newer than that retained entry.
    return max(reversed(attempts), key=lambda row: row.get('repair_round', 0))
