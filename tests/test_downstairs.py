"""Private invariant and quality audit. No Streamlit downloads or uploads."""
import ast
import itertools
import json
import math
import random
import unittest
from collections import Counter
from datetime import date, datetime, time, timedelta
from math import comb, factorial
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "app.py"

RESULTS = []


class QuietUI:
    def progress(self, *args): return self
    def empty(self): return self
    def info(self, *args): pass
    def write(self, *args): pass


def load_source():
    tree = ast.parse(APP.read_text(encoding="utf-8-sig"))
    ns = dict(globals(), st=QuietUI())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "main"]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(APP), "exec"), ns)
    return ns


class Audit(unittest.TestCase):
    def setUp(self):
        self.ns = load_source()

    def run_case(self, n, courts, rounds, limits, seed=0, skill=False, same=False, count=3, samples=64):
        random.seed(seed)
        genders = ["F" if i % 2 else "M" for i in range(n)]
        skills = [4 + i % 4 for i in range(n)]
        result = self.ns["assign_matches"](
            [f"PRIVATE QA {i}" for i in range(n)], genders, skills, courts,
            total_matches=rounds, downstairs_limits=limits, samples=samples,
            enable_skill=skill, reject_mixed=same, same_sex_matches_count=count,
            start_dt=datetime(2026, 10, 24, 10, 30), match_offset=6,
        )
        html, tm, om, rests, matches, used = result
        down, _ = self.ns["identify_downstairs_courts"](courts[:used])
        counted_rests = np.zeros(n, int)
        counted_down = np.zeros(n, int)
        expected_tm, expected_om = self.ns["initialize_matrices"](n)
        previous_down = None
        for idx, (groups, bench) in enumerate(matches):
            self.assertEqual(len(groups), used)
            self.assertTrue(all(len(g) == 4 for g in groups))
            active = list(itertools.chain.from_iterable(groups))
            self.assertEqual(len(set(active)), len(active))
            self.assertEqual(set(active) & set(bench), set())
            self.assertEqual(sorted(active + bench), list(range(n)))
            self.assertEqual(len(bench), n - 4 * used)
            for p in bench: counted_rests[p] += 1
            current_down = {p for c in down for p in groups[c]}
            if idx % 2: self.assertEqual(current_down, previous_down)
            for p in current_down: counted_down[p] += 1
            previous_down = current_down
            self.ns["update_matrices_for_match"](groups, expected_tm, expected_om)
        self.assertTrue(np.array_equal(tm, expected_tm))
        self.assertTrue(np.array_equal(om, expected_om))
        self.assertTrue(np.array_equal(rests, counted_rests))
        self.assertTrue(all(c <= limit for c, limit in zip(counted_down, limits)))
        self.assertTrue(all(c <= 4 for c in counted_down))
        self.assertEqual(len(matches), rounds)
        self.assertEqual(html.count('class="match-block"'), rounds)
        self.assertEqual(html.count("class='downstairs-badge'"), rounds * len(down))
        per_page = 3 if used <= 4 else 2
        self.assertEqual(html.count("page-break-after: always;"), (rounds - 1) // per_page)
        self.assertIn("Match 7 [10:30]", html)
        RESULTS.append({"n": n, "courts": courts, "rounds": rounds, "limits": limits,
                        "skill": skill, "same": same, "seed": seed,
                        "rests": rests.tolist(), "downstairs": counted_down.tolist()})
        return result

    def test_seeded_property_matrix(self):
        rng = random.Random(20261024)
        accepted = rejected = 0
        while accepted < 70:
            down_count = rng.choice((1, 2))
            up_count = rng.randrange(0, 5)
            n = 4 * (up_count + down_count) + rng.randrange(0, 8)
            rounds = rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 10))
            limits = rng.choices((0, 2, 4), weights=(1, 4, 3), k=n)
            courts = [f"Court {i+1}" for i in range(up_count)] + ["Court 12", "Court 13"][:down_count]
            rng.shuffle(courts)
            try:
                self.ns["validate_downstairs_limits"](list(range(n)), rounds, down_count * 4, limits)
            except ValueError:
                rejected += 1
                continue
            with self.subTest(n=n, courts=courts, rounds=rounds, limits=limits):
                self.run_case(n, courts, rounds, limits, seed=accepted,
                              skill=bool(accepted % 2), same=bool(accepted % 3),
                              count=min(3, rounds), samples=32)
            accepted += 1
        print(f"Property matrix: {accepted} complete sessions; {rejected} infeasible inputs rejected.")

    def test_zero_same_sex_count_means_no_preferred_rounds(self):
        original = self.ns["find_best_match"]
        calls = []
        def record(*args, **kwargs):
            calls.append(args[7])
            return original(*args, **kwargs)
        self.ns["find_best_match"] = record
        self.run_case(16, ["Court 1", "Court 2", "Court 3", "Court 12"], 6, [2] * 16,
                      same=True, count=0)
        self.assertFalse(any(calls), "Zero requested same-sex rounds must not enable every round")

    def test_custom_court_number_is_not_a_suffix_collision(self):
        down, _ = self.ns["identify_downstairs_courts"](["Court 12", "Court 13", "Court 112", "Court 213"])
        self.assertEqual(down, [0, 1])
        self.assertEqual(self.ns["identify_downstairs_courts"](
            ["12", "13", "court12", "Downstairs 13 ", "Court 14"])[0], [0, 1, 2, 3])

    def test_duplicate_courts_are_rejected_before_planning(self):
        for courts in (["Court 12", "12"], ["Court 13", "Downstairs 13"],
                       ["Court 7", " court7 "], ["North", " north "], ["Court 12", "Court 12"]):
            with self.subTest(courts=courts), self.assertRaisesRegex(ValueError, "same court"):
                self.run_case(16, courts, 2, [4] * 16)
        self.ns["validate_unique_courts"](["Court 12", "Court 13", "Court 112", "Court 213"])

    def test_small_pool_has_different_teams_on_second_match(self):
        repeated = []
        for seed in range(18):
            result = self.run_case(4, ["Court 12"], 2, [2] * 4, seed=seed, samples=1)
            configs = [self.ns["canonical_config"](groups[0]) for groups, bench in result[4]]
            if configs[0] == configs[1]: repeated.append(seed)
        self.assertFalse(repeated, f"Same four players, unchanged teams in a two-match stay: seeds {repeated}")

    def test_no_consecutive_identical_games_even_after_all_three_splits_used(self):
        for seed in range(24):
            for n, courts, rounds in ((4, ["Court 12"], 4),
                                      (8, ["Court 12", "Court 13"], 4),
                                      (16, ["Court 1", "Court 2", "Court 12", "Court 13"], 6)):
                result = self.run_case(n, courts, rounds, [4] * n, seed=seed, samples=1)
                configs = [{self.ns["canonical_config"](g) for g in groups} for groups, _ in result[4]]
                self.assertTrue(all(not(a & b) for a, b in zip(configs, configs[1:])), (n, seed))

    def test_same_sex_counts_cover_exactly_requested_number(self):
        for total in range(1, 14):
            for count in range(total + 4):
                indices = self.ns["compute_same_sex_match_indices"](total, count)
                self.assertEqual(len(indices), min(total, count), (total, count))
                self.assertEqual(indices, sorted(set(indices)))
                self.assertTrue(all(0 <= idx < total for idx in indices))

    def test_single_court_skill_split_is_optimal(self):
        tm, om = self.ns["initialize_matrices"](4)
        groups = self.ns["find_best_match"](
            [0, 1, 2, 3], 1, 4, tm, om, 0, samples=1,
            genders=["F"] * 4, skills=[4, 4, 7, 7], enable_skill=True)
        self.assertEqual(sum([4, 4, 7, 7][p] for p in groups[0][:2]), 11)

    def test_four_match_permission_is_not_mandatory_assignment(self):
        result = self.run_case(24, ["Court 1", "Court 2", "Court 3", "Court 4", "Court 12", "Court 13"],
                               6, [4] * 24, samples=64)
        self.assertEqual(RESULTS[-1]["downstairs"], [2] * 24)

    def test_planner_does_not_change_random_generator_state(self):
        random.seed(43)
        before = random.getstate()
        self.ns["build_downstairs_rotation"](list(range(18)), 6, 4, 12, limits=[2] * 18,
                                             genders=["F"] * 9 + ["M"] * 9,
                                             skills=[5] * 18, enable_skill=True)
        self.assertEqual(before, random.getstate())

    def test_optional_skill_and_same_sex_improve_quality(self):
        totals = Counter()
        # Same pools, repeated seeds and a practical search budget compare modifiers.
        for seed in range(6):
            for skill, same in ((False, False), (True, False), (False, True)):
                result = self.run_case(24, ["Court 1", "Court 2", "Court 3", "Court 4", "Court 12", "Court 13"],
                                       6, [2] * 24, seed=seed, skill=skill, same=same, count=6, samples=128)
                gap = mixed = 0
                for groups, _ in result[4]:
                    for g in groups:
                        gap += abs(sum(4+p%4 for p in g[:2]) - sum(4+p%4 for p in g[2:]))
                        mixed += len({p % 2 for p in g}) > 1
                totals[(skill, same, "gap")] += gap
                totals[(skill, same, "mixed")] += mixed
        self.assertLess(totals[(True, False, "gap")], totals[(False, False, "gap")])
        self.assertLess(totals[(False, True, "mixed")], totals[(False, False, "mixed")])
        print("Modifier comparison:", dict(totals))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Audit)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    raise SystemExit(not result.wasSuccessful())
