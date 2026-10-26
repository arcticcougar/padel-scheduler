"""Organiser-selected downstairs groups retain all scheduler hard constraints."""
import copy
import itertools
import unittest
from collections import Counter
from datetime import datetime

import numpy as np
from test_downstairs import load_source


class ManualDownstairsChecks(unittest.TestCase):
    def setUp(self):
        self.ns = load_source()

    def generate(self, n, courts, groups, limits=None, rounds=6, **kwargs):
        return self.ns["assign_matches"](
            [f"PRIVATE QA {i}" for i in range(n)], ["F", "M"] * (n // 2) + (["F"] if n % 2 else []),
            [4 + i % 4 for i in range(n)], courts, total_matches=rounds,
            samples=32, downstairs_limits=limits, downstairs_groups=groups,
            start_dt=datetime(2026, 10, 26, 10, 30), match_offset=6, **kwargs,
        )

    def verify_groups(self, result, n, courts, requested, limits):
        markup, tm, om, rests, matches, used = result
        down, _ = self.ns["identify_downstairs_courts"](courts[:used])
        counts, rest_counts = Counter(), Counter()
        for i, (groups, bench) in enumerate(matches):
            actual = {p for cid in down for p in groups[cid]}
            self.assertEqual(actual, set(requested[i // 2]))
            self.assertEqual(len(groups), used)
            self.assertTrue(all(len(g) == 4 for g in groups))
            playing = list(itertools.chain.from_iterable(groups))
            self.assertEqual(sorted(playing + bench), list(range(n)))
            self.assertEqual(len(playing), len(set(playing)))
            self.assertFalse(actual & set(bench))
            counts.update(actual)
            rest_counts.update(bench)
        self.assertEqual(rests.tolist(), [rest_counts[p] for p in range(n)])
        self.assertTrue(all(counts[p] <= limits[p] for p in range(n)))
        self.assertTrue(np.array_equal(tm, tm.T) and np.array_equal(om, om.T))
        self.assertEqual(int(tm.sum()), len(matches) * used * 4)
        self.assertEqual(int(om.sum()), len(matches) * used * 8)
        self.assertEqual(markup.count('class="match-block"'), len(matches))
        self.assertEqual(markup.count("class='downstairs-badge'"), len(matches) * len(down))
        self.assertIn("Match 7 [10:30]", markup)
        per_page = 3 if used <= 4 else 2
        self.assertEqual(markup.count("page-break-after: always;"), (len(matches) - 1) // per_page)

    def test_one_downstairs_court_full_session(self):
        courts = ["Court 7", "Court 8", "Court 9", "Court 12"]
        groups = [list(range(4, 8)), list(range(8, 12)), list(range(12, 16))]
        limits = [0] * 4 + [2] * 15
        self.ns["build_downstairs_rotation"] = lambda *a, **k: self.fail("Manual groups must not be replaced by an automatic plan")
        original = copy.deepcopy(groups)
        result = self.generate(19, courts, groups, limits, enable_skill=True)
        self.verify_groups(result, 19, courts, groups, limits)
        self.assertEqual(groups, original)

    def test_two_downstairs_courts_share_one_group_with_rests(self):
        courts = ["Court 7", "Court 12", "Court 8", "Court 13", "Court 9", "Court 10"]
        groups = [list(range(3, 11)), list(range(11, 19)), list(range(19, 27))]
        limits = [0] * 3 + [2] * 24
        result = self.generate(27, courts, groups, limits, enable_skill=True,
                               reject_mixed=True, same_sex_matches_count=3)
        self.verify_groups(result, 27, courts, groups, limits)

    def test_four_game_permission_allows_two_blocks(self):
        courts = ["Court 7", "Court 8", "Court 12", "Court 13"]
        groups = [list(range(8)), list(range(8, 16)), list(range(8))]
        limits = [4] * 8 + [2] * 8
        self.verify_groups(self.generate(16, courts, groups, limits), 16, courts, groups, limits)
        with self.assertRaisesRegex(ValueError, "limit is 2"):
            self.generate(16, courts, groups, [2] * 16)
        with self.assertRaisesRegex(ValueError, "more than four"):
            self.generate(16, courts, [list(range(8))] * 3, [4] * 16)

    def test_odd_last_block_uses_actual_duration(self):
        courts = ["Court 7", "Court 12"]
        groups = [list(range(4)), list(range(4, 8)), list(range(8, 12))]
        limits = [2] * 12
        self.verify_groups(self.generate(12, courts, groups, limits, rounds=5), 12, courts, groups, limits)
        # The same people may play a two-round block and a final single round
        # only if they have four-game permission.
        with self.assertRaisesRegex(ValueError, "3 downstairs matches"):
            self.generate(8, courts, [list(range(4))] * 2, [2] * 8, rounds=3)
        result = self.generate(8, courts, [list(range(4))] * 2, [4] * 8, rounds=3)
        self.verify_groups(result, 8, courts, [list(range(4))] * 2, [4] * 8)

    def test_invalid_groups_fail_before_search(self):
        valid = [list(range(4)), list(range(4, 8)), list(range(8, 12))]
        scenarios = [
            (valid[:2], [2] * 12, "each of the 3 blocks"),
            ([[], *valid[1:]], [2] * 12, "Choose 4 more players"),
            ([[0, 1, 2, 3, 4], *valid[1:]], [2] * 12, "Remove 1 player"),
            ([[0, 0, 2, 3], *valid[1:]], [2] * 12, "more than once"),
            ([[0, 1, 2, 999], *valid[1:]], [2] * 12, "not playing today"),
            (valid, [0] + [2] * 11, "Keep upstairs"),
            (valid, [6] * 12, "0, 2 or 4"),
        ]
        for groups, limits, expected in scenarios:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                self.generate(12, ["Court 7", "Court 12"], groups, limits)
        with self.assertRaisesRegex(ValueError, "active downstairs court"):
            self.generate(12, ["Court 7", "Court 8"], valid, [2] * 12)

    def test_only_downstairs_and_one_round(self):
        self.verify_groups(self.generate(4, ["Court 12"], [[0, 1, 2, 3]], [2] * 4, rounds=1),
                           4, ["Court 12"], [[0, 1, 2, 3]], [2] * 4)

    def test_modifiers_reach_search_in_requested_rounds(self):
        original, calls = self.ns["find_best_match"], []
        def capture(*args, **kwargs):
            calls.append((args[5], args[7], args[9]))
            return original(*args, **kwargs)
        self.ns["find_best_match"] = capture
        self.generate(16, ["Court 7", "Court 8", "Court 9", "Court 12"],
                      [list(range(4)), list(range(4, 8)), list(range(8, 12))], [2] * 16,
                      reject_mixed=True, same_sex_matches_count=2, enable_skill=True)
        self.assertEqual({r for r, same, skill in calls if same}, {2, 5})
        self.assertTrue(all(skill for r, same, skill in calls))

    def test_context_lists_unused_first_then_other_block_selections(self):
        options, labels = self.ns["downstairs_group_choices"](
            ["Dee", "Aideen", "Chris", "Anna"], [["Aideen"], ["Chris"], ["Aideen"]], 1)
        self.assertEqual(options, ["Anna", "Chris", "Dee", "Aideen"])
        self.assertEqual(labels["Chris"], "Chris")  # Selection in this box alone is not a repeat.
        self.assertEqual(labels["Aideen"], "🔴 Aideen · already in blocks 1, 3")
        self.assertEqual(labels["Anna"], "Anna")

    def test_context_marking_updates_when_another_block_is_cleared(self):
        options, labels = self.ns["downstairs_group_choices"](
            ["Dee", "Aideen", "Chris", "Anna"], [["Aideen"], [], []], 0)
        self.assertEqual(options, ["Aideen", "Anna", "Chris", "Dee"])
        self.assertTrue(all(labels[name] == name for name in options))

    def test_manual_mode_cannot_silently_omit_selected_court_13(self):
        courts = ["Court 7", "Court 8", "Court 9", "Court 12", "Court 13"]
        groups = [list(range(8)), list(range(8, 16)), list(range(8))]
        limits = [4] * 8 + [2] * 11
        with self.assertRaisesRegex(ValueError, "5 selected courts need at least 20 players"):
            self.generate(19, courts, groups, limits)
        courts.remove("Court 9")
        self.verify_groups(self.generate(19, courts, groups, limits), 19, courts, groups, limits)

    def test_manual_court_capacity_counts_all_selected_courts(self):
        validate = self.ns["validate_manual_court_capacity"]
        with self.assertRaisesRegex(ValueError, "2 selected courts need at least 8 players"):
            validate(7, ["Court 12", "Court 13"])
        validate(8, ["Court 12", "Court 13"])
        validate(27, ["Court 7", "Court 8", "Court 9", "Court 10", "Court 12", "Court 13"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
