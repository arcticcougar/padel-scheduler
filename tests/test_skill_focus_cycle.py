"""The default tip visits every preset once before a shuffled new cycle."""
import ast
import random
import unittest
from pathlib import Path

SOURCE = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8-sig"))
FUNCTION = next(n for n in SOURCE.body if isinstance(n, ast.FunctionDef) and n.name == "draw_next_skill_focus")
NAMESPACE = {"random": random}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), "app.py", "exec"), NAMESPACE)
draw = NAMESPACE["draw_next_skill_focus"]
OPTIONS = next(ast.literal_eval(n.value) for n in ast.walk(SOURCE)
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "focus_options" for t in n.targets))
PRESETS = [opt for opt in OPTIONS if opt != "Custom"]


class SkillFocusCycleChecks(unittest.TestCase):
    def test_every_tip_once_per_cycle_with_no_repeat_at_boundary(self):
        state, rng = {}, random.Random(24)
        previous = None
        orders = []
        for _ in range(50):
            choices = [draw(PRESETS, state, rng) for _ in PRESETS]
            self.assertEqual(len(choices), len(set(choices)))
            self.assertEqual(set(choices), set(PRESETS))
            self.assertNotEqual(choices[0], previous)
            previous = choices[-1]
            orders.append(tuple(choices))
        self.assertGreater(len(set(orders)), 1)

    def test_roster_of_tips_can_change(self):
        state, rng = {}, random.Random(7)
        draw(PRESETS, state, rng)
        updated = PRESETS[:-1]
        self.assertEqual(set(draw(updated, state, rng) for _ in updated), set(updated))

    def test_no_scheduling_randomness_consumed(self):
        random.seed(86)
        before = random.getstate()
        draw(PRESETS, {})
        self.assertEqual(before, random.getstate())

    def test_one_tip_and_empty_options(self):
        state = {}
        self.assertEqual([draw(["Only tip"], state) for _ in range(3)], ["Only tip"] * 3)
        with self.assertRaises(ValueError): draw([], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
