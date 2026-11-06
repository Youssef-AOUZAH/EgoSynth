"""Tests the keep-best / delete-the-rest logic of phase 3 with a fake referee (no GPU needed)."""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _fake_modules():
    tf = types.ModuleType("transformers")
    tf.AutoProcessor = mock.MagicMock()
    tf.Qwen2_5_VLForConditionalGeneration = mock.MagicMock()
    qv = types.ModuleType("qwen_vl_utils")
    qv.process_vision_info = lambda msgs: (None, None)
    return {"transformers": tf, "qwen_vl_utils": qv}


class PhaseThreeTests(unittest.TestCase):
    def _run(self, replies, keep_all=False):
        import generate_dataset as gd

        tmp = tempfile.mkdtemp()
        task_dir = os.path.join(tmp, "task_000")
        os.makedirs(task_dir)
        for i in range(1, len(replies) + 1):
            with open(os.path.join(task_dir, f"video_seed_{i}.mp4"), "wb") as f:
                f.write(b"x")
        args = types.SimpleNamespace(output_dir=tmp, num_seeds=len(replies), keep_all=keep_all, qwen_path="fake")
        it = iter(replies)
        with mock.patch.dict(sys.modules, _fake_modules()), \
             mock.patch.object(gd, "referee_video", lambda *a, **k: next(it)), \
             mock.patch.object(gd, "free_gpu", lambda *a, **k: None):
            gd.run_referee({"task_000": "open drawer"}, args)
        with open(os.path.join(task_dir, "eval_results.json")) as f:
            result = json.load(f)
        return result, sorted(os.listdir(task_dir))

    def test_best_kept_others_deleted_ties_go_to_lowest_seed(self):
        result, files = self._run(["Score: [70]", "Score: [88]", "Score: [88]"])
        self.assertEqual(result["winner"], "video_seed_2.mp4")
        self.assertEqual(files, ["eval_results.json", "video_seed_2.mp4"])

    def test_unreadable_reply_is_excluded_not_defaulted(self):
        result, files = self._run(["no score here", "Score: [55]", "Score: [XX]"])
        self.assertEqual(result["winner"], "video_seed_2.mp4")
        self.assertEqual(sorted(result["unparsed"]), ["video_seed_1.mp4", "video_seed_3.mp4"])

    def test_nothing_deleted_when_no_score_is_readable(self):
        result, files = self._run(["?", "?", "?"])
        self.assertIsNone(result["winner"])
        self.assertEqual(len([f for f in files if f.endswith(".mp4")]), 3)

    def test_keep_all(self):
        _, files = self._run(["Score: [10]", "Score: [20]", "Score: [30]"], keep_all=True)
        self.assertEqual(len([f for f in files if f.endswith(".mp4")]), 3)


if __name__ == "__main__":
    unittest.main()
