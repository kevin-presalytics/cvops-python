""" Live video testing with a local video file
    This is omitted from the main test suite because it requires a video file to be present
    and is not a unit test

    Helpful for debugging and viewing live inference on a local video file
"""
import unittest
import json
import tests
import cvops
import cvops.video


class LocalVideoTests(unittest.TestCase):
    """ Tests for inference methods through the C library """

    def setUp(self):
        """ Runs before the tests """
        self.c_source_dir = tests.ROOT_DIR.joinpath("cvops-inference")
        self.c_test_files_dir = self.c_source_dir.joinpath("tests", "files")
        self.c_test_video_path = self.c_test_files_dir.joinpath("videos", "intersection-video.mp4")

    def test_local_video_inference(self):
        """ Displays a video with inference results """

        metadata_path = self.c_test_files_dir.joinpath("models", "yolov8n-metadata.json")

        with open(metadata_path, "r", encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        video_player_args = {
            "video_source": self.c_test_video_path,
            "model_path": self.c_test_files_dir.joinpath("models", "yolov8n.onnx"),
            "metadata": metadata,
            "model_platform": cvops.schemas.ModelPlatforms.YOLO,
            "confidence_threshold": 0.05,
            "iou_threshold": 0.3,
            "show_video": True,
            "debug": True,
        }

        with cvops.video.LocalModelVideoPlayer(**video_player_args) as video_player:
            video_player.play()
