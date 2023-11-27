""" Tests for the object tracking classes """
import unittest
import json
import ctypes
import tests
import cvops.video
import cvops.tracking
import cvops.workflows
import cvops.inference.manager
import cvops.inference.factories



class TrackerTests(unittest.TestCase):
    """ Tests for object tracking methods through the C library """

    def setUp(self):
        """ Runs before the tests """
        self.c_source_dir = tests.ROOT_DIR.joinpath("cvops-inference")
        self.c_test_files_dir = self.c_source_dir.joinpath("tests", "files")
        self.c_test_video_path = self.c_test_files_dir.joinpath("videos", "intersection-video.mp4")

    def test_video_tracking(self):
        """ Tracks objects in a video after initial inference"""
        c_test_model_path = self.c_test_files_dir.joinpath("models", "yolov8n.onnx")
        
        metadata_path = self.c_test_files_dir.joinpath("models", "yolov8n-metadata.json")

        with open(metadata_path, "r", encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        
        max_frames = 100

        video_tracker_args = {
            "video_source": self.c_test_video_path,
            "model_path": c_test_model_path,
            "metadata": metadata,
            "model_platform": cvops.schemas.ModelPlatforms.YOLO,
            "confidence_threshold": 0.05,
            "iou_threshold": 0.3,
            "show_video": True,
            "debug": True,
        }

        # Act

        tracker_state = None
        initial_inference_result = None

        class TestVideoTracker(cvops.tracking.VideoObjectTrackerMixin, cvops.inference.manager.InferenceSessionManager, cvops.video.VideoPlayerBase):
            """ Test class for testing the video tracking Mixin """
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.frame_count = 0
                self.initial_inference_result = None
                self.initial_inference_result_ptr = None
                self.end_tracker_state = None
                self.exception = None

            def process_frame(self, frame):
                """ Processes a frame """
                try:
                    if self.frame_count >= max_frames:
                        self.stop()
                    elif self.frame_count == 0:
                        c_initial_inference_result = self.run_inference(frame)
                        self.initial_inference_result = cvops.inference.factories.inference_result_from_c_type(c_initial_inference_result)
                        new_boxes = [next(b for b in self.initial_inference_result.boxes if b.class_name == "person")]
                        self.initial_inference_result.boxes = new_boxes
                        self.initial_inference_result_ptr = cvops.inference.factories.inference_result_to_c_type_ptr(self.initial_inference_result)
                        self.dispose_inference_result(c_initial_inference_result)
                        self.update_tracker(frame, self.initial_inference_result_ptr)
                    else:
                        self.track_image(frame)
                    self.frame_count += 1
                    return frame
                except Exception as ex:
                    self.exception = ex
                    raise RuntimeError from ex


        with TestVideoTracker(**video_tracker_args) as video_tracker:
            video_tracker.play()
            if video_tracker.exception:
                raise video_tracker.exception
            tracker_state_ptr = video_tracker.get_tracker_state()
            tracker_state = cvops.inference.factories.tracker_state_ptr_to_boxes(tracker_state_ptr)
            initial_inference_result = video_tracker.initial_inference_result
            video_tracker.dispose_tracker_state(tracker_state_ptr)
        
        
        # Assert
        self.assertIsNotNone(tracker_state)
        self.assertIsNotNone(initial_inference_result)


