""" Unit Tests for Inference methods """
import unittest
import tempfile
import shutil
import pathlib
import tests
import cvops
import cvops.workflows


class InferenceTests(unittest.TestCase):
    """ Tests for inference methods through the C library """

    def setUp(self):
        """ Runs before the tests """
        self.output_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        # Uncomment the line below if you want to see the output files
        # self.output_dir = tests.ROOT_DIR.joinpath("out")

        if isinstance(self.output_dir, pathlib.Path):
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
        self.c_source_dir = tests.ROOT_DIR.joinpath("cvops-inference")
        self.c_test_files_dir = self.c_source_dir.joinpath("tests", "files")
        self.c_test_images_dir = self.c_test_files_dir.joinpath("images")

    def tearDown(self):
        """ Runs after the tests """
        if isinstance(self.output_dir, tempfile.TemporaryDirectory):
            self.output_dir.cleanup()

    def test_yolov8_object_detection(self):
        """ Runs inference on a set of test files """

        # Assume
        c_test_model_path = self.c_test_files_dir.joinpath("models", "yolov8n.onnx")
        c_test_metadata_path = self.c_test_files_dir.joinpath("models", "yolov8n-metadata.json")
        input_file_count = len([f for f in self.c_test_images_dir.glob("*") if f.is_file()])

        # Act
        results = cvops.workflows.run_inference_on_directory(
            input_directory=self.c_test_images_dir,
            model_path=c_test_model_path,
            metadata_path=c_test_metadata_path,
            output_directory=self.output_dir,
            model_platform=cvops.schemas.ModelPlatforms.YOLO,
            confidence_threshold=0.5,
            iou_threshold=0.4,
        )

        # Assert

        # Ensure all images were processed
        self.assertEqual(len(results), input_file_count)
        num_output_files = len([f for f in pathlib.Path(self.output_dir.name).glob("*") if f.is_file()])
        self.assertEqual(num_output_files, input_file_count)

        for result in results:
            for box in result.boxes:
                # Ensure boxes are valid sized
                self.assertGreaterEqual(box.width, 0)
                self.assertGreaterEqual(box.height, 0)

                # Ensure boxes are valid positioned (within image bounds)
                self.assertGreater(box.x, -box.width)
                self.assertGreater(box.y, -box.height)

                # Ensure boxes have valid confidence, class_id, class_name, and object_id
                self.assertGreaterEqual(box.confidence, 0)
                self.assertGreaterEqual(box.class_id, 0)
                self.assertIsInstance(box.class_name, str)
                self.assertGreater(len(box.class_name), 0)
                self.assertGreaterEqual(box.object_id, 0)
