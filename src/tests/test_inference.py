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
        temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = pathlib.Path(temp_dir, "out")
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        self.c_source_dir = tests.ROOT_DIR.joinpath("cvops-inference")
        self.c_test_files_dir = self.c_source_dir.joinpath("tests", "files")
        self.c_test_images_dir = self.c_test_files_dir.joinpath("images")

    def tearDown(self):
        """ Runs after the tests """
        shutil.rmtree(self.output_dir)


    def test_yolov8_object_detection(self):
        """ Runs inference on a set of test files """
        
        # Assume
        c_test_model_path = self.c_test_files_dir.joinpath("models", "yolov8n.onnx")
        c_test_metadata_path = self.c_test_files_dir.joinpath( "models", "yolov8n-metadata.json")
        input_file_count = len([f for f in self.c_test_images_dir.glob("*") if f.is_file()])

        # Act
        results = cvops.workflows.run_inference_on_directory(
            input_directory=self.c_test_images_dir,
            model_path=c_test_model_path,
            metadata_path=c_test_metadata_path,
            output_directory=self.output_dir,
            model_platform=cvops.schemas.ModelPlatforms.YOLO,
            confidence_threshold=0.5,
            iou_threshold=0.4
        )

        # Assert
        self.assertEqual(len(results), input_file_count)
        num_output_files = len(list(self.output_dir.glob("*")))
        self.assertEqual(num_output_files, input_file_count)
        
        for result in results:
            self.assertEqual(len(result.boxes), result.boxes_count)
            self.assertGreater(result.image_width, 0)
            self.assertGreater(result.image_height, 0)
            self.assertGreater(result.boxes_count, 0)
            for box in result.boxes:
                self.assertGreater(box.width, 0)
                self.assertGreater(box.height, 0)
                self.assertGreaterEqual(box.x, 0)
                self.assertGreaterEqual(box.y, 0)
                self.assertGreaterEqual(box.confidence, 0)
                self.assertGreaterEqual(box.class_id, 0)
                self.assertIsInstance(box.class_name, str)
                self.assertGreater(len(box.class_name), 0)
                self.assertGreaterEqual(box.object_id, 0)