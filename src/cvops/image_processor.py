import abc
import typing
import io
import ast
import os
import pathlib
import contextlib
import logging
import numpy
import onnxruntime
import requests
import cv2
import cvops
import cvops.schemas
import cvops.util
import cvops.inference
import cvops.inference.manager


logger = logging.getLogger(__name__)


class ImageProcessorBase(contextlib.AbstractContextManager, abc.ABC):
    """ Base class for image processors """

    @abc.abstractmethod
    def run(self, image: typing.Union[str, pathlib.Path, io.BytesIO, numpy.ndarray]) -> numpy.ndarray:
        """ Runs inference on the given image 

            The implementation of this method typically invokes the following methods:
            1. preprocess_image
            2. postprocess_image
        """
        return NotImplemented
    
    @abc.abstractmethod
    def preprocess_image(self, img: numpy.ndarray) -> numpy.ndarray:
        """ Converts an image to a numpy array for model input"""
        return NotImplemented
    
    @abc.abstractmethod
    def postprocess_image(self, img: numpy.ndarray, output: numpy.ndarray) -> numpy.ndarray:
        """ Updates the image to display the results of the inference

            img: The input image.
            output: The output of the model.

            Returns: The input image with detections drawn on it as a numpy.ndarray.
            Can be written to disk with cv2.imwrite        
        """
        return NotImplemented

class ImageUtilsMixIn(abc.ABC):
    """ Mixin with utility methods for working with images """
    _color_palette: typing.Dict[int, typing.Tuple[int, int, int]]
    _classes: typing.Dict[int, str]

    def __init__(self,
                *args,
                 color_palette: typing.Optional[typing.Dict[int, typing.Tuple[int, int, int]]] = None,
                 classes: typing.Optional[typing.Dict[int, str]] = None,
                 metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 **kwargs) -> None:
        self._classes = classes
        if not self._classes:
            self._classes = metadata.get("classes", None)
        self._color_palette = color_palette
        if not self._color_palette:
            if metadata.get("color_palette", None):
                self._color_palette = metadata.get("color_palette")
            self._color_palette = numpy.random.uniform(0, 255, size=(len(self.classes), 3))


    @property
    def classes(self) -> typing.Dict[int, str]:
        """ Returns id, label key-value pairs of the classes for the model """
        if self._classes:
            return self._classes
        else:
            return {}

    @property
    def color_palette(self) -> typing.Dict[int, typing.Tuple[int, int, int]]:
        """ Returns id, color key-value pairs of the classes for the model """
        if self._color_palette:
            if len(self._color_palette) != len(self.classes):
                raise ValueError("Color palette wrong shape: The number of colors in the color palette does not match the number of classes")
            return self._color_palette
        else:
            raise Exception("Color palette not defined")  #pylint: disable

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def visualize_inference(self, img: numpy.ndarray, filepath: typing.Union[pathlib.Path, str]):
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        cv2.imwrite(filepath, img)

    def extract_image_to_array(self, image: typing.Union[str, pathlib.Path, io.BytesIO]) -> numpy.ndarray:
        """ Loads a image to an numpy.ndarray from a reference """
        if isinstance(image, numpy.ndarray):
            return image
        elif isinstance(image, str):
            if image.startswith("http"):
                img_stream = io.BytesIO(requests.get(image).content)
                return cv2.imdecode(numpy.frombuffer(img_stream.read(), numpy.uint8), 1)
            else:
                if not os.path.exists(image):
                    raise ValueError(f"The path to image ${image} does not exist.")
                return cv2.imread(image)
        elif isinstance(image, pathlib.Path):
            return cv2.imread(str(image))
        else:
            raise TypeError(f"The type ${image.__class__.__name__} is not support for reading")
    


class LocalImageProcessor(ImageProcessorBase, ImageUtilsMixIn):
    """
    A pure python implementation of an image processor that runs inference on a local model.
    """
    model_platform: cvops.schemas.ModelPlatforms
    model_path: pathlib.Path
    onnx_session: onnxruntime.InferenceSession
    confidence_threshold: float
    iou_threshold: float
    _classes: typing.Optional[typing.Dict[int, str]]

    def __init__(self,
                 model_platform: cvops.schemas.ModelPlatforms,
                 model_path: pathlib.Path,
                 *args,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 classes: typing.Optional[typing.Dict[int, str]] = None,
                 metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 **kwargs) -> None:
        super().__init__(
            *args,
            model_platform=model_platform,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            metadata=metadata,
            **kwargs
        )
        self.model_platform = model_platform
        if not os.path.exists(model_path):
            raise ValueError(f"Model cannot be found at {model_path}")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __enter__(self):
        try:
            self.onnx_session = onnxruntime.InferenceSession(str(self.model_path))
        except Exception as e:
            logger.execption(e)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.onnx_session.close()
        except:
            pass


    @property
    def input_name(self) -> str:
        """ Name of the input tensor """
        return self.onnx_session.get_inputs()[0].name

    @property
    def output_names(self) -> typing.List[str]:
        """ Name of the output tensor """
        return [o.name for o in self.onnx_session.get_outputs()]

    @property
    def image_height(self) -> int:
        """ Height of the input image for the model """
        return self.onnx_session.get_inputs()[0].shape[2]
    
    @property
    def image_width(self) -> int:
        """ Width of the input image for the model """
        return self.onnx_session.get_inputs()[0].shape[3]

    @property
    def classes(self) -> typing.Dict[int, str]:
        """ Map of class_id to class_name """        
        if self._classes:
            return self._classes
        return ast.literal_eval(self.onnx_session._model_meta.custom_metadata_map['names'])

    def run(self, image: typing.Union[str, pathlib.Path, io.BytesIO]):
        img = self.extract_image_to_array(image)
        input_data = self.preprocess_image(img)
        output_data = self.onnx_session.run(self.output_names, {self.input_name: input_data})
        return self.postprocess_image(img, output_data)

    def preprocess_image(self, img: numpy.ndarray) -> numpy.ndarray:
        """ Converts an image to a numpy array for model input"""
        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.image_width, self.image_height))

        # Normalize the image data by dividing it by 255.0
        image_data = numpy.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = numpy.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
        image_data = numpy.expand_dims(image_data, axis=0).astype(numpy.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess_image(self, img: numpy.ndarray, output: numpy.ndarray) -> numpy.ndarray:
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = numpy.transpose(numpy.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        img_height, img_width = img.shape[:2]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = img_width / self.image_width
        y_factor = img_height / self.image_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = numpy.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_threshold:
                # Get the class ID with the highest score
                class_id = numpy.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(img, box, score, class_id)

        # Return the modified input image
        return img


class AcceleratedImageProcessor(ImageProcessorBase, ImageUtilsMixIn):
    """ A Ctypes implementation of the ImageProcessor """
    session_manager: cvops.inference.manager.InferenceSessionManager
    model_path = pathlib.Path
    model_platform: cvops.schemas.ModelPlatforms
    confidence_threshold: float
    iou_threshold: float
    metadata: typing.Dict[str, typing.Any]

    def __init__(self,
                 *args,
                 model_platform: cvops.schemas.ModelPlatforms,
                 model_path: pathlib.Path,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 **kwargs) -> None:
        super().__init__(*args, 
            model_platform=model_platform,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            metadata=metadata,
            **kwargs
        )
        self.model_platform = model_platform
        if not os.path.exists(model_path):
            raise ValueError(f"Model cannot be found at {model_path}")
        self.model_path = model_path
        self.onnx_session = onnxruntime.InferenceSession(str(self.model_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.metadata = metadata or {}
        self.session_manager = cvops.inference.manager.InferenceSessionManager()

    def __enter__(self):
        try:
            self.session_manager.start_session(
                self.model_platform,
                self.model_path,
                self.metadata,
                self.confidence_threshold,
                self.iou_threshold)
        except Exception as e:
            logger.exception(e)
    
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.session_manager.close_session()
        except Exception:
            pass


    def run(self, image: typing.Union[str, pathlib.Path, io.BytesIO]) -> numpy.ndarray:
        image_bytes = self.get_image_bytes(image)
        try:
            c_inference_result = self.session_manager.run_inference(image_bytes)
            if not c_inference_result:
                raise Exception("Null pointer return from run_inference method in C library")
            return cvops.schemas.InferenceResult.from_c_type(c_inference_result)
        except Exception as ex:
            c_error = self.session_manager.get_error()
            if len(c_error) > 0:
                err_message = f"Error from C Library: {c_error}"
                logger.error(err_message)
                raise Exception(err_message)
            else:
                logger.exception(ex)
                raise ex

    def get_image_bytes(self, image: typing.Union[str, pathlib.Path, io.BytesIO]) -> bytes:
        if isinstance(image, io.BytesIO):
            return image.read()
        elif isinstance(image, str):
            if image.startswith("http"):
                return requests.get(image).content
            else:
                if not os.path.exists(image):
                    raise ValueError(f"The path to image ${image} does not exist.")
                return open(image, "rb").read()
        elif isinstance(image, pathlib.Path):
            return open(str(image), "rb").read()
        elif isinstance(image, numpy.ndarray):
            return cv2.imencode('.png', image)[1].tobytes()
        else:
            raise TypeError(f"The type ${image.__class__.__name__} is not support for reading")

    def preprocess_image(self, img: numpy.ndarray) -> numpy.ndarray:
        pass

    def postprocess_image(self, img: numpy.ndarray, output: numpy.ndarray) -> numpy.ndarray:
        pass