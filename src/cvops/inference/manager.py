# Manager classes for handling resources and methods located in the C library
import typing
import ctypes
import json
import collections
import logging
import numpy
import cv2
import cvops.schemas
import cvops.inference
import cvops.inference.c_api
import cvops.inference.c_interfaces as _types
import cvops.inference.factories


logger = logging.getLogger(__name__)


class InferenceSessionManager(cvops.inference.c_api.CApi):
    """ Wrapper class around the calls to inference methods of the C API """
    session: typing.Optional[ctypes.POINTER(cvops.inference.c_interfaces.IInferenceManager)]
    session_request: _types.InferenceSessionRequest
    _is_in_context_manager: bool

    def __init__(self, session_request: _types.InferenceSessionRequest) -> None:
        super().__init__()
        self.session = None
        assert isinstance(session_request, _types.InferenceSessionRequest), \
            "session_request must be an cvops.inference.c_interfaces.InferenceSessionRequest"
        self.session_request = session_request
        self.session_request_ptr = ctypes.pointer(self.session_request)
        self._is_in_context_manager = False

    def __enter__(self) -> "InferenceSessionManager":
        try:
            self._start_session()
            self._is_in_context_manager = True
            return self
        except Exception as ex:  # pylint: disable=broad-except
            raise RuntimeError("Unable to start session") from ex

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.end_session()
        except Exception:  # pylint: disable=broad-except
            logger.error("Unable to end session")
        finally:
            self._is_in_context_manager = False

    def _start_session(self) -> None:
        self.session = self.dll.start_inference_session(self.session_request_ptr)
        if self.session is None:
            raise Exception("Unable to start inference session")  # pylint: disable=broad-exception-raised

    def _run_inference(self,
                       request: _types.InferenceRequest
                       ) -> _types.InferenceResult:
        result_ptr = self.dll.run_inference(self.session, request)
        assert isinstance(result_ptr, _types.c_inference_result_p), \
            "Invalid Inference return type from C Library"
        return result_ptr

    def run_inference(self,
                      image: typing.Union[numpy.ndarray, bytes],
                      name: str = "",
                      draw_detections: bool = False
                      ) -> _types.c_inference_result_p:
        """ Runs inference on the given image """
        if not self._is_in_context_manager:
            raise RuntimeError("Must use InferenceSessionManager in a context manager")
        if isinstance(image, numpy.ndarray):
            image = cv2.imencode('.png', image)[1].tobytes()
        if not isinstance(image, bytes):
            raise TypeError("Image must be a numpy array or bytes")  # pylint: disable=broad-except
        c_image = ctypes.c_char_p(image)
        c_name = ctypes.c_char_p(name.encode('utf-8'))
        c_size = ctypes.c_int(len(image))
        c_draw_detections = ctypes.c_bool(draw_detections)

        request = _types.InferenceRequest(
            bytes=c_image,
            name=c_name,
            size=c_size,
            draw_detections=c_draw_detections,
        )
        return self._run_inference(request)

    def end_session(self) -> None:
        """ closes session in memory """
        self.dll.end_inference_session(self.session)
        self.session = None

    def multi_run_inference(self,
                            image: typing.Union[numpy.ndarray, bytes],
                            name: str = "",
                            draw_detections: bool = False,
                            num_inferences: int = 5
                            ) -> _types.c_inference_result_p:
        """ Runs inference on the given image multiple times, and return the mode result """
        results = []
        detections = []
        for i in range(num_inferences):
            result_ptr = self.run_inference(image, name, draw_detections)
            results.append(result_ptr)
            detections.append(result_ptr.contents.boxes_count)

        mode = max(set(detections), key=detections.count)
        return next(result for result in results if result.contents.boxes_count == mode)

    def get_error(self) -> str:
        """ Gets error message from the C Library's global error container """
        c_error = self.dll.error_message() or ""
        if isinstance(c_error, bytes):
            return c_error.decode('utf-8')
        return c_error


class InferenceResultRenderer(cvops.inference.c_api.CApi):
    """ Renders InferenceResults onto Images """
    color_palette: typing.Dict[int, typing.Tuple[int, int, int]]
    classes: typing.Dict[int, str]
    _is_in_context_manager: bool

    def __init__(self,
                 classes: typing.Dict[int, str],
                 color_palette: typing.Optional[typing.List[typing.Tuple[int, int, int]]] = None,
                 ) -> None:
        super().__init__()
        self._is_in_context_manager = False
        if not isinstance(classes, dict):
            raise TypeError("Classes must be a dictionary")
        if not all(isinstance(value, str) for value in classes.values()):
            raise TypeError("Classes values (names) must be strings")
        self.classes = classes
        self.color_palette = color_palette
        if color_palette:
            if not isinstance(color_palette, dict):
                raise TypeError("Color palette must be a list")
            if not all(isinstance(value, collections.Sequence) for value in color_palette.values()):
                raise TypeError("Color palette values must be tuples or lists")
            if not all(len(value) == 3 for value in color_palette.values()):
                raise ValueError("Color palette values must be tuples or lists  of length 3")
            if not len(color_palette.keys()) == len(classes.keys()):
                raise ValueError("Color palette must have same number of colors as classes")
        else:
            num_colors = len(classes.keys())
            self.color_palette = cvops.image_processor.generate_color_palette(num_colors)

    def __enter__(self):
        try:
            metadata = {
                "classes": self.classes,
                "color_palette": self.color_palette,
            }
            metadata_string = json.dumps(metadata)
            c_color_palette = ctypes.c_char_p(metadata_string.encode('utf-8'))
            self.dll.set_color_palette(c_color_palette)
            self._is_in_context_manager = True
        except Exception as ex:  # pylint: disable=broad-except
            raise RuntimeError("Unable to set color palette") from ex
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.dll.free_color_palette()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Unable to free color palette")
        finally:
            self._is_in_context_manager = False

    def render(self, inference_result_ptr: _types.c_inference_result_p, image: numpy.ndarray) -> None:
        """ Renders the given inference result onto an image """
        if not self._is_in_context_manager:
            raise RuntimeError("Must use InferenceResultRenderer in a context manager")
        # Channels per https://stackoverflow.com/a/53758304/16580040
        num_channels = image.shape[-1] if image.ndim == 3 else 1
        self.dll.render_inference_result(
            inference_result_ptr,
            image.ctypes._data,  # pylint: disable=protected-access
            image.shape[0],
            image.shape[1],
            num_channels
        )