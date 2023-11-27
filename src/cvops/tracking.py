""" Classes to handle object tracking in videos """
import typing
import ctypes
import contextlib
import logging
import numpy
import cvops.schemas
import cvops.inference.factories
import cvops.inference.c_api
import cvops.inference.c_interfaces as _types


logger = logging.getLogger(__name__)


class VideoObjectTrackerMixin(cvops.inference.c_api.CApi, contextlib.AbstractContextManager):
    """ Wrapper class around the calls to object tracking methods of the C API """
    _tracker_ptr: _types.c_tracker_p
    tracking_algorithm_type: cvops.schemas.TrackingAlgorithmTypes

    def __init__(self, 
                 tracking_algorithm_type: typing.Union[str, cvops.schemas.TrackingAlgorithmTypes] = cvops.schemas.TrackingAlgorithmTypes.MOSSE,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(tracking_algorithm_type, str):
            tracking_algorithm_type = cvops.schemas.TrackingAlgorithmTypes(tracking_algorithm_type)
        assert isinstance(tracking_algorithm_type, cvops.schemas.TrackingAlgorithmTypes), \
            "tracking_algorithm_type must be a cvops.schemas.TrackingAlgorithmType"
        self.tracking_algorithm_type = tracking_algorithm_type

    def create_tracker(self) -> None:
        """ Creates a tracker """
        c_tracking_algorithm_type = _types.TRACKING_ALGORITHMS_C_MAP[self.tracking_algorithm_type]
        self._tracker_ptr = self.dll.create_tracker(ctypes.c_int(c_tracking_algorithm_type))
        if not self._tracker_ptr:
            raise RuntimeError("Unable to create tracker")  # pylint: disable=broad-exception-raised

    def dispose_tracker(self) -> None:
        """ Disposes the tracker """
        if self._tracker_ptr is not None:
            self.dll.dispose_tracker(self._tracker_ptr)
            self._tracker_ptr = None

    def track_image(self, image: numpy.ndarray) -> None:
        """ Tracks the objects in the given image """
        if self._tracker_ptr is None:
            raise RuntimeError("Tracker not initialized")
        if not isinstance(image, numpy.ndarray):
            raise RuntimeError("image must be a numpy.ndarray")
        cv_mat = cvops.inference.factories.frame_to_cv_mat(image)
        self.dll.track_image(self._tracker_ptr, *cv_mat)

    def update_tracker(self, image: numpy.ndarray, inference_result: typing.Union[_types.c_inference_result_p, cvops.schemas.InferenceResult]) -> None:
        """ Updates the tracker with the given image """
        if self._tracker_ptr is None:
            raise RuntimeError("Tracker not initialized")
        if not isinstance(image, numpy.ndarray):
            raise RuntimeError("image must be a numpy.ndarray")
        if isinstance(inference_result, cvops.schemas.InferenceResult):
            inference_result = cvops.inference.factories.inference_result_to_c_type_ptr(inference_result)
        if isinstance(inference_result, _types.InferenceResult):
            inference_result = ctypes.pointer(inference_result)
        assert isinstance(inference_result, _types.c_inference_result_p), \
            "inference_result must be a pointer to a cvops.inference.c_interfaces.InferenceResult instance"
        cv_mat = cvops.inference.factories.frame_to_cv_mat_data(image)
        self.dll.update_tracker(self._tracker_ptr, inference_result, *cv_mat)

    def get_tracker_state(self) -> _types.c_tracker_state_p:
        """ Returns the state of the tracker """
        if self._tracker_ptr is None:
            raise RuntimeError("Tracker not initialized")
        tracker_state_ptr = self.dll.get_tracker_state(self._tracker_ptr)
        if (tracker_state_ptr):
            return tracker_state_ptr
        else:
            raise RuntimeError("Unable to get tracker state")  # pylint: disable=broad-exception-raised

    def dispose_tracker_state(self, tracker_state_ptr: _types.c_tracker_state_p) -> None:
        """ Disposes the tracker state """
        if tracker_state_ptr:
            try:
                self.dll.dispose_tracker_state(tracker_state_ptr)
            except BaseException:  # pylint: disable=broad-exception-caught
                pass

    def __enter__(self) -> "VideoObjectTracker":
        try:
            super().__enter__()
            self.create_tracker()
            return self
        except Exception as ex:  # pylint: disable=broad-exception-caught
            raise RuntimeError("Unable to start tracking") from ex


    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._tracker_ptr:
                self.dispose_tracker()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error("Error while disposing Tracker")
        finally:
            super().__exit__(exc_type, exc_value, traceback)
