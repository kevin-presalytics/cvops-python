""" Utility functions for cvops """
import pathlib
import typing
import os
import logging
import http
import requests
import onnx
import numpy
import cv2
import cvops.schemas


logger = logging.getLogger(__name__)


def download_file(
    url: str, target_file_path: typing.Optional[pathlib.Path] = None,
    download_timeout: int = 3600
) -> pathlib.Path:
    """ Downloads a large file synchronously """
    fname = url.split("/")[-1]
    if not target_file_path:
        target_file_path = pathlib.Path(os.getcwd(), fname)
    with requests.get(url, stream=True, timeout=download_timeout) as r:
        r.raise_for_status()
        with target_file_path.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return target_file_path


def upload_file(
    url: str,
    file_path: pathlib.Path,
    upload_timeout: int = 3600,
    method: str = "PUT"  # Note: no enum for this in std library prior to 3.11.  Use str for backwards compatibility
) -> None:
    """ Upload a local file to a url.  Defaults to PUT http method"""
    try:
        files = [(file_path.name, open(file_path, 'rb'))]
        if method == "PUT":
            resp = requests.put(url, files=files, timeout=upload_timeout)
        else:
            resp = requests.post(url, files=files, timeout=upload_timeout)
        if not resp.ok:
            raise ConnectionError
    except ConnectionError as conn_error:
        logger.error("Connection to upload url unstable.  Please check connection and retry.")
        raise conn_error
    except Exception as ex:  # pylint: disable
        logger.exception(ex, "Unable to upload file")
        raise ex


def export_to_onnx(
    path: pathlib.Path,
    platform: typing.Optional[typing.Union[str, cvops.schemas.ModelPlatforms]] = None,
    export_args: typing.Optional[typing.Dict[str, typing.Any]] = None  # pylint: disable=dangerous-default-value
) -> pathlib.Path:
    """ exports and onnx model to file from a model on a local filepath"""
    try:
        if not export_args:
            export_args = {}
        if platform:
            if not isinstance(platform, cvops.schemas.ModelPlatforms):
                platform = cvops.schemas.ModelPlatforms(platform)
            if platform == cvops.schemas.ModelPlatforms.YOLO:
                from ultralytics import YOLO
                model = YOLO(path)
                output_path = model.export(format='onnx', **export_args)
                return pathlib.Path(output_path)
            else:
                raise NotImplementedError("Only YOLOv8 is currently supported for export")
        else:
            raise NotImplementedError("Only YOLOv8 is currently supported for export")
    except ImportError:
        logger.error("Unable to import required dependencies for model export.  Please install ultralytics and torch to export YOLOv8 models")
        raise