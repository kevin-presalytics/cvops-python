""" Utility functions for cvops """
import pathlib
import typing
import os
import logging
import http
import requests
import onnx
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
        logger.error("Connection to upload url unstable.  Please check connecetion and retry.")
        raise conn_error
    except Exception as ex:  # pylint: disable
        logger.exception(ex, "Unable to upload file")
        raise ex


def export_to_onnx(
    path: pathlib.Path,
    framework: typing.Optional[typing.Union[str, cvops.schemas.ModelFrameworks]] = None,
    export_args: typing.Dict[str, typing.Any] = {},  # pylint: disable=dangerous-default-value
) -> pathlib.Path:
    """ exports and onnx model to file from a model on a local filepath"""
    if not framework:
        framework = cvops.schemas.ModelFrameworks.parse_from_filename(path.name)
    if isinstance(framework, str):
        framework = cvops.schemas.ModelFrameworks(framework)
    if framework == cvops.schemas.ModelFrameworks.ONNX:
        return path

    export_name = path.name.split(".")[-2] + ".onnx"
    export_path = pathlib.Path()

    if framework == cvops.schemas.ModelFrameworks.TORCH:
        try:
            import torch  # pylint: disable=import-outside-toplevel, import-error
            import torch.onnx  # pylint: disable=import-outside-toplevel, import-error
            model = torch.load(str(path))
            torch.onnx.export(model, export_name, **export_args)
        except ImportError:
            logger.error(
                "This model requires pytorch installed in the python environment.  You can install a compatible version with \"pip install cvops[torch]\"")
    if framework == cvops.schemas.ModelFrameworks.TENSORFLOW:
        import tf2onnx  # pylint: disable=import-outside-toplevel, import-error
        tf2onnx.convert.from_keras(model, output_path=export_path, **export_args)

    onnx_model = onnx.load(export_name)
    onnx.checker.check_model(onnx_model)
    return export_path
