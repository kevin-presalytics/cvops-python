"""User configuration for the CVOps SDK Runtime"""
import typing
from pydantic_settings import BaseSettings


class SettingsManager(BaseSettings):
    """ Settings for the CVOps SDK Runtime"""
    device_id: str = ""
    device_secret: str = ""
    mqtt_uri: str = "mqtts://mqtt.cvops.io:8883"
    log_level: str = "INFO"
    mqtt_timeout: typing.Optional[float] = None
    model_path: str = "./models/llama-2-7b-chat"
    deployment_timeout: float = 3600.0


SETTINGS = SettingsManager(_env_file='.env', _env_file_encoding='utf-8')  # type: ignore
