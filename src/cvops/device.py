""" Manage device connections, topics and data transfer to/from the CVOps Mqtt Broker. """
import typing
import platform
import logging
import time
import paho.mqtt.client
import cvops.config
import cvops.schemas
import cvops.mqtt
import cvops.events


logger = logging.getLogger(__name__)


class DeviceManager(cvops.events.EventManager):
    """
    Manage connections, topics and data transfer to/from the current device and
    The CVOps Mqtt Broker.
    """
    device: typing.Optional[cvops.schemas.Device]
    workspace: typing.Optional[cvops.schemas.Workspace]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.workspace = None
        self.device = None

        def event_callback(*args):
            self.handle_platform_event(*args)

        def handle_details_response(*args):
            self.handle_details_response(*args)
        self.set_event_callback(cvops.events.EventTypes.DEVICE_DETAILS_RESPONSE, handle_details_response)

        def handle_workspace_details_response(*args):
            self.handle_workspace_details_response(*args)
        self.set_event_callback(cvops.events.EventTypes.WORKSPACE_DETAILS_RESPONSE, handle_workspace_details_response)
        self.subscribe(self.device_events_topic, event_callback)

    @property
    def device_data_topic(self) -> str:
        """Returns the topic for device data"""
        return f"device/{self.device_id}/data"

    @property
    def device_command_topic(self) -> str:
        """Returns the topic for device commands"""
        return f"device/{self.device_id}/command"

    @property
    def device_status_topic(self) -> str:
        """Returns the topic for device status"""
        return f"device/{self.device_id}/status"

    @property
    def device_events_topic(self) -> str:
        """Returns the topic for device events"""
        return f"events/device/{self.device_id}"

    @property
    def workspace_events_topic(self) -> str:
        """Returns the topic for workspace events"""
        if not self.device:
            raise ValueError("Device not set")
        return f"events/workspace/{self.device.workspace_id}"

    @property
    def workspace_storage_topic(self) -> str:
        """ Returns the topic for workspace storage actions"""
        if not self.device:
            raise ValueError("Device not set.")
        return f"workspace/{self.device.workspace_id}/storage"

    def get_device_info(
        self,
        handler: typing.Optional[typing.Callable[[cvops.events.DeviceDetailsResponseEvent], None]] = None
    ) -> None:
        """ Returns the device info from the MQTT Broker"""
        if handler:
            self.set_event_callback(cvops.events.EventTypes.DEVICE_DETAILS_RESPONSE, handler)
        if not self.is_connected and not self.is_connecting:
            self._start_connection()
        payload = cvops.events.DeviceDetailsRequestEvent()
        payload.response_topic = self.device_events_topic

        message = cvops.mqtt.MqttMessage(
            self.device_events_topic,
            payload.to_json(),
            cvops.mqtt.QualityOfService.EXACTLY_ONCE,
            True,
            self.device_events_topic,
            ""
        )
        self.publish(message)

    def activate_device(self):
        """ Activates the device with the MQTT Broker"""
        if not self.is_connected:
            return None
        self.device.activation_status = "Active"
        self.device.device_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "os_machine": platform.machine(),
            "os_processor": platform.processor(),
            "os_platform": platform.platform(),
            "os_node": platform.node(),
            "os_python_version": platform.python_version(),
        }
        self.device.name = platform.node()
        self.device.description = platform.platform()
        payload = cvops.events.DeviceRegisteredEvent()
        payload.event_data = self.device.model_dump(by_alias=True)
        payload.workspace_id = self.device.workspace_id
        message = cvops.mqtt.MqttMessage(self.device_events_topic, payload.to_json(),
                                         cvops.mqtt.QualityOfService.EXACTLY_ONCE.value, True, None, "")
        self.publish(message)

    def handle_platform_event(self, _, __, message: paho.mqtt.client.MQTTMessage):
        """ Callback for when the device command topic messages are received from the MQTT Broker"""
        try:
            msg_wrapper = cvops.mqtt.MqttMessage(
                message.topic,
                message.payload,
                message.qos,
                message.retain,
                getattr(message.properties, 'ResponseTopic', None),
                getattr(message.properties, 'CorrelationData', None)
            )
            event: cvops.events.BaseEvent = cvops.events.BaseEvent.factory(msg_wrapper)  # type: ignore
            logger.debug("Device Event Received. Event Type: \"%s\"", event.event_type)
            self._invoke_event_callbacks(event)
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception("Error receiving device command topic message: %s", ex)

    def handle_details_response(self, event: cvops.events.DeviceDetailsResponseEvent):
        """ Handles the device the details response from the MQTT Broker"""
        self.device = event.deserialize_event_data()
        if self.device:
            if self.device.activation_status.lower() != "active":
                self.activate_device()

    def get_workspace(self) -> cvops.schemas.Workspace:
        """ Returns the devices workspace from the MQTT Broker"""
        try:
            if self.is_connected or self.is_connecting:
                self._get_workspace()
            else:
                with self.open():
                    self._get_workspace()
            if self.workspace:
                return self.workspace
            raise ValueError("Workspace details not available.")  # Note: This should never happen
        except Exception as ex:
            logger.exception("Error getting workspace: %s", ex)
            raise ex
        
    def _get_workspace(self) -> None:
        """ Logic for getting the workspace """
        self.get_device_info()
        self._wait_for_value("device")
        payload = cvops.events.WorkspaceDetailsRequestEvent()
        payload.response_topic = self.device_events_topic
        if self.device:
            payload.device_id = self.device.id
            payload.workspace_id = self.device.workspace_id
            payload.response_topic = self.device_events_topic
        else:
            raise ValueError("Device details not available")
        message = cvops.mqtt.MqttMessage(
            self.device_events_topic,
            payload.to_json(),
            cvops.mqtt.QualityOfService.EXACTLY_ONCE,
            True,
            self.device_events_topic,
            ""
        )
        self.publish(message)
        self._wait_for_value("workspace")

    def _wait_for_value(self, key: str) -> None:
        """ Waits for the object to be set """
        timeout = cvops.config.SETTINGS.mqtt_timeout
        timer = 0.0
        delay = 0.1
        try:
            while not getattr(self, key):
                time.sleep(delay)
                if timeout:
                    timer += delay
                    if timer > timeout:
                        raise TimeoutError("Timeout waiting for device details")
        except KeyError:
            raise ValueError(
                f"Invalid watch dictionary. Key \"{key}\" does not exist on DeviceManager")  # pylint: disable=raise-missing-from

    def handle_workspace_details_response(self, event: cvops.events.WorkspaceDetailsResponseEvent):
        """ Handles the workspace details response from the MQTT Broker"""
        self.workspace = event.deserialize_event_data()
