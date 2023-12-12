""" Classes to Manage MQTT Connections and Subscriptions """
import logging
import time
import sys
import typing
import json
import threading
import contextlib
from urllib.parse import urlparse
from enum import Enum
import paho.mqtt.client as mqtt
from paho.mqtt.packettypes import PacketTypes
from cvops.config import SETTINGS
import cvops.schemas


logger = logging.getLogger(__name__)


class QualityOfService(Enum):
    """MQTT Quality of Service"""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class ConnectionResultCodes(Enum):
    """MQTT Connection Result Codes"""
    CONNACK_ACCEPTED = 0
    CONNACK_REFUSED_PROTOCOL_VERSION = 1
    CONNACK_REFUSED_IDENTIFIER_REJECTED = 2
    CONNACK_REFUSED_SERVER_UNAVAILABLE = 3
    CONNACK_REFUSED_BAD_USERNAME_PASSWORD = 4
    CONNACK_REFUSED_NOT_AUTHORIZED = 5


class MqttMessage:
    """Wrapper for MQTT Messages"""
    topic: str
    payload: str
    qos: QualityOfService
    retain: bool
    response_topic: typing.Optional[str]
    correlation_data: typing.Optional[str]

    def __init__(self,
                 topic: str,
                 payload: typing.Union[str, bytes],
                 qos: QualityOfService = QualityOfService.AT_MOST_ONCE,
                 retain: bool = False,
                 response_topic: typing.Optional[str] = None,
                 correlation_data: typing.Optional[str] = None,
                 **kwargs
                 ) -> None:
        """Initializes the MQTT Message"""
        super().__init__(**kwargs)
        self.topic = topic
        self.payload = payload if isinstance(payload, str) else payload.decode('utf-8')
        self.qos = qos
        self.retain = retain
        self.response_topic = response_topic
        self.correlation_data = correlation_data

    def load_payload(self) -> typing.Dict[str, typing.Any]:
        """Deserializes the message payload into a dictionary"""
        return json.loads(self.payload)  # type: ignore


class CallbackSignal:
    """ Helper class to signal completion of an message callback"""
    event: threading.Event
    timeout: typing.Optional[float]

    def __init__(self, timeout: typing.Optional[float] = None):
        self.event = threading.Event()
        self.timeout = timeout or SETTINGS.mqtt_timeout

    def wait_for_callback(self) -> bool:
        """ Waits for the callback to be invoked.
        Returns True if the callback was invoked, False if the timeout was reached
        """
        return self.event.wait(timeout=self.timeout)

    def complete_callback(self):
        """ Confirms that the callback was invoked """
        self.event.set()


class MqttManager(cvops.schemas.CooperativeBaseClass):
    """Wrapper for MQTT Client
    """
    client: mqtt.Client
    client_id: str
    device_id: str
    device_secret: str
    mqtt_uri: str
    use_tls: bool
    mqtt_host: str
    mqtt_port: int
    use_autoreconnect: bool
    first_reconnect_delay: int
    reconnect_rate: int
    max_reconnect_count: int
    max_reconnect_delay: int
    connect_callback: typing.Callable[[mqtt.Client, object, dict, int], None]
    on_message_callback: typing.Callable[[mqtt.Client, object, mqtt.MQTTMessage], None]
    on_disconnect_callback: typing.Callable[[mqtt.Client, object, int], None]
    is_connecting: bool

    _subscription_registry: typing.Dict[str, typing.Optional[typing.Callable[[str], None]]]
    _message_queue: typing.List[MqttMessage]
    _connect_actions = typing.Dict[ConnectionResultCodes, typing.Callable[[None], None]]
    _subscription_ids = typing.Dict[int, str]
    _inflight_messages = typing.Dict[int, MqttMessage]
    _is_closing: bool
    _connection_create_count: int

    def __init__(
        self,
        device_id: typing.Optional[str] = None,
        device_secret: typing.Optional[str] = None,
        mqtt_uri: typing.Optional[str] = None,
        use_autoreconnect: bool = True,
        default_qos: QualityOfService = QualityOfService.AT_MOST_ONCE,
        reconnect_delay: int = 1,
        max_reconnect_count: int = 12,
        max_reconnect_delay: int = 60,
        reconnect_rate: int = 2,
        **kwargs
    ) -> None:
        """Initializes the MQTT Manager"""
        super().__init__(**kwargs)
        self.is_connecting = False
        self.mqtt_uri = mqtt_uri or SETTINGS.mqtt_uri
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        url = urlparse(self.mqtt_uri)
        self.client_id = device_id or SETTINGS.device_id
        self.device_id = device_id or SETTINGS.device_id
        self.device_secret = device_secret or SETTINGS.device_secret
        self.use_tls = url.scheme == "mqtts"
        self.mqtt_host = url.hostname or "mqtt.cvops.io"
        self.mqtt_port = 8883 if self.use_tls else 1883
        self.use_autoreconnect = use_autoreconnect
        self.reconnect_rate = reconnect_rate
        self.default_qos = default_qos
        self.max_reconnect_count = max_reconnect_count
        self.max_reconnect_delay = max_reconnect_delay
        self.first_reconnect_delay = reconnect_delay
        self._subscription_registry = {}
        self._message_queue = []
        self._subscription_ids = {}  # type: ignore
        self._inflight_messages = {}  # type: ignore
        self._is_closing = False
        self._connection_create_count = 0

        if self.use_tls:
            self.client.tls_set()

        self._connect_actions = {
            ConnectionResultCodes.CONNACK_ACCEPTED: self.on_connect,
            ConnectionResultCodes.CONNACK_REFUSED_PROTOCOL_VERSION: self._invalid_protocol_version,
            ConnectionResultCodes.CONNACK_REFUSED_BAD_USERNAME_PASSWORD: self._not_authorized,
            ConnectionResultCodes.CONNACK_REFUSED_NOT_AUTHORIZED: self._not_authorized,
            ConnectionResultCodes.CONNACK_REFUSED_SERVER_UNAVAILABLE: self._server_unavailable,
            ConnectionResultCodes.CONNACK_REFUSED_IDENTIFIER_REJECTED: self._invalid_client_id,
        }  # type: ignore

        def on_subscribe_callback(*args):
            """Callback for when the MQTT Client subscribes to a topic"""
            self.on_subscribe(*args)
        self.client.on_subscribe = on_subscribe_callback

        def on_publish_callback(*args):
            """Callback for when the MQTT Client publishes a message"""
            self.on_publish(*args)
        self.client.on_publish = on_publish_callback

        def on_connect_callback(*args):  # pylint: disable=unused-argument
            """Callback for when the MQTT Client connects to the broker"""
            self.on_connect()
        self.client.on_connect = on_connect_callback

        def on_disconnect_callback(*args):
            """Callback for when the MQTT Client disconnects from the broker"""
            self.on_disconnect(*args)
        self.client.on_disconnect = on_disconnect_callback

        def on_message_callback(*args):
            """Callback for when the MQTT Client receives a message"""
            self.on_message(*args)
        self.client.on_message = on_message_callback

        self.client.username_pw_set(self.device_id, self.device_secret)

    @property
    def is_connected(self) -> bool:
        """Returns True if the MQTT Client is connected"""
        return self.client.is_connected()  # type: ignore

    def _start_connection(self) -> None:
        """Connects to the MQTT Broker"""
        try:
            if not self.is_connected and not self.is_connecting:
                self.is_connecting = True
                if self.device_id is None or self.device_secret is None:
                    raise ValueError("Device ID and Device Secret are required to connect to MQTT Broker")
                if self._connection_create_count == 0:
                    logger.info("Connecting to MQTT Broker at %s with device id %s ...", self.mqtt_uri, self.device_id)
                self.client.connect(self.mqtt_host, self.mqtt_port)
        except ValueError as ex:
            logger.error(ex.args[0])
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception("Error connecting to mqtt broker", exc_info=ex)

    def _invalid_protocol_version(self) -> None:
        """ Callback for Invalid MQTT Protocol Version"""
        logger.error("Invalid MQTT protocol version.")
        sys.exit(1)

    def _not_authorized(self) -> None:
        """ Callback for when the MQTT Client is not authorized to connect to the broker"""
        logger.error("MQTT Broker returned not authorized.  Please check your device id and secret and try again.")
        sys.exit(1)

    def _server_unavailable(self) -> None:
        """ Callback for when the MQTT Broker is unavailable"""
        logger.error("MQTT Broker is unavailable." +
                     "Please check your MQTT_URI environment variable settings and try again.")
        sys.exit(1)

    def _invalid_client_id(self) -> None:
        """ Callback for when the MQTT Client ID is invalid"""
        logger.error("Invalid MQTT Client ID." +
                     "Please check your device id or create a new device in the UI and try again.")
        sys.exit(1)

    def on_connect(self) -> None:
        """ Callback for when the MQTT Client connects to the broker"""
        self.is_connecting = False
        self._connection_create_count += 1
        logger.info("Connected to CVOPS MQTT Broker.  Listening for messages...")
        self._flush_subscriptions()
        self._flush_message_queue()

    def on_disconnect(self, client, user_data, result_code, properties) -> None:  # pylint: disable=unused-argument
        """ Callback for when the MQTT Client disconnects from the broker"""
        if self._is_closing:
            self._is_closing = False
            return
        if self.use_autoreconnect:
            reconnect_count = 0
            reconnect_delay = self.first_reconnect_delay
            while reconnect_count < self.max_reconnect_count:
                logging.info("Disconnected. Reconnecting in %d seconds...", reconnect_delay)
                time.sleep(reconnect_delay)
                try:
                    self.client.reconnect()
                    logger.debug("Reconnected to MQTT Broker.")
                    return
                except Exception as ex:  # pylint: disable=broad-except
                    logging.error("%s. Reconnect failed. Retrying...", ex)

                reconnect_delay *= self.reconnect_rate
                reconnect_delay = min(reconnect_delay, self.max_reconnect_delay)
                reconnect_count += 1
            logger.info("Reconnect failed after %s attempts. Exiting...", reconnect_count)

    def listen(self) -> None:
        """ Keep and open connection to the MQTT Broker and listen for messages"""
        try:
            if not self.is_connected and not self.is_connecting:
                self._start_connection()
            self.client.loop_forever()
        except KeyboardInterrupt as ex:
            logger.info("\r\nDisconnecting from MQTT Broker...")
            self.use_autoreconnect = False
            self.client.disconnect()
            raise ex

    @contextlib.contextmanager
    def open(self, timeout: typing.Optional[float] = None) -> typing.Iterator[CallbackSignal]:
        """ Keep an open connection to the MQTT Broker and listen for messages"""
        try:
            if not self.is_connected and not self.is_connecting:
                self._start_connection()
            self.client.loop_start()
            yield CallbackSignal(timeout=timeout)
        except KeyboardInterrupt:
            logger.info("\r\nDisconnecting from MQTT Broker...")
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception("Error connecting to mqtt broker", exc_info=ex)
        finally:
            self.close()

    def close(self) -> None:
        """ Disconnects from the MQTT Broker.  Used after calling open()"""
        self._is_closing = True
        self.client.loop_stop()
        auto = self.use_autoreconnect
        self.use_autoreconnect = False
        self.client.disconnect()
        self.use_autoreconnect = auto

    def on_message(self, client, user_data, message):  # pylint: disable=unused-argument
        """ Default Callback for when the MQTT Client receives a message """
        logger.info("Message Received. Topic %s", message.topic)

    def publish(self, message: MqttMessage):
        """ Publishes a message to an MQTT topic """
        if self.is_connected:
            properties = mqtt.Properties(PacketTypes.PUBLISH)
            if message.response_topic:
                properties.ResponseTopic = message.response_topic
            if message.correlation_data:
                properties.CorrelationData = message.correlation_data
            qos = message.qos if isinstance(message.qos, int) else message.qos.value
            message_info: mqtt.MQTTMessageInfo = self.client.publish(
                message.topic,
                message.payload,
                qos,
                retain=message.retain,
                properties=properties)
            if message_info.rc == 0:
                self._inflight_messages[message_info.mid] = message  # type: ignore
            else:
                raise Exception(  # pylint: disable=broad-exception-raised, raise-missing-from
                    (f"Error publishing message.  Return Code: {message_info.rc}",)
                )
        else:
            self._message_queue.append(message)

    def subscribe(self, topic: str, callback: typing.Optional[typing.Callable[[str], None]] = None):
        """ Subscribes to an MQTT topic

        Args:
            topic (str): The MQTT topic to subscribe to
            callback (Optional[Callable[[str], None]], optional):  Function to invoke when message is received.
                Defaults to None.
        """
        self._subscription_registry[topic] = callback
        self.client.message_callback_add(topic, callback)
        if self.is_connected:
            self._client_subscribe(topic)

    def _client_subscribe(self, topic: str):
        m_ids: typing.Sequence[int] = self.client.subscribe(topic, self.default_qos.value)
        for m_id in m_ids:
            self._subscription_ids[m_id] = topic  # type: ignore

    def on_subscribe(self,
                     client: mqtt.Client,  # pylint: disable=unused-argument
                     user_data: typing.Optional[dict],  # pylint: disable=unused-argument
                     m_id: int,
                     qos: typing.Sequence[int],  # pylint: disable=unused-argument
                     properties: mqtt.Properties  # pylint: disable=unused-argument
                     ) -> None:
        """ Default Callback for when the MQTT Client subscribes to a topic """
        topic = self._subscription_ids.get(m_id)  # type: ignore
        if topic:
            logger.info("Subscribed to topic \"%s\"", topic)
        else:
            logger.error("Subscription ID not found for %s", m_id)

    def _flush_message_queue(self):
        if not self.is_connected:
            return
        while len(self._message_queue) > 0:
            message = self._message_queue.pop(0)
            self.publish(message)

    def _flush_subscriptions(self):
        if not self.is_connected:
            return
        for topic, _ in self._subscription_registry.items():
            self._client_subscribe(topic)

    def unsubscribe(self, topic):
        """ Unsubscribes from an MQTT topic and removes the callback"""
        self.client.unsubscribe(topic)
        self.client.message_callback_remove(topic)
        self._subscription_registry.pop(topic, None)

    def on_publish(self, client: mqtt.Client, user_data: dict, m_id: int) -> None:  # pylint: disable=unused-argument
        """ Default Callback for when the MQTT Client publishes a message """
        message = self._inflight_messages.pop(m_id, None)  # type: ignore
        if message:
            logger.debug("Published message to topic \"%s\"", message.topic)
