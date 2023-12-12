""" Module containing event classes sent and received from MQTT broker """
import datetime
import abc
import typing
import enum
import sys
import pydantic
import pydantic.alias_generators
import cvops.config
import cvops.device
import cvops.mqtt
import cvops.schemas


class EventTypes(enum.Enum):
    """ Enum of all event types """
    DEVICE_DETAILS_REQUEST = "device.details_request"
    DEVICE_DETAILS_RESPONSE = "device.details_response"
    DEVICE_DETAILS_UPDATE = "device.details_update"
    DEVICE_REGISTERED = "device.registered"
    DEVICE_UNREGISTERED = "device.unregistered"
    WORKSPACE_DETAILS_REQUEST = "workspace.details_request"
    WORKSPACE_DETAILS_RESPONSE = "workspace.details_response"
    DEPLOYMENT_CREATED = "deployment.created"
    DEPLOYMENT_UPDATED = "deployment.updated"
    DEPLOYMENT_DELETED = "deployment.deleted"


AnyEvent = typing.TypeVar('AnyEvent', bound='BaseEvent')


class BaseEvent(pydantic.BaseModel, abc.ABC, typing.Generic[AnyEvent]):
    """ Base class for all MQTT events """
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel)
    time: datetime.datetime = pydantic.Field(default_factory=cvops.schemas.now)
    device_id: str = cvops.config.SETTINGS.device_id
    workspace_id: typing.Optional[str] = None
    user_id: typing.Optional[str] = None
    event_type: EventTypes
    event_data: dict = {}
    response_topic: typing.Optional[str] = None
    manager: typing.Optional[typing.Any] = None

    @abc.abstractmethod
    def deserialize_event_data(self) -> typing.Any:
        """ Deserialize the event_data json into a python object """
        return NotImplemented

    def to_json(self):
        """ Returns the event as a json string """
        return self.model_dump_json(by_alias=True, exclude_none=True)

    _EVENT_CLASSES: typing.ClassVar[typing.List] = []

    @classmethod
    def get_event_classes(cls) -> typing.Sequence[AnyEvent]:
        """ Returns all event classes in this module """
        if len(cls._EVENT_CLASSES) == 0:
            for _, obj in sys.modules[cls.__module__].__dict__.items():
                if isinstance(obj, type) and issubclass(obj, cls) and obj != cls:
                    if not cls._EVENT_CLASSES:
                        cls._EVENT_CLASSES = []
                    cls._EVENT_CLASSES.append(obj)
        return cls._EVENT_CLASSES

    @classmethod
    def get_event_type(cls) -> EventTypes:
        """ Returns all event types in this module """
        return cls.model_fields['event_type'].default  # type: ignore

    @classmethod
    def factory(cls: AnyEvent, message: 'cvops.mqtt.MqttMessage', manager: 'cvops.device.DeviceManager' = None) -> AnyEvent:  # type: ignore
        """ Factory method for creating an event """
        payload = message.load_payload()
        event_type = payload.get('eventType', None)
        if event_type is None:
            raise ValueError("Event must have an eventType")
        for klass in cls.get_event_classes():  # type: ignore
            if klass.get_event_type().value == event_type:
                return klass(**payload, manager=manager)  # type: ignore
        raise ValueError(f"Unknown event type: {event_type}")

    def get_device_manager(self) -> typing.Optional['cvops.device.DeviceManager']:
        """ Returns the device manager for this event """
        return self.manager  # type: ignore


class EventCallbackWrapper(pydantic.BaseModel):
    """ Wrapper for event callbacks """
    index: int
    callback: typing.Callable[[AnyEvent], None]
    event_type: EventTypes
    disconnect_after_callback: bool = False


class EventManager(cvops.mqtt.MqttManager):
    """ Class for managing event callbacks"""
    _callbacks: typing.List[EventCallbackWrapper]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = []

    def _invoke_event_callbacks(self, event: AnyEvent) -> None:
        """ Invokes all event callbacks for the specified event type """
        disconnect = False
        for callback in self._callbacks:
            if callback.event_type == event.event_type:
                callback.callback(event)
                if callback.disconnect_after_callback:
                    disconnect = True
        if disconnect:
            self.close()

    def set_event_callback(
            self,
            event_type: EventTypes,
            callback: typing.Callable[[AnyEvent], None],
            disconnect_after_callback: bool = False
    ) -> int:
        """ Sets an event callback for the specified event type.  Returns the callback index. """
        if isinstance(event_type, str):
            event_type = cvops.events.EventTypes(event_type)
        max_index = max([x.index for x in self._callbacks]) if self._callbacks else 0  # pylint: disable=consider-using-generator
        wrapper = EventCallbackWrapper(
            index=max_index + 1,
            callback=callback,
            event_type=event_type,
            disconnect_after_callback=disconnect_after_callback
        )
        self._callbacks.append(wrapper)
        return wrapper.index

    def remove_event_callback(self, index: int) -> None:
        """ Removes an event callback by index """
        self._callbacks = [x for x in self._callbacks if x.index != index]


class DeviceDetailsRequestEvent(BaseEvent):
    """ Event for requesting device info """
    event_type: EventTypes = EventTypes.DEVICE_DETAILS_REQUEST
    event_data: typing.Any = {}

    def deserialize_event_data(self) -> typing.Any:
        return self.event_data


class DeviceDetailsResponseEvent(BaseEvent):
    """ Event returned by hub containing Device Model"""
    event_type: EventTypes = EventTypes.DEVICE_DETAILS_RESPONSE

    def deserialize_event_data(self) -> 'cvops.schemas.Device':
        return cvops.schemas.Device(**self.event_data)


class DeviceDetailsUpdateEvent(BaseEvent):
    """ Event to update the device information stored on the hub"""
    event_type: EventTypes = EventTypes.DEVICE_DETAILS_UPDATE

    def deserialize_event_data(self) -> 'cvops.schemas.Device':
        return cvops.schemas.Device(**self.event_data)


class DeviceRegisteredEvent(BaseEvent):
    """ Event sent by device when it is registered to the hub"""
    event_type: EventTypes = EventTypes.DEVICE_REGISTERED

    def deserialize_event_data(self) -> 'cvops.schemas.Device':
        return cvops.schemas.Device(**self.event_data)


class WorkspaceDetailsRequestEvent(BaseEvent):
    """ Event for requesting workspace info """
    event_type: EventTypes = EventTypes.WORKSPACE_DETAILS_REQUEST
    event_data: typing.Any = {}

    def deserialize_event_data(self) -> typing.Any:
        return self.event_data


class WorkspaceDetailsResponseEvent(BaseEvent):
    """ Event returned by hub containing Workspace Model"""
    event_type: EventTypes = EventTypes.WORKSPACE_DETAILS_RESPONSE

    def deserialize_event_data(self) -> 'cvops.schemas.Workspace':
        return cvops.schemas.Workspace(**self.event_data)


class DeploymentCreatedEvent(BaseEvent):
    """ Event sent by hub when a deployment is created"""
    event_type: EventTypes = EventTypes.DEPLOYMENT_CREATED

    def deserialize_event_data(self) -> 'cvops.schemas.Deployment':
        return cvops.schemas.Deployment(**self.event_data)


class DeploymentUpdatedEvent(BaseEvent):
    """ Event sent by hub when a deployment is updated"""
    event_type: EventTypes = EventTypes.DEPLOYMENT_UPDATED

    def deserialize_event_data(self) -> 'cvops.schemas.Deployment':
        return cvops.schemas.Deployment(**self.event_data)


class DeploymentDeletedEvent(BaseEvent):
    """ Event sent by hub when a deployment is deleted"""
    event_type: EventTypes = EventTypes.DEPLOYMENT_DELETED

    def deserialize_event_data(self) -> 'cvops.schemas.Deployment':
        return cvops.schemas.Deployment(**self.event_data)
