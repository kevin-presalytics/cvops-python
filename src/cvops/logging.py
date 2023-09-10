""" CVOps Logging Defaults"""
import sys
import typing
import logging
import logging.config


def configure_logging(log_level: typing.Union[str, int] = logging.INFO):
    """ Configures logging for the SDK """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'},
            'simple': {'format': '%(message)s'}
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'default' if log_level < logging.INFO else 'simple',
            },
        },
        'loggers': {
            'cvops': {
                'level': log_level,
                'handlers': ['console']
            }
        },
        'disable_existing_loggers': False
    })


def handle_exception(exc_type, exc_value, exc_traceback) -> None:
    """ Catches unhandled exceptions for logger """
    logger = logging.getLogger(__name__)
    try:
        logger.error("--------------------------")
        logger.error("UNHANDLED EXCEPTION: FATAL")
        logger.error(msg="--------------------------", exc_info=(exc_type, exc_value, exc_traceback,))
        logger.error("--------------------------")
    except Exception:  # pylint: disable=broad-except
        pass
    return


# users can override the except hook, but most analysts aren't that skilled yet.
# Makes their scripts more verbose without having to think about it
sys.excepthook = handle_exception  # type: ignore
