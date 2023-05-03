
import os
import logging

from ..common.consts import LOG_FORMAT


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def get_backend_name() -> str:
    """Fetch backend from environment variable"""

    backend_name = os.getenv('CASTLE_BACKEND')
    if backend_name not in ['pytorch', 'mindspore', None]:
        raise TypeError("Please use ``os.environ[CASTLE_BACKEND] = backend`` "
                        "to set backend environment variable to `pytorch` or "
                        "`mindspore`.")
    if backend_name is None:
        backend_name = 'pytorch'
        logging.info(
            "You can use `os.environ['CASTLE_BACKEND'] = backend` to set the "
            "backend(`pytorch` or `mindspore`).")

    return backend_name


backend = get_backend_name()
