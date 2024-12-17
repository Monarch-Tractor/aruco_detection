"""
This script contains the implementation of the
Logger class which is used to create a logger object
for different modules.
"""


import logging


class Logger:
    """
    This class defines the Logger class which is used
    to create a logger object for different modules.
    """
    @staticmethod
    def get_logger(name, level):
        """
        This class method returns a logger object.
        """
        time_format = "%m/%d/%Y-%H:%M:%S"
        formatter = logging.Formatter(
            fmt="[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s",
            datefmt=time_format
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(level)  # Set logging level.
        logger.propagate = False

        return logger
    