import logging
import logging.handlers
import sys
from typing import Optional

COLORS = {
    'RESET': '\033[0m',
    'INFO': '\033[38;2;114;195;253m',
    'INFO_MESSAGE': '\033[38;2;194;226;255m',
    'DEBUG': '\033[38;2;142;156;189m',
    'DEBUG_MESSAGE': '\033[38;2;164;174;196m',
    'WARNING': '\033[38;2;255;191;105m',
    'WARNING_MESSAGE': '\033[38;2;255;191;105m',
    'ERROR': '\033[38;2;255;129;123m',
    'ERROR_MESSAGE': '\033[38;2;255;129;123m',
    'CRITICAL': '\033[48;2;255;129;123m\033[38;2;255;255;255m',
    'CRITICAL_MESSAGE': '\033[48;2;255;129;123m\033[38;2;255;255;255m',
    'NAME': '\033[38;2;152;195;121m',
    'TIME': '\033[38;2;198;198;198m',
}


class ColoredFormatter(logging.Formatter):
    """
    A formatter that constructs the log line manually and injects color codes.
    """

    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        # We let the base class handle storing fmt/datefmt
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # 1. Let the base class set up 'record.asctime' via formatTime if we have %(asctime)s in self._fmt
        #    or we can just call formatTime manually if we want full control:
        if self.datefmt:
            record.asctime = self.formatTime(record, self.datefmt)
        else:
            record.asctime = self.formatTime(record)

        # 2. Prepare the original message
        message = record.getMessage()

        # 3. Colorize each piece (only if self.use_color is True)
        if self.use_color:
            level_color = COLORS.get(record.levelname, COLORS['RESET'])
            msg_color = COLORS.get(f"{record.levelname}_MESSAGE", COLORS['RESET'])
            levelname_colored = f"{level_color}{record.levelname}{COLORS['RESET']}"
            name_colored = f"{COLORS['NAME']}{record.name}{COLORS['RESET']}"
            time_colored = f"{COLORS['TIME']}{record.asctime}{COLORS['RESET']}"
            message_colored = f"{msg_color}{message}{COLORS['RESET']}"
        else:
            # No color: just use the plain text
            levelname_colored = record.levelname
            name_colored = record.name
            time_colored = record.asctime
            message_colored = message

        # 4. Construct the final single-line string
        #    This example uses the same structure as "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_line = f"{time_colored} - {name_colored} - {levelname_colored} - {message_colored}"

        # 5. Return the fully colorized string
        return log_line


def setup_logging(
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Setup logging with modern colored output for console and optional file logging.

    Args:
        log_level: The logging level to use
        log_file: Optional file path for logging to file
        log_format: Format string for log messages
    """
    # Remove any existing root handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter(log_format))
    #console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    logging.root.setLevel(log_level)
    logging.root.addHandler(console_handler)

    if log_file is not None:
        # If user provided a log file, add rotating file handler (without colors)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.root.addHandler(file_handler)


# Testing the logger
if __name__ == "__main__":
    setup_logging(log_level=logging.DEBUG)
    logger = logging.getLogger("TestLogger")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")