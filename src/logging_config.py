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
    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        orig_name = record.name

        record.levelname = f"{COLORS.get(record.levelname, '')}{record.levelname}{COLORS['RESET']}"
        record.name = f"{COLORS['NAME']}{record.name}{COLORS['RESET']}"

        message = super().format(record)

        record.levelname = orig_levelname
        record.name = orig_name

        message = message.replace(record.asctime, f"{COLORS['TIME']}{record.asctime}{COLORS['RESET']}", 1)

        message_start = message.rfind('-') + 2
        message_color = COLORS.get(f"{record.levelname}_MESSAGE", COLORS['RESET'])
        message = (
                message[:message_start] +
                f"{message_color}{message[message_start:]}{COLORS['RESET']}"
        )

        return message


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