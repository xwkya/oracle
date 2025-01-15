import logging
import logging.handlers
import sys

def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
):
    # Remove any existing root handlers (important if you run multiple scripts in the same process, e.g. tests)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout
    )

    if log_file:
        # If user provided a log file, add rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)