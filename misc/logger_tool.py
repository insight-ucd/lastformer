import sys
import time
import copy
import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)

# Create a new logger
logger = logging.getLogger("LASTFormer")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# Configure logger
#loglevel = getattr(logging, "INFO", None)
logger.setLevel("INFO")


class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

