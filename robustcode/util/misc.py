import logging
import os
import time

from sty import bg
from sty import rs


def is_file_empty(filename):
    return os.stat(filename).st_size == 0


def file_size_in_mb(filename):
    return int(os.stat(filename).st_size / 1024 / 1024)


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


class Timer:
    def __init__(self, name=None):
        self.name = name
        self.total_time = 0
        self.start_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.total_time += time.time() - self.start_time

    def __repr__(self):
        return "{:>20s} {:.2f} s".format(self.name or "", self.total_time)


class Logger:
    """
    Helper class to measure execution time between calling `start` and `end`
    """

    active_timers = []

    @staticmethod
    def start_scope(name):
        logging.debug("{}{}...".format(len(Logger.active_timers) * "\t", name))
        Logger.active_timers.append(time.time())

    @staticmethod
    def end_scope():
        assert Logger.active_timers
        start = Logger.active_timers.pop()
        runtime = time.time() - start
        logging.debug(
            "{}Done in {:.2f} s".format(len(Logger.active_timers) * "\t", runtime)
        )
        return runtime

    @staticmethod
    def debug(desc):
        logging.debug("{}{}".format(len(Logger.active_timers) * "\t", desc))

    @staticmethod
    def warn(desc):
        logging.warning("{}{}".format(len(Logger.active_timers) * "\t", desc))

    @staticmethod
    def init(log_file=None):
        logger = logging.getLogger()

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        if log_file is not None:
            if os.path.exists(log_file):
                os.remove(log_file)
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def trim(word, length=20):
    if word is None:  # or not word:
        return " "
    if word == "<null>":
        return ""
    word = str(word)
    if any(c.isspace() for c in word):
        word = repr(word)
    if len(word) <= length:
        return word
    return word[: length - 2] + ".."


def word(word, field):
    if not hasattr(field, "vocab"):
        return "{:>20s}".format(trim(str(word), 20))
    if field.vocab.stoi[word] != 0:
        return "{:>20s}".format(trim(word, 20))
    return bg("da_black") + "{:>20s}".format(trim(word, 20)) + rs.bg


def word_bg(word, color="da_red"):
    return bg(color) + word + rs.bg


def marker(correct):
    return bg("da_green" if correct else "da_red") + " " + rs.bg


def acc(count, total):
    if total == 0:
        return 0
    return count * 100.0 / total
