"""
Module for a general Logger class
"""
import os
import sys
import time
from time import time
import datetime
import logging
import psutil
from compdisteval.code.util.paths import joinPaths, logsFolder


class Logger(object):
    """The Logger Class"""

    def __init__(self, fname):
        logFileName = os.path.join(logsFolder, fname)
        self.file = open(logFileName, 'w')

    def logit(self, msg):
        """Log a message with a new line."""
        t = datetime.datetime.fromtimestamp(time.time())
        self.file.write("%s: %s\n" % (t, msg))
        self.file.flush()
        sys.stdout.write("%s\n" % msg)

    def logit_nobreak(self, msg):
        """Log a message but without a new line."""
        t = datetime.datetime.fromtimestamp(time.time())
        self.file.write("%s: %s\t" % (t, msg))
        self.file.flush()
        sys.stdout.write("%s\t" % msg)

    def logit_flush(self, msg):
        """Log a message with a new line, but flush the stdout."""
        t = datetime.datetime.fromtimestamp(time.time())
        self.file.write("%s: %s\n" % (t, msg))
        self.file.flush()
        sys.stdout.write("%s\r" % msg)
        sys.stdout.flush()

    def logit_noprint(self, msg):
        """Log a message without writing to stdout."""
        t = datetime.datetime.fromtimestamp(time.time())
        self.file.write("%s: %s\n" % (t, msg))
        self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.close()


def setupLogging(logFileName):
    """Sets up logging to a (fresh) file and to console."""
    logFilePath = joinPaths(logsFolder, logFileName)
    if os.path.isfile(logFilePath):
        os.remove(logFilePath)
    logging.basicConfig(filename=logFilePath, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def logMemory():
    process = psutil.Process(os.getpid())
    used = process.memory_info().rss
    available = float(os.popen('free -b').read().split('\n')[1].split()[1])
    percent_used = 100*used/available
    logging.info("Memory usage: %.2f%%", percent_used)


def logMemoryReturn():
    process = psutil.Process(os.getpid())
    used = process.memory_info().rss
    available = float(os.popen('free -b').read().split('\n')[1].split()[1])
    percent_used = 100*used/available
    logging.info("Memory usage: %.2f%%", percent_used)
    return percent_used


def logETA(start, steps, total_steps):
    now = time()
    tte = (now - start)
    tte_format = ('%d:%02d:%02d' %
                  (tte // 3600, (tte % 3600) // 60, tte % 60))
    time_per_unit = tte / steps
    eta = time_per_unit * (total_steps - steps)
    eta_format = ('%d:%02d:%02d' %
                  (eta // 3600, (eta % 3600) // 60, eta % 60))
    logging.info('Time elapsed: %s - ETA: %s', tte_format, eta_format)

# Currently used memory
# import psutil
# process = psutil.Process(os.getpid())
# used = process.memory_info().rss
# Available memory in bytes
# available = float(os.popen('free -b').read().split('\n')[1].split()[-1])
# percent_used = used/available
