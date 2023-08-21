"""Deepstream utils."""

import rootutils

ROOT = rootutils.autosetup()

import sys
import time
from threading import Lock

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.utils.logger import get_logger

log = get_logger()

start_time = time.time()
fps_mutex = Lock()


def bus_call(bus, message, loop) -> bool:
    """Callback function for bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


class GETFPS:
    """Get FPR for each stream."""

    def __init__(self, stream_id):
        global start_time
        self.start_time = start_time
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id

    def update_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        else:
            global fps_mutex
            with fps_mutex:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        end_time = time.time()
        with fps_mutex:
            stream_fps = float(self.frame_count / (end_time - self.start_time))
            self.frame_count = 0
        self.start_time = end_time
        return round(stream_fps, 2)

    def print_data(self):
        """Print fps data."""
        log.info(f"frame_count={self.frame_count}")
        log.info(f"start_time={self.start_time}")


class PERF_DATA:
    """Get performance data for each stream."""

    def __init__(self, num_streams=1):
        self.perf_dict = {}
        self.all_stream_fps = {}
        for i in range(num_streams):
            self.all_stream_fps["stream{0}".format(i)] = GETFPS(i)

    def perf_print_callback(self):
        self.perf_dict = {
            stream_index: stream.get_fps()
            for (stream_index, stream) in self.all_stream_fps.items()
        }
        log.info(f"PERF: {self.perf_dict}\n")
        return True

    def update_fps(self, stream_index):
        self.all_stream_fps[stream_index].update_fps()
