"""Deepstream app 1"""
import rootutils

ROOT = rootutils.autosetup()

import argparse
import math
from typing import List

import gi

gi.require_version("Gst", "1.0")
import cv2
import pyds
from gi.repository import GLib, Gst

from src.stream.common.ds_utils import PERF_DATA, bus_call
from src.utils.logger import get_logger

log = get_logger()

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3


class DeepstreamApp3:
    """Deepstream app 1."""

    def __init__(
        self,
        source_uris: List[str],
        pgie_config_path: str,
        requested_pgie: str = "nvinfer",
        disable_probe: bool = False,
        osd_process_mode: int = 0,
        osd_display_text: int = 1,
        no_display: bool = False,
        tiler_shape: tuple = (1280, 720),
        verbose: bool = False,
    ) -> None:
        """Initialize deepstream app 3."""
        assert requested_pgie in ["nvinfer", "nvinferserver", "nvinferserver-grpc"]
        self.source_uris = source_uris
        self.num_sources = len(source_uris)
        self.pgie_config_path = pgie_config_path
        self.requested_pgie = requested_pgie
        self.disable_probe = disable_probe
        self.osd_process_mode = osd_process_mode
        self.osd_display_text = osd_display_text
        self.no_display = no_display
        self.tiler_output_width = tiler_shape[0]
        self.tiler_output_height = tiler_shape[1]
        self.verbose = verbose

        # global variable
        self.file_loop = False
        self.nvdslogger = None
        self.perf_data = PERF_DATA(len(source_uris))

    def run(self) -> None:
        """Run deepstream pipeline."""
        log.info(f"Running deepstream pipeline...")

        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self.loop)
        pgie_src_pad = self.pgie.get_static_pad("src")
        if not pgie_src_pad:
            log.error("Unable to get src pad of pgie")
            RuntimeError("Unable to get src pad of pgie")
        else:
            if not self.disable_probe:
                pgie_src_pad.add_probe(
                    Gst.PadProbeType.BUFFER, self.pgie_src_pad_buffer_probe, 0
                )
                # performance measurement every 5 seconds
                GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        # List the sources
        log.info("Now playing...")
        for i, uri in enumerate(self.source_uris):
            log.info(f"Source {i}: {uri}")

        # start play back and listen to events
        log.info("Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except Exception as e:
            log.error(f"Error: {e}")
            self.pipeline.set_state(Gst.State.NULL)
            raise e

        # cleanup
        log.info("Exiting app...")
        self.pipeline.set_state(Gst.State.NULL)

    def setup(self) -> None:
        """Setup deepstream pipeline."""
        log.info(f"Setup deepstream pipeline...")

        # standard GStreamer initialization
        Gst.init(None)

        # create gstreamer elements
        log.info(f"Creating pipeline...")
        self.pipeline = Gst.Pipeline()
        self.is_live = False
        if not self.pipeline:
            log.error("Unable to create Pipeline")
            RuntimeError("Unable to create Pipeline")

        # create nvstreammux instance to form batches from one or more sources.
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            log.error("Unable to create NvStreamMux")
            RuntimeError("Unable to create NvStreamMux")

        # add streammux to the pipeline
        self.pipeline.add(self.streammux)
        for i in range(self.num_sources):
            log.info(f"Creating source_bin {i}...")
            uri_name = self.source_uris[i]
            if uri_name.find("rtsp://") == 0:
                self.is_live = True
            source_bin = self.create_source_bin(i, uri_name)
            if not source_bin:
                log.error("Unable to create source bin")
                RuntimeError("Unable to create source bin")
            self.pipeline.add(source_bin)
            padname = "sink_%u" % i
            sinkpad = self.streammux.get_request_pad(padname)
            if not sinkpad:
                log.error("Unable to create sink pad bin")
                RuntimeError("Unable to create sink pad bin")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                log.error("Unable to create src pad bin")
                RuntimeError("Unable to create src pad bin")
            srcpad.link(sinkpad)

        # create queue to store metadata
        self.queue1 = Gst.ElementFactory.make("queue", "queue1")
        self.queue2 = Gst.ElementFactory.make("queue", "queue2")
        self.queue3 = Gst.ElementFactory.make("queue", "queue3")
        self.queue4 = Gst.ElementFactory.make("queue", "queue4")
        self.queue5 = Gst.ElementFactory.make("queue", "queue5")
        self.pipeline.add(self.queue1)
        self.pipeline.add(self.queue2)
        self.pipeline.add(self.queue3)
        self.pipeline.add(self.queue4)
        self.pipeline.add(self.queue5)

        # create pgie to run inference
        log.info(f"Creating pgie element...")
        if self.requested_pgie == "nvinfer":
            self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        elif (
            self.requested_pgie == "nvinferserver"
            or self.requested_pgie == "nvinferserver-grpc"
        ):
            self.pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
        if not self.pgie:
            log.error("Unable to create pgie")
            RuntimeError("Unable to create pgie")

        # check probe config
        if self.disable_probe:
            # use nvdslogger for performance measurement instead of probe
            log.info(f"Creaing nvdslogger...")
            self.nvdslogger = Gst.ElementFactory.make("nvdslogger", "nvdslogger")

        # create tiler to render boxes
        log.info(f"Creating tiler element...")
        self.tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not self.tiler:
            log.error("Unable to create tiler")
            RuntimeError("Unable to create tiler")

        # create nvvidconv to convert RGBA to RGB
        log.info(f"Creating nvvidconv element...")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not self.nvvidconv:
            log.error("Unable to create nvvidconv")
            RuntimeError("Unable to create nvvidconv")

        # create nvosd to draw on the converted RGBA buffer
        log.info(f"Creating nvosd element...")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not self.nvosd:
            log.error("Unable to create nvosd")
            RuntimeError("Unable to create nvosd")
        self.nvosd.set_property("process-mode", self.osd_process_mode)
        self.nvosd.set_property("display-text", self.osd_display_text)

        if self.file_loop:
            self.streammux.set_property("nvbuf-memory-type", 2)

        # check for display option
        if self.no_display:
            log.info(f"Creating fakesink element...")
            self.sink = Gst.ElementFactory.make("fakesink", "fakesink")
            self.sink.set_property("sync", 0)
            self.sink.set_property("enable-last-sample", 0)
        else:
            log.info(f"Creating EGLSink element...")
            self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not self.sink:
                log.error("Unable to create egl sink")
                RuntimeError("Unable to create egl sink")
        if not self.sink:
            log.error("Unable to create sink")
            RuntimeError("Unable to create sink")

        if self.is_live:
            log.info("At least one of the sources is live")
            self.streammux.set_property("live-source", 1)

        # set properties of streammux
        # TODO: set width and height
        width, height, fps = self.get_source_properties()
        self.streammux.set_property("width", width)
        self.streammux.set_property("height", height)
        self.streammux.set_property("batch-size", self.num_sources)
        self.streammux.set_property("batched-push-timeout", 4000000)  # 4 seconds

        # set properties of pgie
        self.pgie.set_property("config-file-path", self.pgie_config_path)
        pgie_batch_size = self.pgie.get_property("batch-size")
        if pgie_batch_size != self.num_sources:
            log.warning(
                f"Overriding infer-config batch-size {pgie_batch_size} with {self.num_sources}"
            )
            self.pgie.set_property("batch-size", self.num_sources)

        # set properties of tiler
        tiler_rows = int(math.sqrt(self.num_sources))
        tiler_columns = int(math.ceil((1.0 * self.num_sources) / tiler_rows))
        self.tiler.set_property("rows", tiler_rows)
        self.tiler.set_property("columns", tiler_columns)
        self.tiler.set_property("width", self.tiler_output_width)
        self.tiler.set_property("height", self.tiler_output_height)
        self.sink.set_property("qos", 0)

        # add elements to pipeline
        log.info(f"Adding elements to Pipeline...")
        self.pipeline.add(self.pgie)
        if self.nvdslogger:
            self.pipeline.add(self.nvdslogger)
        self.pipeline.add(self.tiler)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        # link the elements together
        log.info(f"Linking elements in the Pipeline...")
        self.streammux.link(self.queue1)
        self.queue1.link(self.pgie)
        self.pgie.link(self.queue2)
        if self.nvdslogger:
            self.queue2.link(self.nvdslogger)
            self.nvdslogger.link(self.tiler)
        else:
            self.queue2.link(self.tiler)
        self.tiler.link(self.queue3)
        self.queue3.link(self.nvvidconv)
        self.nvvidconv.link(self.queue4)
        self.queue4.link(self.nvosd)
        self.nvosd.link(self.queue5)
        self.queue5.link(self.sink)

    def get_source_properties(self) -> tuple:
        """
        Get source property. Like width, height, fps, etc.
        """
        cap = cv2.VideoCapture(self.source_uris[0])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return width, height, fps

    def create_source_bin(self, index: int, uri: str):
        """Create source bin."""
        log.info(f"Creating source bin ({index}) {uri}...")

        # create a source GstBin to abstract this bin's content from the rest of the pipeline
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            log.error(f"Unable to create source bin {bin_name}")
            RuntimeError(f"Unable to create source bin {bin_name}")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        if self.file_loop:
            # use nvurisrcbin to enable file loop
            uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
            uri_decode_bin.set_property("file-loop", 1)
            uri_decode_bin.set_property("cudadec-memtype", 0)
        else:
            uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            log.error("Unable to create uri decode bin")
            RuntimeError("Unable to create uri decode bin")

        # we set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)

        # connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added", self.decodebin_pad_added_handler, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added_handler, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            log.error("Failed to add ghost pad in source bin")
            RuntimeError("Failed to add ghost pad in source bin")
            return None

        return nbin

    def decodebin_pad_added_handler(self, decodebin, decoder_src_pad, data):
        """Decodebin pad added handler."""
        log.info(f"In decodebin_pad_added_handler...")
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            caps = decoder_src_pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        # check if the pad created by the decodebin is for video and not audio
        log.info(f"Gstname: {gstname}")
        if gstname.find("video") != -1:
            # link the decodebin pad only if decodebin has picked nvidia
            # decoder plugin nvdec_*. we do this by checking if the pad caps contain
            # NVMM memory features.
            log.info(f"Features: {features}")
            if features.contains("memory:NVMM"):
                # get the source bin ghost pad
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    log.error("Failed to link decoder src pad to source bin ghost pad")
                    RuntimeError(
                        "Failed to link decoder src pad to source bin ghost pad"
                    )
            else:
                log.error("Error: Decodebin did not pick nvidia decoder plugin.")
                RuntimeError("Error: Decodebin did not pick nvidia decoder plugin.")

    def decodebin_child_added_handler(self, child_proxy, Object, name, user_data):
        """Decodebin child added handler."""
        log.info(f"Dynamically created element {name}")
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added_handler, user_data)

        if "source" in name:
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property("drop-on-latency") != None:
                Object.set_property("drop-on-latency", True)

    def pgie_src_pad_buffer_probe(self, pad, info, u_data):
        """Probe to get pgie src pad buffer."""
        frame_number = 0
        num_rects = 0
        got_fps = False
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return
        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            l_obj = frame_meta.obj_meta_list
            num_rects = frame_meta.num_obj_meta
            obj_counter = {
                PGIE_CLASS_ID_VEHICLE: 0,
                PGIE_CLASS_ID_PERSON: 0,
                PGIE_CLASS_ID_BICYCLE: 0,
                PGIE_CLASS_ID_ROADSIGN: 0,
            }
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                obj_counter[obj_meta.class_id] += 1
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            if self.verbose:
                log.info(
                    f"Frame Number={frame_number} | Number of Objects={num_rects} | Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]} | Person_count={obj_counter[PGIE_CLASS_ID_PERSON]}"
                )

            # Update frame rate through this probe
            stream_index = "stream{0}".format(frame_meta.pad_index)
            # global perf_data
            self.perf_data.update_fps(stream_index)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Deepstream app 3")
    parser.add_argument(
        "--source_uris",
        nargs="+",
        default=[
            "rtsp://admin:widya123@192.168.0.64:554/Streaming/channels/101",
            "rtsp://admin:011118widya@192.168.111.188:554/Streaming/channels/401",
        ],
        help="List of sources, separated by space",
    )
    parser.add_argument(
        "--pgie_config_path",
        type=str,
        default="configs/pgies/app3_pgie_config.txt",
        help="Path to pgie config file",
    )
    parser.add_argument(
        "--requested_pgie",
        type=str,
        default="nvinfer",
        choices=["nvinfer", "nvinferserver", "nvinferserver-grpc"],
        help="Requested pgie",
    )
    parser.add_argument(
        "--disable_probe",
        action="store_true",
        default=False,
        help="Disable probe",
    )
    parser.add_argument(
        "--osd_process_mode",
        type=int,
        default=0,
        help="OSD process mode",
    )
    parser.add_argument(
        "--osd_display_text",
        type=int,
        default=1,
        help="OSD display text",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        default=False,
        help="No display",
    )
    parser.add_argument(
        "--tiler_shape",
        nargs="+",
        type=int,
        default=(1280, 720),
        help="Tiler shape",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose",
    )
    args = parser.parse_args()

    app = DeepstreamApp3(
        args.source_uris,
        args.pgie_config_path,
        args.requested_pgie,
        args.disable_probe,
        args.osd_process_mode,
        args.osd_display_text,
        args.no_display,
        args.tiler_shape,
        args.verbose,
    )
    app.setup()
    app.run()
