"""Deepstream app 1"""

import rootutils

ROOT = rootutils.autosetup()

import argparse
import configparser

import gi

gi.require_version("Gst", "1.0")
import cv2
import pyds
from gi.repository import GLib, Gst

from src.stream.common.ds_utils import bus_call
from src.utils.logger import get_logger

log = get_logger()

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3


class DeepstreamApp2:
    """Deepstream app 1."""

    def __init__(
        self,
        source_file: str,
        pgie_config_path: str,
        sgie1_config_path: str,
        sgie2_config_path: str,
        sgie3_config_path: str,
        tracker_config_path: str,
    ) -> None:
        """Initialize deepstream app 1."""
        self.source_file = source_file
        self.pgie_config_path = pgie_config_path
        self.sgie1_config_path = sgie1_config_path
        self.sgie2_config_path = sgie2_config_path
        self.sgie3_config_path = sgie3_config_path
        self.tracker_config_path = tracker_config_path

    def run(self) -> None:
        """Run deepstream pipeline."""
        log.info(f"Running deepstream pipeline...")

        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self.loop)

        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        osdsinkpad = self.nvosd.get_static_pad("sink")
        if not osdsinkpad:
            log.error("Unable to get sink pad of nvosd")
            RuntimeError("Unable to get sink pad of nvosd")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        # start play back and listen to events
        log.info(f"Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except Exception as e:
            log.error(f"Exception: {e}")
            self.pipeline.set_state(Gst.State.NULL)
            raise e

        # cleanup
        log.info(f"Stopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)

    def setup(self) -> None:
        """Setup deepstream pipeline."""
        log.info(f"Setup deepstream pipeline...")

        # standard GStreamer initialization
        Gst.init(None)

        # create gstreamer elements
        log.info(f"Creating pipeline...")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            log.error("Unable to create Pipeline")
            RuntimeError("Unable to create Pipeline")

        # create source element for reading from the file
        log.info(f"Creating source element...")
        self.source = Gst.ElementFactory.make("filesrc", "file-source")
        if not self.source:
            log.error("Unable to create Source")
            RuntimeError("Unable to create Source")

        # create h264parser
        log.info(f"Creating h264parser element...")
        self.h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not self.h264parser:
            log.error("Unable to create h264parser")
            RuntimeError("Unable to create h264parser")

        # create decoder with nvdec_h264 for hardware decoding with GPU
        log.info(f"Creating decoder element...")
        self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not self.decoder:
            log.error("Unable to create decoder")
            RuntimeError("Unable to create decoder")

        # create streammux instance to form batches from one or more sources
        log.info(f"Creating streammux element...")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            log.error("Unable to create streammux")
            RuntimeError("Unable to create streammux")

        # Use nvinfer to run inferencing on decoder's output,
        # behaviour of inferencing is set through config file
        log.info(f"Creating pgie element...")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not self.pgie:
            log.error("Unable to create pgie")
            RuntimeError("Unable to create pgie")

        # Use nvtracker to give objects unique-ids
        log.info(f"Creating tracker element...")
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not self.tracker:
            log.error("Unable to create tracker")
            RuntimeError("Unable to create tracker")

        # Use nvinfer for secondary classifier
        log.info(f"Creating sgie1 element...")
        self.sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
        if not self.sgie1:
            log.error("Unable to create sgie1")
            RuntimeError("Unable to create sgie1")

        self.sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
        if not self.sgie2:
            log.error("Unable to create sgie2")
            RuntimeError("Unable to create sgie2")

        self.sgie3 = Gst.ElementFactory.make("nvinfer", "secondary3-nvinference-engine")
        if not self.sgie3:
            log.error("Unable to create sgie3")
            RuntimeError("Unable to create sgie3")

        # Use convertor to convert from NV12 to RGBA as required by nvosd
        log.info(f"Creating nvvidconv element...")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not self.nvvidconv:
            log.error("Unable to create nvvidconv")
            RuntimeError("Unable to create nvvidconv")

        # Create OSD to draw on the converted RGBA buffer
        log.info(f"Creating OSD element...")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not self.nvosd:
            log.error("Unable to create nvosd")
            RuntimeError("Unable to create nvosd")

        # Finally render the osd output
        log.info(f"Creating sink element...")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not self.sink:
            log.error("Unable to create sink")
            RuntimeError("Unable to create sink")

        # Get width, height, fps of the source
        width, height, fps = self.get_source_properties()

        # Set properties of the elements
        log.info(f"Setting properties of elements...")
        self.source.set_property("location", self.source_file)
        self.streammux.set_property("batch-size", 1)
        self.streammux.set_property("width", width)
        self.streammux.set_property("height", height)
        self.streammux.set_property("batched-push-timeout", 4000000)  # 4 seconds

        # Set properties of pgie and sgies
        self.pgie.set_property("config-file-path", self.pgie_config_path)
        self.sgie1.set_property("config-file-path", self.sgie1_config_path)
        self.sgie2.set_property("config-file-path", self.sgie2_config_path)
        self.sgie3.set_property("config-file-path", self.sgie3_config_path)

        # Set properties of tracker
        self.set_tracker_properties()

        # Add elements to the pipeline
        log.info(f"Adding elements to the pipeline...")
        self.pipeline.add(self.source)
        self.pipeline.add(self.h264parser)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.sgie1)
        self.pipeline.add(self.sgie2)
        self.pipeline.add(self.sgie3)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        # Link the elements together
        log.info(f"Linking elements in the pipeline...")
        self.source.link(self.h264parser)
        self.h264parser.link(self.decoder)

        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            log.error("Unable to get the sink pad of streammux")
            RuntimeError("Unable to get the sink pad of streammux")
        srcpad = self.decoder.get_static_pad("src")
        if not srcpad:
            log.error("Unable to get source pad of decoder")
            RuntimeError("Unable to get source pad of decoder")
        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.sgie1)
        self.sgie1.link(self.sgie2)
        self.sgie2.link(self.sgie3)
        self.sgie3.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)

        log.info(f"Deepstream pipeline created successfully!")

    def set_tracker_properties(self) -> None:
        """Set properties of tracker."""
        config = configparser.ConfigParser()
        config.read(self.tracker_config_path)
        config.sections()

        for key in config["tracker"]:
            if key == "tracker-width":
                tracker_width = config.getint("tracker", key)
                self.tracker.set_property("tracker-width", tracker_width)
            if key == "tracker-height":
                tracker_height = config.getint("tracker", key)
                self.tracker.set_property("tracker-height", tracker_height)
            if key == "gpu-id":
                tracker_gpu_id = config.getint("tracker", key)
                self.tracker.set_property("gpu_id", tracker_gpu_id)
            if key == "ll-lib-file":
                tracker_ll_lib_file = config.get("tracker", key)
                self.tracker.set_property("ll-lib-file", tracker_ll_lib_file)
            if key == "ll-config-file":
                tracker_ll_config_file = config.get("tracker", key)
                self.tracker.set_property("ll-config-file", tracker_ll_config_file)

    def get_source_properties(self) -> tuple:
        """
        Get source property. Like width, height, fps, etc.
        """
        cap = cv2.VideoCapture(self.source_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return width, height, fps

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        frame_number = 0
        # Intiallizing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0,
        }
        num_rects = 0
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            log.info("Unable to get GstBuffer ")
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
            num_rects = frame_meta.num_obj_meta
            l_obj = frame_meta.obj_meta_list
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

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.
            py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(
                frame_number,
                num_rects,
                obj_counter[PGIE_CLASS_ID_VEHICLE],
                obj_counter[PGIE_CLASS_ID_PERSON],
            )

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            log.info(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        # past tracking meta data
        l_user = batch_meta.batch_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if (
                user_meta
                and user_meta.base_meta.meta_type
                == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META
            ):
                try:
                    # Note that user_meta.user_meta_data needs a cast to pyds.NvDsPastFrameObjBatch
                    # The casting is done by pyds.NvDsPastFrameObjBatch.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone
                    pPastFrameObjBatch = pyds.NvDsPastFrameObjBatch.cast(
                        user_meta.user_meta_data
                    )
                except StopIteration:
                    break
                # for trackobj in pyds.NvDsPastFrameObjBatch.list(pPastFrameObjBatch):
                #     print("streamId=", trackobj.streamID)
                #     print("surfaceStreamID=", trackobj.surfaceStreamID)
                #     for pastframeobj in pyds.NvDsPastFrameObjStream.list(trackobj):
                #         print("numobj=", pastframeobj.numObj)
                #         print("uniqueId=", pastframeobj.uniqueId)
                #         print("classId=", pastframeobj.classId)
                #         print("objLabel=", pastframeobj.objLabel)
                #         for objlist in pyds.NvDsPastFrameObjList.list(pastframeobj):
                #             print("frameNum:", objlist.frameNum)
                #             print("tBbox.left:", objlist.tBbox.left)
                #             print("tBbox.width:", objlist.tBbox.width)
                #             print("tBbox.top:", objlist.tBbox.top)
                #             print("tBbox.right:", objlist.tBbox.height)
                #             print("confidence:", objlist.confidence)
                #             print("age:", objlist.age)
            try:
                l_user = l_user.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Deepstream app 1")
    parser.add_argument(
        "--source-file",
        type=str,
        default="/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264",
        help="Path to the source file",
    )
    parser.add_argument(
        "--pgie-config-path",
        type=str,
        default="configs/pgies/app2_pgie_config.txt",
        help="Path to the pgie config file",
    )
    parser.add_argument(
        "--sgie1-config-path",
        type=str,
        default="configs/sgies/app2_sgie1_config.txt",
        help="Path to the sgie1 config file",
    )
    parser.add_argument(
        "--sgie2-config-path",
        type=str,
        default="configs/sgies/app2_sgie2_config.txt",
        help="Path to the sgie2 config file",
    )
    parser.add_argument(
        "--sgie3-config-path",
        type=str,
        default="configs/sgies/app2_sgie3_config.txt",
        help="Path to the sgie3 config file",
    )
    parser.add_argument(
        "--tracker-config-path",
        type=str,
        default="configs/trackers/app2_tracker_config.txt",
        help="Path to the tracker config file",
    )
    args = parser.parse_args()

    pipeline = DeepstreamApp2(
        args.source_file,
        args.pgie_config_path,
        args.sgie1_config_path,
        args.sgie2_config_path,
        args.sgie3_config_path,
        args.tracker_config_path,
    )
    pipeline.setup()
    pipeline.run()
