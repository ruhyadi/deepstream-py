# usage: docker build -f dockerfile -t ruhyadi/dspy:v0.0.1 .
FROM nvcr.io/nvidia/deepstream:6.3-samples

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV LOGLEVEL="INFO"
ENV GST_DEBUG=2
ENV GST_DEBUG_FILE=/app/output/GST_DEBUG.log

# Install prerequisites
RUN apt-get update 2>/dev/null; exit 0
RUN apt install --reinstall -y \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly libavresample-dev libavresample4 \
    libavutil-dev libavutil56 libavcodec-dev libavcodec58 \
    libavformat-dev libavformat58 libavfilter7 \
    libde265-dev libde265-0 libx264-155 libx265-179 \
    libvpx6 libmpeg2encpp-2.1-0 libmpeg2-4 libmpg123-0
RUN apt-get install -y gstreamer1.0-libav python3-gi python3-dev \
    python3-numpy python3-opencv \
    python3-gst-1.0 python-gi-dev git wget \
    python-dev python3 python3-pip python3.8-dev \
    cmake g++ build-essential libglib2.0-dev libglib2.0-dev-bin \
    libgstreamer1.0-dev libtool m4 autoconf automake \
    libgirepository1.0-dev libcairo2-dev

# Install RTSP server
RUN apt-get install -y --no-install-recommends \
    libgstrtspserver-1.0-0 gstreamer1.0-rtsp \
    libgirepository1.0-dev gobject-introspection \
    gir1.2-gst-rtsp-server-1.0

# Install gstreamer bindings
RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_x86_64.whl \
    && pip install pyds-1.1.8-py3-none-linux_x86_64.whl \
    && rm pyds-1.1.8-py3-none-linux_x86_64.whl

# Install python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Working directory
WORKDIR /app

# Entrypoint
ENTRYPOINT [ "bash" ]