# CPU bring-up targets: replay, preview, causal MediaPipe live and its tests.
FROM python:3.10-slim-bookworm AS replay

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COPY pyproject.toml .
# The common CLI loads the perception profile registry through its public API;
# cv2 is needed for those imports, but no camera SDK or model weights are used.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade "pip>=25" hatchling editables && \
    pip install "numpy>=1.26" "scipy>=1.11" "h5py>=3.10" \
        "ur-rtde>=1.6.3,<2" "typing_extensions>=4.12" "opencv-python-headless>=4.9"
RUN mkdir -p viki && touch viki/__init__.py && pip install --no-build-isolation --no-deps -e .
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh", "viki"]
CMD ["--help"]

FROM replay AS preview
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libegl-mesa0 libglx-mesa0 libglfw3 libgl1-mesa-dri libosmesa6 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "mujoco>=3.3,<4"

# No CUDA/Torch dependency for the first human-following mode. Recorded Kinect
# colour geometry still needs libk4a to decode the saved factory calibration.
FROM preview AS live
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git libudev1 libx11-6 libusb-1.0-0 libportaudio2 libglib2.0-0 libgomp1 libgles2 && \
    printf '%s\n' \
      'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' \
      'libk4a1.4 libk4a1.4/accept-eula boolean true' | debconf-set-selections && \
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 180 \
      -o /tmp/libk4a.deb \
      https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends /tmp/libk4a.deb && \
    rm /tmp/libk4a.deb && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy>=2,<2.3" "pin>=4.1,<4.2" "pin-pink>=4,<5" \
      "qpsolvers>=4.4" quadprog osqp robot_descriptions xacrodoc \
      "pyrealsense2>=2.58" "requests>=2.31" "threadpoolctl>=3" \
      "fastapi>=0.111" "uvicorn[standard]>=0.29" \
      "onnxruntime>=1.22" "tqdm>=4.66" \
      "matplotlib>=3.8" "absl-py~=2.3" certifi "sounddevice~=0.5" "flatbuffers~=25.9" && \
    pip install --no-deps "mediapipe==0.10.35" rtmlib

FROM live AS live-test
RUN --mount=type=cache,target=/root/.cache/pip pip install "pytest>=8" httpx
ENTRYPOINT ["/app/docker/entrypoint.sh", "pytest"]
CMD ["tests/", "-q"]
