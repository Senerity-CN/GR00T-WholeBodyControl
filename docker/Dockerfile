# SONIC Finetune 环境镜像
# 纯环境，不含训练数据和代码（运行时通过挂载提供）
#
# 基于 ubuntu:22.04 + Python 3.11，pip install 所有依赖
# CUDA 运行时由 PyTorch pip 包自带，GPU 驱动由宿主机通过 nvidia-container-toolkit 映射
# 镜像内包含：系统库 + Python + PyTorch + IsaacSim + IsaacLab + gear_sonic 依赖

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 使用中山大学镜像源
RUN echo "\
deb http://mirror.sysu.edu.cn/ubuntu/ jammy main restricted universe multiverse\n\
deb http://mirror.sysu.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse\n\
deb http://mirror.sysu.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse\n\
deb http://mirror.sysu.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" \
    > /etc/apt/sources.list

# 系统依赖
# - Vulkan: libvulkan1 (loader), mesa-vulkan-drivers (fallback ICD)
# - EGL/GL: libegl1, libgles2, libgl1-mesa-glx, libglu1-mesa, libglvnd0
# - X11: libxt6, libxrandr2, libxi6, libxinerama1, libxcursor1
# - IsaacSim iray: libglu1-mesa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    curl \
    wget \
    cmake \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    libegl1 \
    libgles2 \
    libgl1-mesa-glx \
    libglvnd0 \
    libglu1-mesa \
    libxt6 \
    libxrandr2 \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    libxkbcommon0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline-dev \
    zlib1g-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# 接受 Conda 默认频道的 ToS（新版 conda 非交互模式必需）
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建 Python 3.11 环境
RUN conda create -n isaaclab python=3.11 -y && \
    conda clean -afy

# 后续所有操作在 conda env 内
SHELL ["conda", "run", "-n", "isaaclab", "/bin/bash", "-c"]

# PyTorch 2.7 + CUDA 12.8
COPY docker/wheels/ /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*

# IsaacSim 5.1.0（含所有扩展和 extscache，否则 SimulationApp 为 None）
RUN pip install --no-cache-dir \
    "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

# === 修复 flatdict 编译报错（新增步骤） ===
# 提前安装兼容的 setuptools、wheel，并以"无构建隔离"方式安装 flatdict
RUN pip install --no-cache-dir setuptools==69.5.1 wheel && \
    pip install --no-cache-dir --no-build-isolation flatdict==4.0.1 && \
    pip install --no-cache-dir egl_probe==1.0.2

# IsaacLab（从源码 editable install）
COPY IsaacLab/ /opt/IsaacLab/
RUN pip install --no-cache-dir -e /opt/IsaacLab/source/isaaclab && \
    pip install --no-cache-dir -e /opt/IsaacLab/source/isaaclab_assets && \
    pip install --no-cache-dir -e /opt/IsaacLab/source/isaaclab_mimic && \
    pip install --no-cache-dir -e /opt/IsaacLab/source/isaaclab_rl && \
    pip install --no-cache-dir -e /opt/IsaacLab/source/isaaclab_tasks

# 项目核心依赖（已去除 ROS 相关包，仅保留训练所需）
COPY docker/requirements_train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt && \
    rm /tmp/requirements_train.txt

# rl_games 声明不支持 Python 3.11 但实际兼容，强制安装
# 注意：rl_games 依赖 gym[classic-control] → pygame==2.1.0，
# 但 pygame 2.1.0 没有 Python 3.11 预编译 wheel，源码编译需要 libsdl2-dev。
# 解决方案：先装有预编译 wheel 的新版 pygame，再用 --no-deps 跳过依赖解析
# （rl_games 的其他依赖 PyYAML/gym/opencv/tensorboard 等已在 requirements_train.txt 中安装）
RUN pip install --no-cache-dir pygame && \
    pip install --no-cache-dir --ignore-requires-python --no-deps rl_games==1.6.1


# gear_sonic 代码（环境的一部分，editable install）
# 将 gear_sonic/pyproject.toml 复制为 /opt/pyproject.toml，
# 然后通过 sed 将 where=[".."] 改为 where=["."]，
# 使得 setuptools 在 /opt/ 目录下查找 gear_sonic 包
COPY gear_sonic/pyproject.toml /opt/pyproject.toml
COPY gear_sonic/ /opt/gear_sonic/
# gear_sonic/pyproject.toml 中 where=[".."] 是相对于原始项目结构的
# 在 Docker 中 pyproject.toml 位于 /opt/，".." 会解析到 / （根文件系统）
# 导致 setuptools 扫描整个文件系统，必须改为 ["."] 指向 /opt/ 自身
RUN sed -i 's|where = \["\.\."\]|where = ["."]|g' /opt/pyproject.toml && \
    cd /opt && pip install --no-cache-dir -e . --no-deps --no-build-isolation

WORKDIR /workspace

# 激活 conda env
RUN echo "conda activate isaaclab" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=isaaclab
ENV PATH="/opt/conda/envs/isaaclab/bin:$PATH"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "isaaclab"]
CMD ["bash"]
