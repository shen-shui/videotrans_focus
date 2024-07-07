# # 使用官方 Python 运行时作为父镜像
# FROM python:3.11-slim AS base

# # 设置工作目录为 /app
# WORKDIR /app

# # 将当前目录内容复制到容器的 /app 中
# COPY . /app

# # 安装所需包
# RUN pip install --no-cache-dir -r requirements.txt

# # 安装 ffmpeg
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# # 使端口 5000 可供此容器外的环境使用
# EXPOSE 5000

# # 定义环境变量
# ENV TEMP=/tmp
# ENV deeplx_address=https://service-f5qam0f8-1253433727.gz.tencentapigw.com.cn/translate

# # 在容器启动时运行
# CMD ["python", "app.py"]

## 第一阶段 ##
# FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 AS base

# ENV DEBIAN_FRONTEND=noninteractive
# ENV NVIDIA_INSTALLER_NO_INTERACTIVE=1

# 安装依赖
# RUN apt-get update && apt-get install -y --no-install-recommends\
#     ffmpeg \
#     python3.10 \
#     python3-pip \
#     nvidia-driver-535 \
#     libnvidia-compute-535 \
#     libnvidia-encode-535 \
#     nvidia-utils-535 \
#     && rm -rf /var/lib/apt/lists/*

# # 运行时需要NVIDIA Docker runtime
# LABEL com.nvidia.volumes.needed="nvidia_driver"

# ## 第二阶段 ##
# FROM base 
# # 设置工作目录为 /app
# WORKDIR /app

# # 将当前目录内容复制到容器的 /app 中
# COPY . /app

# RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# # 安装所需包
# RUN pip install -r requirements.txt --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

# # 使端口 5000 可供此容器外的环境使用
# EXPOSE 5000

# # 定义环境变量
# ENV TEMP=/tmp
# ENV deeplx_address=https://service-f5qam0f8-1253433727.gz.tencentapigw.com.cn/translate

# # 在容器启动时运行
# CMD ["python", "app.py"]

# Ubuntu:22.04
# # 第一阶段：从本地系统复制ffmpeg
# FROM python:3.11-slim AS base
# WORKDIR /app

# # 复制本地的ffmpeg和ffprobe到镜像中
# COPY /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
# COPY /usr/local/bin/ffprobe /usr/local/bin/ffprobe

# # 第二阶段：复制当前目录到Docker中
# FROM base AS copy
# COPY . .

# # 第三阶段：安装requirements
# FROM base AS final
# WORKDIR /app

# # 从第二阶段复制应用代码
# COPY --from=copy /app /app

# # 安装requirements
# RUN pip install --no-cache-dir -r requirements.txt

# # 设置默认命令
# CMD ["python", "app.py"]

# RUN ffmpeg -encoders

# RUN ffmpeg -vsync 0 -i src.mp4 -c:v h264_nvenc /tmp/output.mp4

# CMD ["ffmpeg","-vsync", "0", "-i", "src.mp4", "-c:v", "h264_nvenc", "/tmp/output.mp4"]


FROM docker.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_INSTALLER_NO_INTERACTIVE=1

RUN apt-get update && apt-get install --no-install-recommends -y \
  python3.10 \
  python3-pip \
  ffmpeg \
  pkg-config \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
    
FROM base AS insreq

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR  $HOME/app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

FROM insreq AS final

RUN useradd -m -u 1000 user

RUN chown -R user:user /home/user

USER user

COPY --chown=user . $HOME/app

CMD ["python3", "app.py"]