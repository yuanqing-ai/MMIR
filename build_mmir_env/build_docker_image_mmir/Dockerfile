FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Use bash instead of dash
# Set CST timezone
# Disable core dump
RUN ln -sf /bin/bash /bin/sh && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo 'Asia/Shanghai' >/etc/timezone && \
    ulimit -c 0

# 更换apt-get源，安装vim
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# Install packages and clean cached
# RUN gpg --keyserver keyserver.ubuntu.com --recv A4B469963BF863CC
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv A4B469963BF863CC
RUN rm /etc/apt/sources.list.d/cuda.list

#重新安装pyav，以及代码所需的python环境
RUN pip install pip -U && \
    pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple && \
    pip config set install.trusted-host pypi.mirrors.ustc.edu.cn && \
    pip install pandas && \
    pip install torchinfo && \
    pip install tensorboardX
