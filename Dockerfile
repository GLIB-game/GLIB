FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN echo 'deb http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse \
deb http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse \
deb http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse \
deb http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse \
deb http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse \
deb-src http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse \
deb-src http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse \
deb-src http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse \
deb-src http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse \
deb-src http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse' > /etc/apt/sources.list

COPY sources/GLIB /code
RUN apt-get update && \
apt-get install -y wget \
gnupg \
apt-transport-https \
tzdata \
net-tools \
dnsutils \
iproute2 \
gcc \
tmux \
htop \
git \
vim \
sudo \
cmake \
libgl1-mesa-glx \
libglib2.0-0 \
openssh-server && \
mkdir -p /var/run/sshd && \
cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add - && \
echo 'deb https://artifacts.elastic.co/packages/6.x/apt stable main' | tee -a /etc/apt/sources.list.d/elastic-6.x.list && \
apt-get clean && rm -rf /var/lib/apt/lists/*



RUN apt-get install -y python3.5 && \
ln -sf /usr/bin/python3.5 /usr/bin/python && \
ln -sf /usr/bin/python3.5 /usr/bin/python3
# can be ignored as automatic call for ubuntu/debian

RUN wget https://bootstrap.pypa.io/pip/3.5/get-pip.py
RUN python get-pip.py

RUN pip install --upgrade pip
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ imutils==0.5.4 \
matplotlib==3.0.3 \
numpy==1.14.5 \
opencv-contrib-python==4.4.0.42 \
opencv-python==3.4.0.14 \
pandas==0.24.2 \
Pillow==5.4.1 \
scipy==1.2.0 \
torch==0.4.0 \
torchvision==0.2.2.post3 \
sklearn
