FROM nvidia/cuda:10.0-cudnn7-devel-centos7

RUN yum -y --enablerepo=extras install epel-release

RUN yum update -y

RUN yum groupinstall -y "Development Tools"

RUN yum install -y python3 python3-pip python3-devel cmake3

RUN python3 -m pip install -U  pip

RUN python3 -m pip install setuptools wheel numpy scipy

RUN ln -s /usr/bin/cmake3 /usr/bin/cmake

