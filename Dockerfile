# use debian as base image
FROM debian:latest

# get list of installable packets and install wget
RUN apt-get upgrade
RUN apt-get update && \
    apt-get -y install \
        'wget'

# download snap installer version 8.0
RUN wget http://step.esa.int/downloads/8.0/installers/esa-snap_sentinel_unix_8_0.sh

#change file execution rights for snap installer
RUN chmod +x esa-snap_sentinel_unix_8_0.sh

# install snap with gpt
RUN ./esa-snap_sentinel_unix_8_0.sh -q

# link gpt so it can be used systemwide
RUN ln -s /usr/local/snap/bin/gpt /usr/bin/gpt

# set gpt max memory to 128GB
#RUN sed -i -e 's/-Xmx1G/-Xmx128G/g' /usr/local/snap/bin/gpt.vmoptions
RUN sed -i 's/-Xmx.*/-Xmx20G/' /usr/local/snap/bin/gpt.vmoptions

# install jdk and python with required modules
RUN apt-get -y install default-jdk python python-pip git maven python-jpy
RUN python -m pip install --user --upgrade setuptools wheel

# set JDK_HOME env
ENV JDK_HOME="/usr/lib/jvm/default-java"
ENV JAVA_HOME=$JDK_HOME
ENV PATH=$PATH:/root/.local/bin

# install snappy the SNAP python module
RUN /usr/local/snap/bin/snappy-conf /usr/bin/python
RUN cd /root/.snap/snap-python/snappy/ && \
    python setup.py install
RUN ln -s /root/.snap/snap-python/snappy /usr/lib/python2.7/dist-packages/snappy

# change max memory and max cache etc.
RUN sed -i 's/java_max_mem.*/java_max_mem: 20G/' /root/.snap/snap-python/snappy/snappy.ini
RUN sed -i 's/snap.jai.tileCacheSize.*/snap.jai.tileCacheSize=10000/' /usr/local/snap/etc/snap.properties
RUN sed -i 's/snap.jai.defaultTileSize.*/snap.jai.defaultTileSize=512/' /usr/local/snap/etc/snap.properties

# copy python files and install dependencies
COPY ./api api/
WORKDIR /api
RUN python -m pip install --user -r requirements.txt

RUN apt-get -y install nano
RUN apt-get -y install tmux

