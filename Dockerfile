FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update
RUN apt-get install -y git build-essential python3-pip cmake libboost-all-dev libconfig++-dev libyaml-cpp-dev

WORKDIR /setup/accelergy
RUN git clone https://github.com/HewlettPackard/cacti.git
RUN git clone https://github.com/Accelergy-Project/accelergy.git
RUN git clone https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git
RUN git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
RUN git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
RUN cd cacti && make
RUN cd accelergy && pip3 install .
RUN cd accelergy-aladdin-plug-in/ && pip3 install .
RUN cd accelergy-cacti-plug-in/ && pip3 install .
RUN cp -r cacti /usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
RUN cd accelergy-table-based-plug-ins/ && pip3 install .

RUN accelergy
RUN accelergyTables

WORKDIR /setup/
COPY . /setup/medea
RUN mkdir medea/build
RUN cd medea/build && cmake .. && make -j4
RUN cp medea/build/medea /usr/local/bin/medea
WORKDIR /app/