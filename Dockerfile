FROM pytorch/pytorch

# download necessary files
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    libsqlite3-0 \
    libsqlite3-dev \
    sqlite3 \
    tar \
    vim \
    pkg-config \
    build-essential


# install pip and REL
RUN conda install -y pip
RUN pip install --upgrade pip

COPY . REL
RUN cd REL && \
    pip install -e .


# ????
RUN chmod -R 777 /workspace && chown -R root:root /workspace

# Download data
RUN /workspace/REL/scripts/download_data.sh /workspace/data generic wiki_2019 ed-wiki-2019

# expose the API port
EXPOSE 5555

CMD ["bash", "-c"]
