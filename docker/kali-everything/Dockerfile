FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

# Install the full Kali toolset and Python required by terminal_verify.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 \
    && apt-get install -y kali-linux-everything \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["/bin/bash"]
