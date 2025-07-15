FROM manimcommunity/manim:stable

# Install additional LaTeX packages needed for mathematical notation
USER root

RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-latex-recommended \
    texlive-science \
    texlive-xetex \
    tipa \
    dvipng \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory with proper permissions
RUN mkdir -p /manim && chown manimuser:manimuser /manim
WORKDIR /manim

# Switch back to manimuser
USER manimuser

# Set environment for better LaTeX rendering
ENV TEXMFHOME=/manim/.texmf

# Default command
CMD ["/bin/bash"]