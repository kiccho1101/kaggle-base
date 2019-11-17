FROM gcr.io/kaggle-images/python:latest

WORKDIR /app

COPY ./requirements.txt /app

# Install Mecab
RUN apt-get update \
    && apt-get install -y mecab \
    && apt-get install -y libmecab-dev \
    && apt-get install -y mecab-ipadic-utf8 \
    && apt-get install -y git \
    && apt-get install -y make \
    && apt-get install -y curl \
    && apt-get install -y xz-utils \
    && apt-get install -y file \
    && apt-get install -y sudo \
    && apt-get install -y wget \
    && apt-get install -y software-properties-common vim
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && bin/install-mecab-ipadic-neologd -n -y

# Upgraade pip
RUN pip install -U pip
RUN pip install kaggle --upgrade

# Install libraries
RUN pip install -r /app/requirements.txt 

# Install jupyterlab
RUN pip install jupyterlab \
    && jupyter serverextension enable --py jupyterlab

# Install Jupyter Notebook Extensions
RUN jupyter contrib nbextension install --user \
    && jupyter nbextensions_configurator enable --user

# Install Jupyter Black
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user \
    && jupyter nbextension enable jupyter-black-master/jupyter-black

# Install Jupyter Isort
RUN jupyter nbextension install https://github.com/benjaminabel/jupyter-isort/archive/master.zip --user \
    && jupyter nbextension enable jupyter-isort-master/jupyter-isort

# Enable Nbextensions (Reference URL: https://qiita.com/simonritchie/items/88161c806197a0b84174)
RUN jupyter nbextension enable table_beautifier/main \
    && jupyter nbextension enable toc2/main \
    && jupyter nbextension enable toggle_all_line_numbers/main \
    && jupyter nbextension enable autosavetime/main \
    && jupyter nbextension enable collapsible_headings/main \
    && jupyter nbextension enable execute_time/ExecuteTime \
    && jupyter nbextension enable codefolding/main \
    && jupyter nbextension enable varInspector/main \
    && jupyter nbextension enable notify/notify 

# Setup jupyter-vim
RUN mkdir -p $(jupyter --data-dir)/nbextensions \
    && cd $(jupyter --data-dir)/nbextensions \
    && git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding \
    && jupyter nbextension enable vim_binding/vim_binding

# # Change Theme
RUN jt -t chesterish -T -f roboto -fs 9 -tf merriserif -tfs 11 -nf ptsans -nfs 11 -dfs 8 -ofs 8 -cellw 99% \
    && sed -i '1s/^/.edit_mode .cell.selected .CodeMirror-focused:not(.cm-fat-cursor) { background-color: #1a0000 !important; }\n /' /root/.jupyter/custom/custom.css \
    && sed -i '1s/^/.edit_mode .cell.selected .CodeMirror-focused.cm-fat-cursor { background-color: #1a0000 !important; }\n /' /root/.jupyter/custom/custom.css
