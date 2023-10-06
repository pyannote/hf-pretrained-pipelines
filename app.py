# MIT License
#
# Copyright (c) 2022- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import io
import base64
import torch
import numpy as np
import scipy.io.wavfile
from typing import Text
import streamlit as st
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote.core import Segment

import streamlit.components.v1 as components


def to_base64(waveform: np.ndarray, sample_rate: int = 16000) -> Text:
    """Convert waveform to base64 data"""
    waveform /= np.max(np.abs(waveform)) + 1e-8
    with io.BytesIO() as content:
        scipy.io.wavfile.write(content, sample_rate, waveform)
        content.seek(0)
        b64 = base64.b64encode(content.read()).decode()
        b64 = f"data:audio/x-wav;base64,{b64}"
    return b64


PYANNOTE_LOGO = "https://avatars.githubusercontent.com/u/7559051?s=400&v=4"
EXCERPT = 30.0

st.set_page_config(page_title="pyannote pretrained pipelines", page_icon=PYANNOTE_LOGO)

col1, col2 = st.columns([0.2, 0.8], gap="small")

with col1:
    st.image(PYANNOTE_LOGO)

with col2:
    st.markdown(
        """
# pretrained pipelines
Make the most of [pyannote](https://github.com/pyannote) thanks to our [consulting services](https://herve.niderb.fr/consulting.html)
"""
    )


PIPELINES = [
    "pyannote/speaker-diarization-3.0",
]

audio = Audio(sample_rate=16000, mono=True)

selected_pipeline = st.selectbox("Select a pretrained pipeline", PIPELINES, index=0)


with st.spinner("Loading pipeline..."):
    try:
        use_auth_token = st.secrets["PYANNOTE_TOKEN"]
    except FileNotFoundError:
        use_auth_token = None
    except KeyError:
        use_auth_token = None

    pipeline = Pipeline.from_pretrained(
        selected_pipeline, use_auth_token=use_auth_token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

uploaded_file = st.file_uploader("Upload an audio file")
if uploaded_file is not None:
    try:
        duration = audio.get_duration(uploaded_file)
    except RuntimeError as e:
        st.error(e)
        st.stop()
    waveform, sample_rate = audio.crop(
        uploaded_file, Segment(0, min(duration, EXCERPT))
    )
    uri = "".join(uploaded_file.name.split())
    file = {"waveform": waveform, "sample_rate": sample_rate, "uri": uri}

    with st.spinner(f"Processing {EXCERPT:g} seconds..."):
        output = pipeline(file)

    with open("assets/template.html") as html, open("assets/style.css") as css:
        html_template = html.read()
        st.markdown("<style>{}</style>".format(css.read()), unsafe_allow_html=True)

    colors = [
        "#ffd70033",
        "#00ffff33",
        "#ff00ff33",
        "#00ff0033",
        "#9932cc33",
        "#00bfff33",
        "#ff7f5033",
        "#66cdaa33",
    ]
    num_colors = len(colors)

    label2color = {
        label: colors[k % num_colors] for k, label in enumerate(sorted(output.labels()))
    }

    BASE64 = to_base64(waveform.numpy().T)

    REGIONS = ""
    for segment, _, label in output.itertracks(yield_label=True):
        REGIONS += f"regions.addRegion({{start: {segment.start:g}, end: {segment.end:g}, color: '{label2color[label]}', resize : false, drag : false}});"

    html = html_template.replace("BASE64", BASE64).replace("REGIONS", REGIONS)
    components.html(html, height=250, scrolling=True)

    with io.StringIO() as fp:
        output.write_rttm(fp)
        content = fp.getvalue()
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a download="{output.uri}.rttm" href="data:file/text;base64,{b64}">Download</a> result in RTTM file format or run it locally:'
        st.markdown(href, unsafe_allow_html=True)

    code = f"""
# load pretrained pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("{selected_pipeline}", 
                                    use_auth_token=HUGGINGFACE_TOKEN)

# (optional) send pipeline to GPU
import torch
pipeline.to(torch.device("cuda"))

# process audio file
output = pipeline("audio.wav")"""
    st.code(code, language="python")
