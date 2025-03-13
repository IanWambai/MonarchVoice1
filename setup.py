# Copyright (c) Monarch Voice-1 (MV1) Project
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="monarch_voice",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["py.typed", "cards/*.yaml"]},
    description="Monarch Voice-1 (MV1) -- Speech-to-Speech AI Translation for African Languages",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Monarch Voice Project",
    url="https://github.com/monarchvoice/monarch_voice",
    license="Creative Commons",
    install_requires=[
        "datasets==2.18.0",
        "fairseq2==0.2.*",
        "fire",
        "librosa",
        "openai-whisper",
        "simuleval~=1.1.3",
        "sonar-space==0.2.*",
        "soundfile",
        "scipy",
        "torchaudio",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "mv1_evaluate=monarch_voice.cli.m4t.evaluate.evaluate:main",
            "mv1_predict=monarch_voice.cli.m4t.predict.predict:main",
            "mv1_finetune=monarch_voice.cli.m4t.finetune.finetune:main",
            "mv1_prepare_dataset=monarch_voice.cli.m4t.finetune.dataset:main",
            "mv1_audio_to_units=monarch_voice.cli.m4t.audio_to_units.audio_to_units:main",
            "mv1_expressivity_evaluate=monarch_voice.cli.expressivity.evaluate.evaluate:main",
            "mv1_expressivity_predict=monarch_voice.cli.expressivity.predict.predict:main",
            "mv1_streaming_evaluate=monarch_voice.cli.streaming.evaluate:main",
        ],
    },
    include_package_data=True,
)
