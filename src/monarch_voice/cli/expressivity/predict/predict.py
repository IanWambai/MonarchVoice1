# Copyright (c) Monarch Voice-1 (MV1) Project
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import torch
import torchaudio
from pathlib import Path

from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter

from monarch_voice.cli.expressivity.predict.pretssel_generator import (
    PretsselGenerator,
)
from monarch_voice.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)

from monarch_voice.inference import Translator, SequenceGeneratorOptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_expressive_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model_name",
        type=str,
        default="monarch_expressivity",
        choices=["monarch_expressivity"],
        help="Model type",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default="vocoder_model",
        help="Vocoder name",
    )
    parser.add_argument(
        "--duration_factor",
        type=float,
        default=1.0,
        help="Duration factor for prosody",
    )
    return parser


def remove_prosody_tokens_from_text(text: str) -> str:
    # filter out prosody tokens, there is only emphasis '*', and pause '='
    return text.replace("*", "").replace("=", "")


def main() -> None:
    """CLI for expressivity prediction"""
    parser = argparse.ArgumentParser(description="Monarch Voice Expressivity")
    parser = add_inference_arguments(parser)
    parser = add_expressive_arguments(parser)
    args = parser.parse_args()

    if args.task in ["s2st", "t2st"] and args.output_path is None:
        raise ValueError(
            f"Output path must be provided for {args.task} tasks. Please specify --output_path."
        )

    if args.task not in ["asr"] and args.tgt_lang is None:
        raise ValueError(
            f"Target language must be provided for {args.task} tasks. Please specify --tgt_lang."
        )

    if args.task in ["t2tt", "t2st"] and args.src_lang is None:
        raise ValueError(
            f"Source language must be provided for {args.task} tasks. Please specify --src_lang."
        )

    text_gen_opts, unit_gen_opts = set_generation_opts(args)

    translator = Translator(
        model_name=args.model_name,
        vocoder_name=args.vocoder_name,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Currently supporting only s2st expressivity
    if args.task != "s2st":
        raise NotImplementedError(
            f"Task type {args.task} is not supported. "
            "Currently only s2st is supported."
        )

    logger.info(f"Translating speech in {args.input!r} to {args.tgt_lang} speech...")
    audio_input = torchaudio.load(args.input)[0]
    result = translator.predict(
        input=audio_input,
        task_str=args.task,
        tgt_lang=args.tgt_lang,
        text_gen_opts=text_gen_opts,
        unit_gen_opts=unit_gen_opts,
    )

    pretssel_generator = PretsselGenerator(
        device=args.device, 
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        vocoder_name=args.vocoder_name,
    )

    waveform_to_fbank_converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
        device=args.device,
        dtype=torch.float32,
    )

    src_audio_fbank = waveform_to_fbank_converter(
        SequenceData(audio_input.to(torch.float32))
    )
    
    out_wav = pretssel_generator.generate(
        src_fbank=src_audio_fbank,
        tgt_lang=args.tgt_lang,
        tgt_unit=result["tgt_unit"],
        tgt_text=result["tgt_text"],
        duration_factor=args.duration_factor,
    )

    tgt_text_without_prosody = remove_prosody_tokens_from_text(result["tgt_text"])
    logger.info(f"Speech translation: {tgt_text_without_prosody}")
    logger.info(f"With prosody tokens: {result['tgt_text']}")

    outpath = Path(args.output_path)
    torchaudio.save(outpath, out_wav.cpu(), 16000)
    logger.info(f"Output audio saved to {outpath!r}.")


if __name__ == "__main__":
    main() 