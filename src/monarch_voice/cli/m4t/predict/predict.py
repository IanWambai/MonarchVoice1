# Copyright (c) Monarch Voice-1 (MV1) Project
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
from fairseq2.generation import NGramRepeatBlockProcessor

from monarch_voice.inference import SequenceGeneratorOptions, Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "input",
        type=str,
        help="Input text for t2tt/t2st tasks or path to input audio file for s2st/s2tt/asr tasks.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["s2st", "s2tt", "t2st", "t2tt", "asr"],
        help="Task type",
    )
    parser.add_argument("--tgt_lang", type=str, help="Target language.")
    parser.add_argument("--src_lang", type=str, help="Source language.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mv1_large",
        choices=["mv1_large", "mv1_medium"],
        help="Model type",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default="vocoder_v2",
        help="Vocoder type.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output file path, required for s2st and t2st tasks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference.",
    )
    # UnitY2 args
    parser.add_argument(
        "--unity2_model_name",
        type=str,
        default="mv1_unity2",
        help="Model name of UnitY2.",
    )
    # generation args
    parser.add_argument(
        "--beam",
        type=int,
        default=5,
        help="Beam size. Ignored for text to units generation.",
    )
    parser.add_argument(
        "--max_len_a",
        type=float,
        default=1.0,
        help="Max length (per input token). Ignored for text to units generation.",
    )
    parser.add_argument(
        "--max_len_b",
        type=int,
        default=200,
        help="Max length (static value). Ignored for text to units generation.",
    )
    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="Use unnormalized text. Ignored for speech input tasks.",
    )
    # text / unit generation specific args
    parser.add_argument(
        "--len_penalty",
        type=float,
        default=1.0,
        help="Length penalty. Ignored for text to units generation.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Size of no repeat ngram. Ignored for text to units generation.",
    )
    # text generation specific args
    parser.add_argument(
        "--t_text_generation_temperature",
        type=float,
        default=1.0,
        help="Temperature for text generation.",
    )
    parser.add_argument(
        "--t_text_generation_topp",
        type=float,
        default=1.0,
        help="Nucleus sampling probability for text generation.",
    )
    # unit generation specific args
    parser.add_argument(
        "--t_unit_generation_temperature",
        type=float,
        default=1.0,
        help="Temperature for unit generation.",
    )
    parser.add_argument(
        "--t_unit_generation_topp",
        type=float,
        default=0.8,
        help="Nucleus sampling probability for unit generation.",
    )
    parser.add_argument(
        "--unit_generation_ngram_blocking",
        type=int,
        default=8,
        help="NGram blocking size for unit generation.",
    )
    # Vocoder temp control
    parser.add_argument(
        "--vocoder_temp",
        type=float,
        default=0.8,
        help="Vocoder sampling temperature.",
    )
    return parser


def set_generation_opts(
    args: Namespace,
) -> Tuple[SequenceGeneratorOptions, SequenceGeneratorOptions]:
    # Set text, unit generation opts.
    text_gen_opts = SequenceGeneratorOptions(
        beam_size=args.beam,
        soft_max_seq_len=(args.max_len_a, args.max_len_b),
        len_penalty=args.len_penalty,
    )
    if args.no_repeat_ngram_size > 0:
        text_gen_opts.logits_processor = NGramRepeatBlockProcessor(
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )
    text_gen_opts.temperature = args.t_text_generation_temperature
    text_gen_opts.top_p = args.t_text_generation_topp

    unit_gen_opts = SequenceGeneratorOptions()
    if args.task in ["s2st", "t2st"]:
        # We always use sampling for UnitY as it improves naturalness.
        unit_gen_opts.temperature = args.t_unit_generation_temperature
        unit_gen_opts.top_p = args.t_unit_generation_topp
        if args.unit_generation_ngram_blocking > 0:
            unit_gen_opts.logits_processor = NGramRepeatBlockProcessor(
                no_repeat_ngram_size=args.unit_generation_ngram_blocking
            )
    return text_gen_opts, unit_gen_opts


def main() -> None:
    """CLI for MV1 inference"""
    parser = argparse.ArgumentParser(description="MV1 inference")
    parser = add_inference_arguments(parser)
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

    if args.task in ["s2st", "s2tt", "asr"]:
        # Handle audio file as input for s2st, s2tt, asr tasks.
        audio_input = torchaudio.load(args.input)[0]
        if args.task == "s2st":
            logger.info(
                f"Translating speech in {args.input!r} to {args.tgt_lang} speech..."
            )
            result = translator.predict(
                input=audio_input,
                task_str=args.task,
                tgt_lang=args.tgt_lang,
                text_gen_opts=text_gen_opts,
                unit_gen_opts=unit_gen_opts,
                vocoder_temp=args.vocoder_temp,
            )
            logger.info(f"Speech translation: {result['tgt_text']}")
            outpath = Path(args.output_path)
            torchaudio.save(outpath, result["tgt_waveform"].cpu(), 16000)
            logger.info(f"Output audio saved to {outpath!r}.")
        elif args.task == "s2tt":
            logger.info(f"Translating speech in {args.input!r} to {args.tgt_lang} text...")
            result = translator.predict(
                input=audio_input,
                task_str=args.task,
                tgt_lang=args.tgt_lang,
                text_gen_opts=text_gen_opts,
                unit_gen_opts=unit_gen_opts,
            )
            logger.info(f"Speech translation: {result['tgt_text']}")
        elif args.task == "asr":
            if args.tgt_lang is None:
                logger.warning("Target language not specified, using english for ASR.")
                args.tgt_lang = "eng"
            logger.info(f"Transcribing speech in {args.input!r} to {args.tgt_lang} text...")
            result = translator.predict(
                input=audio_input,
                task_str=args.task,
                tgt_lang=args.tgt_lang,
                text_gen_opts=text_gen_opts,
                unit_gen_opts=unit_gen_opts,
            )
            logger.info(f"Transcription: {result['tgt_text']}")
    else:
        # Handle text as input
        normalize = not args.unnormalized
        if args.task == "t2tt":
            logger.info(
                f"Translating {args.src_lang} text {args.input!r} to {args.tgt_lang} text..."
            )
            result = translator.predict(
                input=args.input,
                task_str=args.task,
                tgt_lang=args.tgt_lang,
                src_lang=args.src_lang,
                text_gen_opts=text_gen_opts,
                unit_gen_opts=unit_gen_opts,
                normalize_text=normalize,
            )
            logger.info(f"Text translation: {result['tgt_text']}")
        elif args.task == "t2st":
            logger.info(
                f"Translating {args.src_lang} text {args.input!r} to {args.tgt_lang} speech..."
            )
            result = translator.predict(
                input=args.input,
                task_str=args.task,
                tgt_lang=args.tgt_lang,
                src_lang=args.src_lang,
                text_gen_opts=text_gen_opts,
                unit_gen_opts=unit_gen_opts,
                vocoder_temp=args.vocoder_temp,
                normalize_text=normalize,
            )
            logger.info(f"Text to speech translation: {result['tgt_text']}")
            outpath = Path(args.output_path)
            torchaudio.save(outpath, result["tgt_waveform"].cpu(), 16000)
            logger.info(f"Output audio saved to {outpath!r}.")


if __name__ == "__main__":
    main() 