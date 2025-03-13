![](monarch_voice_demo.wav)
# Monarch Voice-1 (MV1)

Monarch Voice-1 (MV1) is an initiative in artificial intelligence designed to explore new ways of overcoming linguistic barriers and improving digital accessibility for communities across Africa.

While still early in development, this project aims to provide insights into how voice-based technologies can more effectively serve diverse linguistic communities.

## What is MV1?

Monarch Voice is a speech-to-speech artificial intelligence system created to enable direct, real-time conversations across multiple languages without relying on intermediate text-based translations.

Built on advanced neural network architectures, it processes spoken language inputs by capturing the nuances of context, intonation, and emotional expression, and then quickly generates coherent and contextually relevant spoken outputs.

Unlike traditional translation systems that primarily focus on textual accuracy, Monarch Voice emphasizes maintaining the unique vocal characteristics, emotional nuances, and conversational tone of each speaker, making interactions feel significantly more natural and authentic.

## How It Works

Differing from conventional speech-to-text translation methods, Monarch Voice directly converts spoken inputs into spoken outputs in selected target languages.

It specifically prioritizes preserving the expressive and emotional qualities inherent in the original voice, thus fostering a more engaging and human-like communication experience.

The underlying technology combines advanced speech recognition models, real-time audio processing capabilities, and expressive voice synthesis algorithms. This integrated approach is designed to support various practical use cases, such as:

- **Software development**, enabling clearer communication between international developer teams.
- **Remote education**, facilitating accessible learning opportunities for students regardless of language proficiency.
- **Entrepreneurship**, empowering individuals to engage in global business activities seamlessly.
- **International collaboration**, removing language as a barrier to innovation and cooperation.

This technology seeks to empower users, particularly those without strong English language skills, to actively participate and thrive in various digital domains.

## Languages

MV1 currently supports speech-to-speech translation in a focused subset of key global and regional languages, including:
- English
- Kiswahili
- Mandarin Chinese
- Spanish
- French
- Hindi
- Arabic
- German
- Japanese
- Korean
- Portuguese
- Russian
- Indonesian
- Turkish
- And several others

We plan to steadily expand our coverage in future updates to include additional African languages such as Yoruba, Hausa, Igbo, and more. We remain committed to broadening accessibility and inclusivity by continuously growing the linguistic capabilities of MV1.

This broad linguistic coverage positions MV1 as a practical, globally scalable communication tool, explicitly designed to transcend traditional language barriers and empower communities worldwide with frictionless, expressive, and culturally resonant conversations.

## Voice Samples

MV1 has been developed to demonstrate basic expressive voice interaction capabilities, specifically focusing on translating spoken English into languages such as Swahili.

Through this initial release, we invite community members, potential end-users, and developers to engage with the model, test its functionality, and provide valuable feedback that will guide future refinements and improvements.

## Quick Start
### Installation

```
pip install .
```

> [!NOTE]
> One of the prerequisites is [fairseq2](https://github.com/facebookresearch/fairseq2) which has pre-built packages available only
> for Linux x86-64 and Apple-silicon Mac computers. In addition it has a dependency on [libsndfile](https://github.com/libsndfile/libsndfile) which
> might not be installed on your machine. If you experience any installation issues, please refer to its
> [README](https://github.com/facebookresearch/fairseq2) for further instructions.

### Running inference

#### Speech-to-Speech Translation
```bash
mv1_predict <path_to_input_audio> --task s2st --tgt_lang <tgt_lang> --output_path <path_to_save_audio>
```

#### Text-to-Text Translation
```bash
mv1_predict <input_text> --task t2tt --tgt_lang <tgt_lang> --src_lang <src_lang>
```

#### Expressive Speech Translation
```bash
mv1_expressivity_predict <path_to_input_audio> --tgt_lang <tgt_lang> --model_name monarch_expressivity --vocoder_name vocoder_model --output_path <path_to_save_audio>
```

#### Streaming Translation
Please check the [streaming documentation](src/monarch_voice/cli/streaming) for detailed instructions on running streaming translation.

### Running Demo Locally

To launch a local demo:

```bash
cd demo
pip install -r requirements.txt
python app.py
```

## Model Details

Monarch Voice is built on advanced neural network architectures that combine:

1. Speech recognition components to capture source language inputs
2. Translation components to render the meaning in the target language
3. Expressive voice synthesis to maintain vocal characteristics
4. Streaming capabilities for real-time translation

## Contributing

We welcome contributions from the community. If you're interested in improving Monarch Voice, please check out our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Virtual Environment Setup

The `seamless-env` directory has been excluded from this repository due to file size limitations. To set up the environment:

1. Create a new virtual environment and install dependencies:
   ```bash
   python -m venv monarch-env
   source monarch-env/bin/activate  # On Windows, use: monarch-env\Scripts\activate
   pip install -e .
   ```

This will install all the necessary dependencies including fairseq2, torch, and other libraries required to run the models.
