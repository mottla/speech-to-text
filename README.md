# Speech to Text with Whisper v3

## Overview

This project implements a speech-to-text (STT) system using the Whisper v3 large model. The entire process is executed locally, ensuring that all audio data and transcriptions remain private and secure. The system learns to identify speaker voices over time and annotates the transcript accordingly, enhancing the clarity and context of the transcriptions.

Transcription can occur in two ways:
- **Just-in-Time (Streaming)**: Transcription can happen in real-time via a stream.
- **Static**: Transcription can also be performed from local WAV files.

## Features

- **Local Execution**: All processing is done locally to maintain privacy and security.
- **Adaptive Learning**: The system learns to identify speaker voices and annotates the transcript respectively, improving transcription accuracy and context.
- **Model Flexibility**: The Whisper v3 model can be replaced with any other text-to-speech (TTS) model that adheres to the Hugging Face interface.
- **Translation Support**: Whisper v3 supports real-time translation, allowing for multilingual transcription.

## Requirements

- **VRAM**: Approximately 7 GB of VRAM is required for optimal performance.
- **CUDA**: Ensure that CUDA is installed and configured for GPU acceleration.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies. The following libraries are not installed by default and need to be installed via pip:

   ```bash
   pip install torch sounddevice torchaudio pyannote.audio transformers speechbrain numpy
   ```

3. Ensure that you have the necessary hardware and software requirements met, including CUDA and the appropriate drivers for your GPU.


### Note on Pyannote

For the `pyannote.audio` library, you may need to install additional dependencies based on your specific use case. Refer to the [Pyannote documentation](https://pyannote.github.io/) for more details.

## Usage

To run the speech-to-text system, execute the following command:

```bash
python main.py --input <path_to_audio_file>
```

Replace `<path_to_audio_file>` with the path to the audio file you want to transcribe. For streaming transcription, ensure your audio input is set up correctly.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech-to-text model.
- [Hugging Face](https://huggingface.co/) for providing a flexible interface for model integration.
