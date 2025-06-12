# Speech to Text

## Overview

This project integrates state-of-the-art speech-to-text (STT) models with speaker diarization algorithms to address the common issue of audio segmentation. Traditional transcription methods often divide audio into fixed-size blocks, which can lead to cutting off speakers mid-sentence and introducing errors in the STT output.

To enhance speaker identification, the project also implements speaker embedding extraction techniques, allowing for consistent recognition and differentiation of speakers across multiple sessions and use cases.

All processes are executed locally, ensuring that audio data and transcriptions remain private and secure.

Transcription can occur in two ways:
- **Just-in-Time (Streaming)**: Transcription can happen in real-time. Select one or more devices to transcribe for example an online meeting and your microphone input together. 
- **Static**: Transcription can also be performed from local audio files. (currently supported: WAV,MP3)

## Features

- **Local Execution**: All processing is done locally to maintain privacy and security.
- **Adaptive Learning**: (Optionally) The system learns to identify speaker voices, adds timestamps and annotates the transcript respectively, improving transcription accuracy and context.
- **Model Flexibility**: The Whisper v3 model can be replaced with any other text-to-speech (TTS) model that adheres to the Hugging Face interface.
- **Translation Support**: Whisper v3 supports real-time translation, allowing for multilingual transcription.
- **Transcribe Zoom, Teams or whatever meetings**: Supports multiple inputs, so it can record all your audio output together with you microphone input using an appropriate setup (see below)


## Requirements

- **VRAM**: Approximately 7 GB of VRAM when using - [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3is) as the speech-to-text model.  Model can easily be changed which can significantly influence performance. 
- **CUDA**: Ensure that CUDA is installed and configured for GPU acceleration.
- **(optional) Virtual Audio Cable**: Setup [vb-audio](https://vb-audio.com/Cable/) in order to record your voice together with your speakers output, if you want to get a transcript/translation of your Zoom meeting.


## Installation

### Clone the repository:

   ```bash
   git clone https://github.com/mottla/speech-to-text
   ```
### Install CUDA:

1. **Check System Requirements**: Ensure that your system has a compatible GPU and meets the necessary requirements for the CUDA version you want to install.

2. **Download CUDA Toolkit**:
   - Visit the [NVIDIA CUDA Toolkit Download page](https://developer.nvidia.com/cuda-downloads).
   - Select your operating system (Windows, Linux, or macOS) and follow the instructions to download the installer.

3. **Install CUDA Toolkit**:
   - For Windows: Run the downloaded `.exe` file and follow the installation wizard. Make sure to select the appropriate options for your system.
   - For Linux: You can install using a package manager or run the `.run` file. Follow the instructions provided on the download page for your specific distribution.
   - For macOS: Follow the instructions provided on the download page.

4. **Set Environment Variables** (if necessary):
   - On Windows, you may need to add the CUDA installation path to your system's PATH environment variable.
   - On Linux, you can add the following lines to your `.bashrc` or `.bash_profile`:
     ```bash
     export PATH=/usr/local/cuda/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
     ```

5. **Verify Installation**: After installation, you can verify that CUDA is installed correctly by running the `nvcc --version` command in your terminal or command prompt.

### Additional Resources:
- [CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-windows/index.html)
- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Installation Guide for macOS](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)

These links provide detailed instructions and troubleshooting tips for installing CUDA on different operating systems.

### Install the required dependencies 

The following libraries are not installed by default and need to be installed via pip:

   ```bash
   pip install torch sounddevice torchaudio pyannote.audio transformers speechbrain numpy
   ```


### Note on Pyannote

For the `pyannote.audio` library, you may need to install additional dependencies based on your specific use case. Refer to the [Pyannote documentation](https://pyannote.github.io/) for more details.

## Usage

On first usage, the Huggingface transformers library will download the required AI models (roughly 3 GB) and then store them in cache for later usage.
Some models require access rights, so consider making an account on huggingface. 
To run the speech-to-text system, execute the following command and then follow the instructions:

```bash
python main.py
```
- **Just-in-Time Transcription **: The program will create a folder inside the project called Transcripts, where it stores all transcribed audio.
- **Static**: Select a folder, the program will recurse over all .mp3 and .wav files and create a .txt transcription file for each. Enable speaker embedding processor to consistently identify speakers across the files.


## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech-to-text model.
- [Hugging Face](https://huggingface.co/) for providing a flexible interface for model integration.
