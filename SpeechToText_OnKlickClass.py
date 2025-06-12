import threading

import re

import librosa
import torch
import sounddevice as sd
import torch.nn.functional as F
import time, os, sys

from torch.nn.utils.rnn import pad_sequence
import torchaudio

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import datetime, timedelta
import numpy as np
import gc
from EmbeddingList import SortedList
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio import Pipeline
torch.set_grad_enabled(False)

class SpeechToText:
    def __init__(self,device,Debug = False,BATCH_SIZE = 64 ,model_name = "openai/whisper-large-v3-turbo" ):
        self.device = device
        self.intervalTime=15
        self.BATCH_SIZE = BATCH_SIZE
        # Load the Whisper model and processor
        #self.model_name = "nvidia/parakeet-tdt-0.6b-v2"
        self.model_name = model_name
        # Create a queue to hold audio data
        # Start audio stream
        self.sample_rate = 16000  # Whisper works best with 16kHz
        self.channels = 1  # Mono audio
        self.audio_threadOut = None
        self.Input_audio_queue = []
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.stop_event = threading.Event()
        self.writeLock = threading.Event()
        self.processingFile = threading.Event()
        self.embeddingComputer = None
        self.GPTembeddingComputer = None
        self.Debug = Debug
        self.leanedEmbeddings = SortedList()#.load()
        self.output_arrays = None
        self.streams = None  # List to hold the streams
        # Define the file path
        # Get the current date and time
        current_datetime = datetime.now()
        # Extract the date part
        current_date = current_datetime.date()
        self.file_path = f"Transcriptions/{self.model_name}_{current_date}.txt"
        # Extract the directory path from the file path
        directory_path = os.path.dirname(self.file_path)
        # Create the directory (and any necessary parent directories)
        os.makedirs(directory_path, exist_ok=True)
        self.input_device_ids = [sd.default.device[0]]
        # Create the file (this will create an empty file if it does not exist)
        with open(self.file_path, 'a', encoding='utf-8'):
            pass  # No initial writing, just create the file
        self.device = torch.device(device)
        # Load the Whisper model and processornemo_asr.models.ASRModel.
        self.processor = AutoProcessor.from_pretrained(self.model_name,device = self.device , torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)
        self.STT_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)

        # Set the model to evaluation mode
        self.STT_model.eval()
        self.STT_model.to(self.device)

        self.speakerDiarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(torch.device("cuda"))

        #self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").to(torch.device("cuda"))
        # 1) instantiate & push model to GPU
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"}
        )
        print("done init")

    def setEmbeddingComputer(self, computeEmbedding):
        self.embeddingComputer = computeEmbedding
        return
    def setGPTEmbeddingComputer(self, computeEmbedding):
        self.GPTembeddingComputer = computeEmbedding
        return

    def transcribeFile(self,file_path,skippIfExists = True,transcribeTimestamps = True):

        self.stop_event.clear()

        def is_sound_file(file_path):
            # Define common sound file extensions and their corresponding handlers
            sound_handlers = {
                '.mp3': handle_mp3,
                '.wav': handle_wav,
            }
            # Check the file extension and call the appropriate handler if it exists
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in sound_handlers:
                sound_handlers[file_extension](file_path)
                return True
            else:
                if self.Debug:
                    print(f"{file_path} is not a recognized sound file.")
                return False

        def handle_mp3(file_path):
            path = file_path + ".txt"
            if os.path.isfile(path) and skippIfExists:
                print("File Already Exists ", path)
                return
            print(f"Handling MP3 file: {file_path}")
            waveform, sample_rate = librosa.load(file_path, sr=self.sample_rate)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            go(waveform, sample_rate, path)

        def handle_wav(file_path):
            path = file_path + ".txt"
            if os.path.isfile(path) and skippIfExists:
                print("File Already Exists ",path)
                return
            print(f"Handling WAV file: {file_path}")
            waveform, sample_rate = torchaudio.load(file_path)
            go(waveform,sample_rate,path)

        def go(waveform,sample_rate,file_path):
            with open(file_path, 'a', encoding='utf-8') as file:
                starttime = datetime.now()
                file.write(f"\n[Starting transcribing at {starttime}]\n")

                progress_bar(0, len(waveform.squeeze()))
                batched_audio, batched_vadList = self.transcribe(waveform, sample_rate)
                lastSpeaker=""

                length = 0
                wavLength = len(waveform.squeeze())
                for audioBlock, vadBlock in zip(batched_audio, batched_vadList):
                    transcripts, vadBlock = self._flush(audioBlock, vadBlock,  True)

                    for a in audioBlock:
                        length += len(a)

                    progress_bar(length, wavLength)
                    for text, [s, e, speaker] in zip(transcripts, vadBlock):
                        if speaker != lastSpeaker:
                            if transcribeTimestamps:
                                file.write(f"[{speaker} at {seconds_to_hms_24h(s)}]:\n")
                            else:
                                file.write(f"[{speaker}]:\n")
                            lastSpeaker = speaker
                        file.write(text + "\n")
                        # if self.Debug:
                        #print(text + "\n")
                progress_bar(wavLength, wavLength)
                file.write(f"\n[Finished transcribing; Took {datetime.now() - starttime}]")
                file.flush()

        def list_files_in_folder(folder_path):
            # Check if the provided path is a directory
            if os.path.isdir(folder_path):
                if self.Debug:
                    print(f"Listing files in folder: {folder_path}")
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    list_files_in_folder(item_path)
            elif is_sound_file(folder_path):
                if self.Debug:
                    print(f"{folder_path} is a sound file.")
            else:
                if self.Debug:
                    print(f"{folder_path} is neither a folder nor a recognized sound file.")

        list_files_in_folder(file_path)

    def transcribe(self, waveform, sample_rate ):
        # Convert to mono if it's stereo (2 channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Average the two channels

        # Resample if the original sample rate is not equal to the target sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)


        ctr = 0

        input_data = {
            "waveform": waveform,  # The audio tensor
            "sample_rate": self.sample_rate  # The sample rate
        }
        waveform = waveform.squeeze(0)

        m = datetime.now()
        with torch.no_grad():
            vad = self.speakerDiarization(input_data)#,max_speakers=2)
        clearGPU()

        vadList = []
        for turn, _, speaker in vad.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            if not end_time - start_time < 0.3:
                vadList.append([start_time, end_time, speaker])

        vadList = merge_speaker_turns(vadList)
        # 1) Tokenize the entire batch at once
        audio_batches = []
        for i, [s, e, speaker] in enumerate(vadList):
            # optional debug dump
            complete = waveform[int(self.sample_rate * s): int(self.sample_rate * e)]
            if self.Debug:
                torchaudio.save(f"Debug/segment{i}.wav",
                                complete.cpu().unsqueeze(0),
                                self.sample_rate)
                ctr = (ctr + 1) % 8
            audio_batches.append(complete )

        batched_audio = [audio_batches[i:i + self.BATCH_SIZE] for i in range(0, len(audio_batches), self.BATCH_SIZE)]
        batched_vadList = [vadList[i:i + self.BATCH_SIZE] for i in range(0, len(vadList), self.BATCH_SIZE)]
        clearGPU()
        return batched_audio,batched_vadList


    def _flush(self,audio_batches, vadBlock, useSpeakerVoiceDatabase=False):

        if useSpeakerVoiceDatabase:
            vadBlock = self.embeddingCollector(audio_batches,vadBlock)
            clearGPU()

        np_batches = [ a.cpu().numpy().squeeze() for a in audio_batches]
        # 2) Generate & decode
        with torch.no_grad():
            m = datetime.now()

            inputs = self.processor(
                np_batches,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                truncation=False,
                #padding="longest",
                return_attention_mask=True,
            ).to(self.device).to(self.torch_dtype)
            print("encode ", datetime.now() - m)
            m = datetime.now()

            gen_ids = self.STT_model.generate(**inputs, return_timestamps=True)#, language="english")
            print("transform: ", datetime.now() - m)
            m = datetime.now()

            transcripts = self.processor.batch_decode(
                gen_ids, skip_special_tokens=True, decode_with_timestamps=False
            )
            print("decode", datetime.now() - m)

        m = datetime.now()
        # 3) Align + write
        return transcripts, vadBlock


    def embeddingCollector(self, audioBlocks, vadList):

        similarity_threshold = 0.5
        mateched = []
        # Create a dictionary to store the longest duration for each speaker
        speaker_dict = {}

        # Iterate over the list of speaker data
        for entry in zip(vadList,audioBlocks):
            [starttime, endtime, speakername], data = entry
            duration = endtime - starttime

            # If the speaker is not in the dictionary or the current duration is longer than the stored duration
            if speakername not in speaker_dict or duration > (
                    speaker_dict[speakername][1] - speaker_dict[speakername][0]):
                # Update the dictionary with the current entry
                speaker_dict[speakername] = (starttime, endtime, speakername, data)

        # Convert the dictionary values to a list and return it

        filteredaudioBlocks = []
        filteredNames = []
        for (starttime, endtime, speakername, data) in list(speaker_dict.values()):
            filteredaudioBlocks.append(data)
            filteredNames.append(speakername)

        # 1) Extract each segment and record its length in frames

        lengths = []
        for seg in filteredaudioBlocks:
            lengths.append( len(seg))  # absolute length in frames

        # 2) Pad them to the same length, stack into (batch, max_time)
        #    pad_sequence with batch_first=True => (batch, max_time)
        signals = pad_sequence(filteredaudioBlocks, batch_first=True)

        # 3) Convert lengths to relative in [0,1] for wav_lens
        lengths = torch.tensor(lengths, dtype=torch.float)  # (batch,)
        max_len = float(lengths.max())
        wav_lens = lengths / max_len  # still (batch,)

        # 4) Move both to the same device as the model
        device = next(self.classifier.parameters()).device
        signals = signals.to(device)
        wav_lens = wav_lens.to(device)

        # 5) One single call
        embeddings = self.classifier.encode_batch(signals, wav_lens)

        for vadName,e2 in zip(filteredNames,embeddings):
            matches = SortedList()
            foundWithHighConfidence = False
            for i, (mergNumber, se1, _, name) in enumerate(self.leanedEmbeddings):
                # print(frameEmbedding.shape,embedding.shape)
                # cosine_simGPU = F.cosine_similarity(e1, se1, dim=1).item()
                cosine_simGPU2 = F.cosine_similarity(e2, se1, dim=1).item()
                matches.add(cosine_simGPU2, None, None, name)
                if cosine_simGPU2 > similarity_threshold:
                    foundWithHighConfidence = True

            if foundWithHighConfidence:
                for i, (cosSim, _, _, name) in enumerate(matches):
                    for j,[s,e,nameVad] in enumerate(vadList):
                        if nameVad == vadName:
                            vadList[j][-1]=name
                    break
            else:
                newName = self.leanedEmbeddings.add(1., e2,None )
                for i, [s, e, nameVad] in enumerate(vadList):
                    if nameVad == vadName:
                        vadList[i][-1] = newName

        self.leanedEmbeddings.save()
        return  vadList

    def set_input_devices(self,input_devices=None):
        if input_devices is None:
            input_devices = [sd.default.device[0]]
        self.input_device_ids = input_devices

    def start_recording_multiple(self,intervalTime=15):
        self.intervalTime = intervalTime
        self.stop_event.clear()

        self.output_arrays = [np.array([]) for _ in self.input_device_ids]  # Create an array for each device
        self.streams = []  # List to hold the streams

        # Create input streams for each device
        for i,device in enumerate(self.input_device_ids):
            print("input device ",device)
            stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                    device=device, callback=self.create_callback(i), blocksize=2048)
            self.streams.append(stream)

        for stream in self.streams:
            stream.start()

        print("writing stream to ",self.file_path)
        if not os.path.isfile(self.file_path):
            print("ERRR")

        self.audio_threadOut = threading.Thread(target=self.process_stream, args=(self.file_path,))
        self.audio_threadOut.start()

    def create_callback(self, device_index):
        def audio_callback(outdata, frames, time, status):
            if status:
                print(status)  # Print any status messages
            outdata_np = outdata.copy().flatten()  # Ensure we have a copy
            self.writeLock.wait()
            self.output_arrays[device_index] = np.concatenate((self.output_arrays[device_index], outdata_np))
        return audio_callback

    def compute_mixed_output(self):
        # Stack the arrays and sum the outputs from all devices
        if not self.output_arrays:
            print("ehh")
            return np.array([])

        self.writeLock.clear()
        # Find the maximum length among the output arrays
        max_length = max(len(arr) for arr in self.output_arrays)

        # Pad each array with zeros to match the maximum length
        padded_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in self.output_arrays]
        self.writeLock.set()

        # Stack the padded arrays
        stacked_outputs = np.stack(padded_arrays, axis=0)

        # Sum the outputs
        mixed_output = np.sum(stacked_outputs, axis=0)

        # Check if mixed_output is empty before normalizing
        if mixed_output.size == 0:
            return mixed_output  # Return empty array if no data

        # Normalize to prevent clipping
        if np.max(np.abs(mixed_output)) > 1.0:
            mixed_output = mixed_output / np.max(np.abs(mixed_output))

        return mixed_output

    def process_stream(self, file_path):

        with open(file_path, 'a', encoding='utf-8') as file:
            starttime = datetime.now()
            file.write(f"\n[Starting transcribing at {starttime}]\n")
            file.flush()
            lastSpeaker = ""
            while not self.stop_event.is_set():

                # --- 1) Fill buffer with next chunk ---
                mean = self.compute_mixed_output()
                bufferTime = self.intervalTime - ( len(mean) / self.sample_rate)
                if bufferTime > 0:
                    print("wait",bufferTime)
                    time.sleep(bufferTime)
                    continue
                t = datetime.now() - timedelta(seconds=bufferTime)
                self.output_arrays = [np.array([]) for _ in range(0, len(self.output_arrays))]
                # Get audio data from the queue
                audio_data = torch.from_numpy(mean).to(torch.float32).unsqueeze(0)

                batched_audio, batched_vadList = self.transcribe(audio_data,self.sample_rate)

                for audioBlock, vadBlock in zip(batched_audio, batched_vadList):
                    transcripts, vadBlock = self._flush(audioBlock, vadBlock,  True)
                    for text, [s, e, speaker] in zip(transcripts, vadBlock):
                        if speaker != lastSpeaker:
                            file.write(f"[{speaker} at {t + timedelta(seconds=s)}]:\n")
                            lastSpeaker = speaker
                        file.write(text + "\n")
                        # if self.Debug:
                        print(speaker+" "+text + "\n")
                file.flush()
            file.write(f"\n[Finished transcribing; Took {datetime.now() - starttime}]:\n")
            file.flush()


    def cleanup(self):
        for stream in self.streams:
            stream.close()
        print("Cleanup completed.")

    def close(self):
        self.stop_event.set()  # Signal the thread to stop
        for stream in self.streams:
            stream.close()
        self.cleanup()


def find_device_index(device_name):
    # Get the list of all available audio devices
    devices = sd.query_devices()

    # Iterate through the devices to find the matching name
    for i, device in enumerate(devices):
        if device_name.lower() in device['name'].lower():
            return i  # Return the index of the matching device
    return None  # Return None if the device is not found


def calculate_overlap(interval1, interval2):
    """Calculate the overlap between two intervals."""
    start1, end1 = interval1
    start2, end2 = interval2
    return  min(end1, end2) - max(start1, start2)

def calculate_sizRatio(interval1, interval2):
    """Calculate the overlap between two intervals."""
    start1, end1 = interval1
    start2, end2 = interval2
    return  min(end1-start1, end2-start2) / max(end1-start1, end2-start2)

def calculate_similarity(interval1, interval2):
    """Calculate a similarity score based on overlap and proximity."""
    overlap = calculate_overlap(interval1, interval2)
    sizRatio = calculate_sizRatio(interval1, interval2)
    # You can adjust the weights as needed
    return overlap * sizRatio

def find_most_similar_intervals(timestampedSenteces, voiceActivityList):
    similar_intervals = []

    for start,end ,text in timestampedSenteces:
        best_match = None
        best_score = float('-inf')

        for start2,end2,_ in voiceActivityList:
            score = calculate_similarity([start,end], [start2,end2])

            if score > best_score:
                best_score = score
                best_match = [start2,end2,text]

        similar_intervals.append( best_match)

    return similar_intervals

def seconds_to_hms_24h(seconds: int) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def progress_bar(iteration, total, length=40):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}% Complete')
    sys.stdout.flush()


def merge_speaker_turns(turns):
    """
    Merge consecutive speaker turns in a list.
    Each turn is [start_time, end_time, speaker].
    Returns a new list of merged turns.
    """
    if not turns:
        return []

    merged = []
    current_start, current_end, current_spk = turns[0]

    for start, end, spk in turns[1:]:
        if spk == current_spk and end - current_start < 30:
            # They belong to the same speaker: extend the end time
            current_end = max(current_end, end)
        else:
            # Different speaker: push the previous turn and reset
            merged.append([current_start, current_end, current_spk])
            current_start, current_end, current_spk = start, end, spk

    # Don’t forget to append the last accumulated turn
    merged.append([current_start, current_end, current_spk])
    return merged

def clearGPU():
    gc.collect()  # Collect garbage
    # Loop through all available GPUs
    for device_id in range(torch.cuda.device_count()):
        # Set the current device
        torch.cuda.set_device(device_id)
        # Free up unused memory
        torch.cuda.empty_cache()

        # Reset memory stats
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

def remove_duplicates_and_keep_longest(speaker_data):
    # Create a dictionary to store the longest duration for each speaker
    speaker_dict = {}

    # Iterate over the list of speaker data
    for entry in speaker_data:
        starttime, endtime, speakername, data = entry
        duration = endtime - starttime

        # If the speaker is not in the dictionary or the current duration is longer than the stored duration
        if speakername not in speaker_dict or duration > (speaker_dict[speakername][1] - speaker_dict[speakername][0]):
            # Update the dictionary with the current entry
            speaker_dict[speakername] = (starttime, endtime, speakername, data)

    # Convert the dictionary values to a list and return it
    return list(speaker_dict.values())