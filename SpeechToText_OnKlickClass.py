import threading

import re

import librosa
import torch
import sounddevice as sd
import torch.nn.functional as F
import time, os, sys
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from select import error
from tensorflow.python.ops.numpy_ops.np_dtypes import float32
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import datetime
import numpy as np

from EmbeddingList import SortedList
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio import Pipeline
torch.set_grad_enabled(False)

class SpeechToText:
    def __init__(self,device,Debug = False):
        self.device = device
        self.intervalTime=15
        # Load the Whisper model and processor
        #self.model_name = "openai/whisper-large-v3"
        self.model_name = "openai/whisper-large-v3-turbo"
        # Create a queue to hold audio data
        # Start audio stream
        self.sample_rate = 16000  # Whisper works best with 16kHz
        self.channels = 1  # Mono audio
        self.audio_threadOut = None
        self.Input_audio_queue = []
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.stop_event = threading.Event()
        self.writeLock = threading.Event()
        self.force_processing = threading.Event()
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
        # Load the Whisper model and processor
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

    def transcribe(self,waveform,sample_rate,path):
        # Convert to mono if it's stereo (2 channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Average the two channels

        def progress_bar(iteration, total, length=40):
            percent = (iteration / total) * 100
            filled_length = int(length * iteration // total)
            bar = '█' * filled_length + '-' * (length - filled_length)
            sys.stdout.write(f'\r|{bar}| {percent:.2f}% Complete')
            sys.stdout.flush()

        with open(path, 'a', encoding='utf-8'):
            pass  # No initial writing, just create the file
        # Resample if the original sample rate is not equal to the target sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()
        current_length = waveform.shape[0]

        BLOCKSIZE = self.sample_rate * self.intervalTime
        # Calculate the required length to be the smallest multiple of BLOCKSIZE
        required_length = ((current_length + BLOCKSIZE - 1) // BLOCKSIZE) * BLOCKSIZE
        # Pad the waveform with zeros if necessary
        if required_length > current_length:
            padding = required_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        self.stop_event.clear()
        self.audio_threadOut = threading.Thread(target=self.processAudioOut,args=(path,))
        self.force_processing.set()

        self.audio_threadOut.start()
        for i in range(0, len(waveform), BLOCKSIZE):
            self.output_arrays = [waveform[i:i + BLOCKSIZE].numpy()]
            while len(self.output_arrays[0]) != 0:
                time.sleep(0.1)
            progress_bar(i, len(waveform))
        progress_bar(len(waveform), len(waveform))
        print("set stop")
        self.stop_event.set()
        self.force_processing.clear()
        self.audio_threadOut.join()



    def transcribeFile(self,file_path,intervalTime = 15,skippIfExists = True):

        self.stop_event.clear()
        self.intervalTime = intervalTime

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
            print(f"Handling MP3 file: {file_path}")
            path = file_path + ".txt"
            if os.path.isfile(path) and skippIfExists:
                return
            waveform, sample_rate = librosa.load(file_path, sr=self.sample_rate)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            self.transcribe(waveform, sample_rate,path)

        def handle_wav(file_path):
            print(f"Handling WAV file: {file_path}")
            path = file_path + ".txt"
            if os.path.isfile(path) and skippIfExists:
                print("File Already Exists ",path)
                return
            waveform, sample_rate = torchaudio.load(file_path)
            self.transcribe(waveform, sample_rate,path)


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


    def set_input_devices(self,input_devices=None):
        if input_devices is None:
            input_devices = [sd.default.device[0]]
        self.input_device_ids = input_devices

    def start_recording_multiple(self, interval_time):

        self.stop_event.clear()
        self.intervalTime = interval_time
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

        self.audio_threadOut = threading.Thread(target=self.processAudioOut,args=(self.file_path))
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



    def start_recording(self,interval_time = 10):
        self.stop_event.clear()
        self.intervalTime = interval_time
        self.output_arrays = [np.array([])]
        # List all available audio devices
        #devices = sd.query_devices()
        #for i, device in enumerate(devices):
        #   print( f"Device {i}: {device['name']} - Input Channels: {device['max_input_channels']}, Output Channels: {device['max_output_channels']}")

        # Get the default input device
        default_input_device = find_device_index("CABLE Output (VB-Audio Virtual")

        # Get the default input device
        #default_input_device = sd.default.device[0]  # The first element is the input device
        device_info = sd.query_devices(default_input_device)
        print(f"Listening to input device: {device_info['name']}")

        # Callback function for output (audio device)
        def audio_callback(outdata, frames, time, status):
            if status:
                print(status)  # Print any status messages
            outdata_np = outdata.copy().flatten()  # Ensure we have a copy
            self.Output_audio_queue = np.concatenate( (self.Output_audio_queue , outdata_np))


        # Create input and output streams with specified devices
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                     device=default_input_device, callback=audio_callback ,blocksize=1024)

        self.stream.start()
        self.audio_threadOut = threading.Thread(target=self.processAudioOut,args=(self.file_path))
        self.audio_threadOut.start()

    def processAudioOut(self, file_path):
        """
        Read streaming audio chunks, accumulate fixed‐length intervals,
        run speaker diarization, then batch ASR over every BATCH_SIZE intervals.
        """
        BATCH_SIZE = 64


        ctr = 0
        with open(file_path, 'a', encoding='utf-8') as file:
            file.seek(0, 2)
            starttime = datetime.now()
            file.write(f"\n\n[Starting transcribing at {starttime}]\n")
            lastSpeaker = ""
            # these lists will accumulate BATCH_SIZE items before running ASR
            batch_audio = []
            batch_vad_lists = []
            batch_timestamps = []  # (optional) if you care about global offsets
            ts_pattern = re.compile(r"<\|(\d+\.\d+)\|>\s*(.*?)<\|(\d+\.\d+)\|>")

            # helper to flush whatever is in batch_audio right now
            def _flush_batch(lastSpeaker):

                if not batch_audio:
                    print("empty")
                    return

                # 1) Tokenize the entire batch at once
                np_list = [t.cpu().numpy().squeeze() for t in batch_audio]
                print([ len(x)/self.sample_rate for x in np_list ])


                # 2) Generate & decode
                with torch.no_grad():
                    m = datetime.now()

                    inputs = self.processor(
                        np_list,
                        sampling_rate=self.sample_rate,
                        return_attention_mask=True,
                        return_tensors="pt"
                    ).to(self.device).to(self.torch_dtype)
                    print("encode ",datetime.now()-m)
                    m = datetime.now()

                    gen_ids = self.STT_model.generate(**inputs, return_timestamps=True, language="english")
                    print("transform: ", datetime.now()-m)
                    m = datetime.now()

                    transcripts = self.processor.batch_decode(
                        gen_ids, skip_special_tokens=True, decode_with_timestamps=True
                    )
                    print("decode", datetime.now()-m)

                m = datetime.now()
                # 3) Align + write
                for audioSegment, vadList, transcript in zip(batch_audio, batch_vad_lists, transcripts):
                    sentencesFromTransformer = ts_pattern.findall(transcript)
                    words = [[float(s), float(e), txt.strip()] for s, txt, e in sentencesFromTransformer]

                    segments = find_most_similar_intervals(words, vadList)


                    res2 = self.embeddingCollector2(audioSegment, segments)

                    for s, e, text, speaker in res2:
                        if speaker != lastSpeaker:
                            file.write(f"[{speaker}]:\n")
                            lastSpeaker = speaker
                        file.write(text + "\n")
                        if self.Debug:
                            print(text + "\n")
                print("Speaker Assignment + write to file: ", datetime.now() - m)


                file.flush()
                batch_audio.clear()
                batch_vad_lists.clear()
                return lastSpeaker

            # initialize an empty GPU buffer for raw audio
            buffer = torch.tensor([], device=self.device, dtype=torch.float32)

            while not self.stop_event.is_set():
                # --- 1) Fill buffer with next chunk ---
                mean = self.compute_mixed_output()

                bufferTime = self.intervalTime - ((len(buffer) + len(mean)) / self.sample_rate)
                if bufferTime > 0 and not self.force_processing.is_set():
                    time.sleep(bufferTime)
                    continue

                if len(buffer) + len(mean) >= self.intervalTime * self.sample_rate:
                    # Get audio data from the queue
                    audio_data = torch.cat((buffer, torch.from_numpy(mean).to(self.device)), dim=0).to(torch.float32)
                    if len(audio_data) == 0:
                        continue
                    # print(audio_data.shape)
                    self.output_arrays = [np.array([]) for _ in range(0, len(self.output_arrays))]

                    # audio_data = torch.from_numpy(audio_data).float().to(self.device)
                    # audio_data = self.classifier.audio_normalizer(audio_data, self.sample_rate)

                    input_data = {
                        "waveform": audio_data.unsqueeze(0),  # The audio tensor
                        "sample_rate": self.sample_rate  # The sample rate
                    }
                    with torch.no_grad():
                        vad = self.speakerDiarization(input_data)
                    timeline = vad.get_timeline().support()
                    vadList = []
                    for turn, _, speaker in vad.itertracks(yield_label=True):
                        start_time = turn.start
                        end_time = turn.end
                        vadList.append([start_time, end_time, speaker])

                    last_frame_was_cut = False
                    if len(timeline) != 0:
                        last_frame_was_cut = timeline[-1].end * self.sample_rate >= len(
                            audio_data) * 0.99
                        # audio_data = audio_data[vad.get_timeline().support()[0].start : ] #todo remove pre all noise

                    # non_silent_audio_frames, last_frame_was_cut = remove_all_silence(audio_data,self.sample_rate,35,0.2,self.Debug)
                    # non_silent_audio_frames, last_frame_was_cut = remove_silence_pt(audio_data,self.sample_rate, threshold=0.0001, min_silence_length=0.4,debug=True)
                    if len(timeline) == 0:
                        print("silence..")
                        continue
                    # print(buffer.shape,np.concatenate(non_silent_audio_frames).shape)
                    complete = torch.tensor([]).to(self.device)
                    if last_frame_was_cut:
                        # print("last frame was cut")
                        if len(timeline) > 1:
                            complete = audio_data[:int(self.sample_rate * timeline[-2].end)]
                            buffer = audio_data[int(self.sample_rate * timeline[-2].end):]
                        else:
                            complete = audio_data
                            if len(complete) < self.sample_rate * self.intervalTime * 2.:
                                buffer = complete
                                print("collecting more..")
                                continue
                            buffer = torch.tensor([]).to(self.device)
                    else:
                        complete = audio_data
                        buffer = torch.tensor([]).to(self.device)

                    # optional debug dump
                    if self.Debug:
                        torchaudio.save(f"Debug/segment{ctr}.wav",
                                        complete.cpu().unsqueeze(0),
                                        self.sample_rate)
                        ctr = (ctr + 1) % 8

                    # --- 4) Accumulate for batch ASR ---
                    batch_audio.append(complete)  # a 1-D FloatTensor on GPU
                    batch_vad_lists.append(vadList)

                    # when we've got enough segments, flush the batch
                    if len(batch_audio) >= BATCH_SIZE:
                        lastSpeaker = _flush_batch(lastSpeaker)
            # end of while → we've been asked to stop
            # Flush any leftovers:
            _flush_batch(lastSpeaker)
            file.write(f"\n\n[Finished transcribing; Took {datetime.now()-starttime}]\n")


    def processAudioOut2(self,file_path):
        ctr = 0
        with (open(file_path, 'r+', encoding='utf-8') as file):
            file.seek(0, 2)
            file.write(f"\n\n[Starting transcribing at {datetime.now()}]\n")
            lastSpeaker = ""
            ts_pattern    = re.compile(r"<\|(\d+\.\d+)\|>\s*(.*?)<\|(\d+\.\d+)\|>")
            try:
                buffer = torch.tensor([]).to(self.device)
                while not self.stop_event.is_set():  # Check if the stop event is set
                    mean = self.compute_mixed_output()

                    bufferTime = self.intervalTime - ((len(buffer) + len(mean)) / self.sample_rate)
                    if bufferTime > 0 and not self.force_processing.is_set():
                        time.sleep(bufferTime)

                    if len(buffer) + len(mean) >= self.intervalTime * self.sample_rate or self.force_processing.is_set():
                        # Get audio data from the queue
                        audio_data = torch.cat((buffer, torch.from_numpy(mean).to(self.device)), dim=0).to(torch.float32)
                        if len(audio_data)==0:
                            continue
                        #print(audio_data.shape)
                        self.output_arrays = [np.array([]) for _ in range(0,len(self.output_arrays))]

                        # audio_data = torch.from_numpy(audio_data).float().to(self.device)
                        #audio_data = self.classifier.audio_normalizer(audio_data, self.sample_rate)

                        input_data = {
                            "waveform": audio_data.unsqueeze(0),  # The audio tensor
                            "sample_rate": self.sample_rate  # The sample rate
                        }
                        with torch.no_grad():
                            vad = self.speakerDiarization(input_data)
                        timeline = vad.get_timeline().support()
                        vadList = []
                        for turn, _, speaker in vad.itertracks(yield_label=True):
                            start_time = turn.start
                            end_time = turn.end
                            vadList.append([start_time, end_time, speaker])

                        last_frame_was_cut = False
                        if len(timeline) != 0:
                            last_frame_was_cut = timeline[-1].end * self.sample_rate >= len(
                                audio_data) * 0.99
                            # audio_data = audio_data[vad.get_timeline().support()[0].start : ] #todo remove pre all noise

                        # non_silent_audio_frames, last_frame_was_cut = remove_all_silence(audio_data,self.sample_rate,35,0.2,self.Debug)
                        # non_silent_audio_frames, last_frame_was_cut = remove_silence_pt(audio_data,self.sample_rate, threshold=0.0001, min_silence_length=0.4,debug=True)
                        if len(timeline) == 0:
                            print("silence..")
                            continue
                        # print(buffer.shape,np.concatenate(non_silent_audio_frames).shape)
                        complete = torch.tensor([]).to(self.device)
                        if last_frame_was_cut:
                            #print("last frame was cut")
                            if len(timeline)>1:
                                complete =  audio_data[:int(self.sample_rate * timeline[-2].end)]
                                buffer = audio_data[int(self.sample_rate * timeline[-2].end):]
                            else:
                                complete = audio_data
                                if len(complete) < self.sample_rate * self.intervalTime * 2.:
                                    buffer = complete
                                    print("collecting more..")
                                    continue
                                buffer = torch.tensor([]).to(self.device)
                        else:
                            complete = audio_data
                            buffer = torch.tensor([]).to(self.device)

                        if self.Debug:
                            torchaudio.save(f'Debug\segment{ctr}.wav', complete.cpu().unsqueeze(0), self.sample_rate)
                            #print("saved to ", ctr)
                            ctr += 1
                            ctr = ctr % 8

                        #From here start changing code significantly!
                        #instead of running the processor on complete, store complete together with the speakerdiarization results vadList
                        #once we have n=4 batches, continue in the below logic
                        encodedAudio = self.processor(
                            complete.cpu().numpy().squeeze(),
                            sampling_rate=self.sample_rate,
                            return_attention_mask=True,
                            # padding="max_length",  # Change to "max_length" to pad to a specific length
                            # max_length=3000,  # Specify the maximum length for padding
                            truncation=False,
                            return_tensors="pt").to(self.device).to(self.torch_dtype)

                        gen_kwargs = {
                            # "max_new_tokens": 448,
                            # "num_beams": 1,
                            # "condition_on_prev_tokens": False,
                            # "compression_ratio_threshold": 1.35,
                            # zlib compression ratio threshold (in token space)
                            # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                            # "logprob_threshold": -1.0,
                            # "no_speech_threshold": 0.6,
                            "return_timestamps": True,
                            # "task": "translate"
                            "language": "english"
                        }

                        with torch.no_grad():
                            generated_ids = self.STT_model.generate(**encodedAudio, **gen_kwargs)
                            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True,
                                                                    decode_with_timestamps=True)[0]
                        # Find all matches
                        matches = ts_pattern.findall( transcription)

                        # Create tuples from matches
                        result = [[float(start), float(end), text.strip()] for start, text, end in matches]
                        # Split the audio based on diarization results

                        res = find_most_similar_intervals(result,vadList)
                        res2 = self.embeddingCollector2(complete, res)
                        for s,e,segment, speaker  in res2:
                            if speaker is not lastSpeaker:
                                file.write(f"[{speaker}]:")
                                file.write(f"\n")
                                lastSpeaker = speaker
                            if self.Debug:
                                print(speaker,segment)
                            file.write(segment)
                            file.write(f"\n")

                        file.flush()
            except KeyboardInterrupt:
                print("Stopped by user. (STRG+C)")
            finally:
                print("done")


    def cleanup(self):
        for stream in self.streams:
            stream.close()
        print("Cleanup completed.")

    def close(self):
        self.stop_event.set()  # Signal the thread to stop
        for stream in self.streams:
            stream.close()
        self.cleanup()


    def embeddingCollector2(self, audio_input1D, speech_intervalls):

        similarity_threshold = 0.3
        mateched = []

        # 1) Extract each segment and record its length in frames
        segments = []
        lengths = []
        for start, end, _text in speech_intervalls:
            sF = int(start * self.sample_rate)
            eF = int(end * self.sample_rate)
            seg = audio_input1D[sF:eF]  # -> (frame_length,)
            segments.append(seg)
            lengths.append(seg.size(0))  # absolute length in frames

        # 2) Pad them to the same length, stack into (batch, max_time)
        #    pad_sequence with batch_first=True => (batch, max_time)
        signals = pad_sequence(segments, batch_first=True)

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

        for [start, end ,text],e2 in zip(speech_intervalls,embeddings):
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
                    mateched.append([start, end ,text,name])
                    break
            else:
                newName = self.leanedEmbeddings.add(1., e2,None )
                mateched.append([start, end, text, newName])
        return  mateched


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