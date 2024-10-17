import threading

import re
import torch
import sounddevice as sd
import torch.nn.functional as F
import time, os
from pyannote.audio.pipelines import VoiceActivityDetection
import torchaudio
from select import error
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import datetime
import numpy as np
from pyannote.audio import Model
from EmbeddingList import SortedList
from speechbrain.inference.speaker import EncoderClassifier


class SpeechToText:
    def __init__(self,device,intervalTime = 10,Debug = False):
        self.device = device
        self.intervalTime=intervalTime
        # Load the Whisper model and processor
        #self.model_name = "openai/whisper-large-v3"
        self.model_name = "openai/whisper-large-v3"
        # Create a queue to hold audio data
        # Start audio stream
        self.sample_rate = 16000  # Whisper works best with 16kHz
        self.channels = 1  # Mono audio
        self.input_stream = None
        self.stream = None
        self.audio_threadOut = None
        self.Input_audio_queue = []
        self.Output_audio_queue =   np.array([]) #torch.tensor([]).to(device)
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.stop_event = threading.Event()
        self.embeddingComputer = None
        self.GPTembeddingComputer = None
        self.Debug = Debug
        self.leanedEmbeddings = SortedList().load()

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

        # Create the file (this will create an empty file if it does not exist)
        with open(self.file_path, 'a', encoding='utf-8'):
            pass  # No initial writing, just create the file
        self.device = torch.device(device)
        # Load the Whisper model and processor
        self.processor = AutoProcessor.from_pretrained(self.model_name,device = self.device)
        self.STT_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
        # Set the model to evaluation mode
        self.STT_model.eval()
        self.STT_model.to(self.device)
        self.VAD_model = Model.from_pretrained(
            "pyannote/segmentation-3.0").to(self.device)
        self.VAD_model.eval()
        HYPER_PARAMETERS = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.1,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.5
        }
        self.VAD_pipeline = VoiceActivityDetection(segmentation=self.VAD_model).to(self.device)
        self.VAD_pipeline.instantiate(HYPER_PARAMETERS)
        #self.OSD_pipeline = OverlappedSpeechDetection(segmentation=self.VAD_model).to(self.device)
        #self.OSD_pipeline.instantiate(HYPER_PARAMETERS)
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        print("done init")



    def setEmbeddingComputer(self, computeEmbedding):
        self.embeddingComputer = computeEmbedding
        return
    def setGPTEmbeddingComputer(self, computeEmbedding):
        self.GPTembeddingComputer = computeEmbedding
        return

    def transcribeFile(self,file_path):
        self.stop_event.clear()
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono if it's stereo (2 channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Average the two channels

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

        self.audio_threadOut = threading.Thread(target=self.processAudioOut)
        self.audio_threadOut.start()


        for i in range(0,len(waveform),BLOCKSIZE):
            self.Output_audio_queue = waveform[i:i+BLOCKSIZE].numpy()
            while len(self.Output_audio_queue) is not 0:
                time.sleep(0.1)



    def start_recording(self):
        self.stop_event.clear()
        # List all available audio devices
        #devices = sd.query_devices()
        #for i, device in enumerate(devices):
        #   print( f"Device {i}: {device['name']} - Input Channels: {device['max_input_channels']}, Output Channels: {device['max_output_channels']}")

        # Get the default input device
        #default_input_device = find_device_index("CABLE Output (VB-Audio Virtual")

        # Get the default input device
        default_input_device = sd.default.device[0]  # The first element is the input device
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
        self.audio_threadOut = threading.Thread(target=self.processAudioOut)
        self.audio_threadOut.start()

    def processAudioOut(self):
        ctr = 0
        with (open(self.file_path, 'r+', encoding='utf-8') as file):
            file.seek(0, 2)
            file.write(f"\n\n [Starting transcribing at {datetime.now()}]\n")
            try:
                buffer = torch.tensor([]).to(self.device)
                while not self.stop_event.is_set():  # Check if the stop event is set
                    if len(buffer) + len(self.Output_audio_queue) >= self.intervalTime * self.sample_rate:
                        # Get audio data from the queue
                        audio_data = torch.cat(
                            (buffer, torch.from_numpy(self.Output_audio_queue).float().to(self.device)), dim=0)
                        self.Output_audio_queue = np.array([])  # torch.tensor([],device=self.device) #clear the queue

                        # audio_data = torch.from_numpy(audio_data).float().to(self.device)
                        audio_data = self.classifier.audio_normalizer(audio_data, self.sample_rate)

                        input_data = {
                            "waveform": audio_data.unsqueeze(0),  # The audio tensor
                            "sample_rate": self.sample_rate  # The sample rate
                        }

                        vad = self.VAD_pipeline(input_data)

                        # Create a list to hold the speech segments
                        speech_segments = []

                        # Extract speech segments based on the VAD annotations
                        for segment in vad.get_timeline().support():
                            start = int(segment.start * self.sample_rate)
                            end = int(segment.end * self.sample_rate)
                            speech_segments.append(audio_data[start:end])

                        last_frame_was_cut = False
                        if len(vad.get_timeline().support()) != 0:
                            last_frame_was_cut = vad.get_timeline().support()[-1].end * self.sample_rate >= len(
                                audio_data) * 0.99
                            # audio_data = audio_data[vad.get_timeline().support()[0].start : ] #todo remove pre all noise

                        # non_silent_audio_frames, last_frame_was_cut = remove_all_silence(audio_data,self.sample_rate,35,0.2,self.Debug)
                        # non_silent_audio_frames, last_frame_was_cut = remove_silence_pt(audio_data,self.sample_rate, threshold=0.0001, min_silence_length=0.4,debug=True)
                        if len(speech_segments) == 0:
                            print("silence..")
                            continue
                        # print(buffer.shape,np.concatenate(non_silent_audio_frames).shape)
                        complete = torch.tensor([]).to(self.device)
                        if last_frame_was_cut:
                            for i, s in enumerate(speech_segments):
                                complete = torch.cat((complete, s), dim=0)
                                if len(complete) >= self.sample_rate * self.intervalTime:
                                    buffer = audio_data[int(self.sample_rate * vad.get_timeline().support()[i].end):]
                                    break
                            if len(complete) < self.sample_rate * self.intervalTime:
                                buffer = torch.cat((buffer, audio_data), dim=0)
                                print("collecting more..")
                                continue
                        else:
                            complete = torch.cat(speech_segments, dim=0)
                            buffer = torch.tensor([]).to(self.device)

                        if self.Debug:
                            torchaudio.save(f'segment{ctr}.wav', complete, self.sample_rate)
                            ctr += 1
                            ctr = ctr % 8

                        encodedAudio = self.processor(
                            complete.cpu().numpy().squeeze(),
                            sampling_rate=self.sample_rate,
                            return_attention_mask=True,
                            # padding="max_length",  # Change to "max_length" to pad to a specific length
                            # max_length=3000,  # Specify the maximum length for padding
                            truncation=False,
                            return_tensors="pt").to(self.device)

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
                        }

                        with torch.no_grad():
                            generated_ids = self.STT_model.generate(**encodedAudio, **gen_kwargs)

                        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True,
                                                                    decode_with_timestamps=True)[0]

                        pattern = r"<\|(\d+\.\d+)\|>\s*(.*?)<\|(\d+\.\d+)\|>"
                        # Find all matches
                        matches = re.findall(pattern, transcription)
                        # Create tuples from matches
                        result = [[float(start), float(end), text.strip()] for start, text, end in matches]

                        speakers, delete = self.embeddingCollector(complete, result)
                        # assignments = assign_speakers_to_segments(speakers,transcription,vad.get_timeline().support()[0].start ,len(audio_data)/self.sample_rate)

                        for deleteName, newName in delete:
                            file.seek(0)
                            content = file.read()
                            # Replace all occurrences of X with Y
                            modified_content = content.replace(deleteName, newName)
                            # Move the cursor to the beginning of the file
                            file.seek(0)
                            # Write the modified content back to the file
                            file.write(modified_content)
                            # Truncate the file to the new size (in case the modified content is shorter)
                            file.truncate()
                            # Move the cursor to the end of the file
                            file.seek(0, 2)  # 0 is the offset, 2 means to seek from the end of the file

                        for s, e, segment, speaker in speakers:
                            s = f"{speaker}: \t\t {segment}"
                            print(s)
                            file.write(f"\n")
                            file.write(s)
                        file.flush()
                    else:
                        time.sleep(
                            self.intervalTime - ((len(buffer) + len(self.Output_audio_queue)) / self.sample_rate))

            except KeyboardInterrupt:
                print("Stopped by user. (STRG+C)")
            finally:
                self.cleanup()

    def stop_recording(self):
        self.stop_event.set()  # Signal the thread to stop
        self.stream.stop()
        self.cleanup()

    def cleanup(self):
        if self.stream is not None:
            self.stream.close()
        self.Output_audio_queue = np.array([])
        print("Cleanup completed.")

    def close(self):
        self.stop_recording()


    def embeddingCollector(self, audio_input1D, speech_intervalls):

        similarity_threshold = 0.7
        thresholBoosted= similarity_threshold
        if len(speech_intervalls)>0:
            thresholBoosted = similarity_threshold + (0.05 * similarity_threshold * int(np.log2(len(speech_intervalls))))
        #audio = inputData["waveform"].squeeze(0)
        #overlapps = self.OSD_pipeline(inputData).get_timeline().support()
        #overlapp_free_intervalls = get_single_speaker_annotations(speech_intervalls,overlapps)

        # Extract speech segments based on the VAD annotations

        def recurseTill(speech_intervalls,threshhold):
            if len(speech_intervalls) == 0:
                return  [], []
            if len(speech_intervalls) == 1:
                start, end ,_ = speech_intervalls[0]
                startFrame = int(start * self.sample_rate)
                endFrame = int(end * self.sample_rate)
                foundWithHighConfidence, matches, replaced ,embedding = self.fkt(audio_input1D[startFrame:endFrame],threshhold)
                for name, _ in replaced:
                    self.leanedEmbeddings.delete_by_name(name)
                    matches.delete_by_name(name)

                if not foundWithHighConfidence:
                    newName = self.leanedEmbeddings.add(1, None, embedding, )
                    bestC = 0
                    for i, (cosSim, _, _, name) in enumerate(matches):
                        bestC= cosSim
                        break
                    matches.add(1.-max(bestC, 0) , None, None, f"New Speaker {newName}")
                    #bestC = (bestC+1.)/2.  #scale cosin to [0,1] .. not mathematically correct to interpret it as probability anyway but good enough
                for i, (cosSim, _, _, name) in enumerate(matches):
                    speech_intervalls[0].append(f"[{name}]({round( cosSim *100.,2)}%)")
                    return  speech_intervalls, replaced
            if len(speech_intervalls) > 1:
                start, _,_ = speech_intervalls[0]
                _, end , _ = speech_intervalls[-1]
                foundWithHighConfidence, matches, replaced ,_ = self.fkt( audio_input1D[int(start * self.sample_rate):int(end * self.sample_rate)] ,threshhold )
                if foundWithHighConfidence:
                    #print("found early", len(speech_intervalls))
                    for name, _ in replaced:
                        self.leanedEmbeddings.delete_by_name(name)
                        matches.delete_by_name(name)
                    best = 0
                    for i, (cosine_simGPU2, _, _ , name) in enumerate(matches):
                        best = f"[{name}]({round(cosine_simGPU2*100.,2)}%)"
                        break
                    for i in range(0,len(speech_intervalls)):
                        speech_intervalls[i].append(best)
                    return  speech_intervalls, replaced
                length = len(speech_intervalls)
                first_half = speech_intervalls[:length // 2]
                second_half = speech_intervalls[length // 2:]
                #print("deeper", len(speech_intervalls))
                speech_intervallsL, replacedL = recurseTill(first_half,threshhold*0.8)
                speech_intervallsR, replacedR = recurseTill(second_half, threshhold*0.8)
                return  speech_intervallsL+speech_intervallsR, replacedL+replacedR

        speech_intervallsN, replacedN = recurseTill(speech_intervalls,  similarity_threshold)
        self.leanedEmbeddings.save()
        return speech_intervalls, replacedN

    def fkt(self,  frame, similarity_threshold):
        try:
            e2 = self.classifier.encode_batch(frame.unsqueeze(0))
        except error as err:
            print("err ",err)
            return
        matches = SortedList()
        foundWithHighConfidence = False
        for i, (mergNumber, se1, se2, name) in enumerate(self.leanedEmbeddings):
            # print(frameEmbedding.shape,embedding.shape)
            # cosine_simGPU = F.cosine_similarity(e1, se1, dim=1).item()
            cosine_simGPU2 = F.cosine_similarity(e2, se2, dim=2).item()
            matches.add(cosine_simGPU2, None, None, name)

        replaced = []
        for i in range(0, matches.len() - 1):
            (_, _, _, name1) = list(reversed(matches))[i]  # [3,2,1] ->  [1,2,3] -> 1
            (_, _, _, name2) = list(reversed(matches))[i + 1]
            (mergNumber1, _, se1, name1) = self.leanedEmbeddings.getByName(name1)
            (mergNumber2, _, se2, name2) = self.leanedEmbeddings.getByName(name2)
            cosine_simGPU2 = F.cosine_similarity(se1, se2, dim=2).item()
            if cosine_simGPU2 > similarity_threshold:
                foundWithHighConfidence = True
                s = mergNumber2 + mergNumber1 + 1.
                new_mean2 = 3. / s * torch.mean(torch.stack([se1 * mergNumber1, se2 * mergNumber2, e2]), dim=0)
                if mergNumber1 < mergNumber2:
                    replaced.append((name1, name2))
                    # self.leanedEmbeddings.update_by_index(i,new_mean1,new_mean2,mergNumber+cosine_simGPU2)
                    self.leanedEmbeddings.update_by_name(name2, None, new_mean2, s)
                else:
                    replaced.append((name2, name1))
                    self.leanedEmbeddings.update_by_name(name1, None, new_mean2, s)

        return foundWithHighConfidence, matches, replaced , e2


def get_single_speaker_annotations(speak_intervals, overlap_speakers_intervals):
    # Create a new Annotation object to hold the modified segments
    new_annotations = []

    for a_segment in speak_intervals:
        # Get the start and end of the current segment in A
        a_start, a_end = a_segment.start, a_segment.end

        # Initialize a variable to track the current start of the segment
        current_start = a_start

        # Check each segment in B
        for b_segment in overlap_speakers_intervals:
            # If the segment in B is contained within the segment in A
            if b_segment.start >= a_start and b_segment.end <= a_end:
                # Create a new segment from the current start to the start of the B segment
                if current_start < b_segment.start:
                    #new_annotations[Segment(current_start, b_segment.start)] = "unknown Speaker"
                    new_annotations.append((current_start, b_segment.start, "unknown Speaker"))

                # Update the current start to the end of the B segment
                current_start = b_segment.end

        # After checking all B segments, add the remaining part of A if any
        if current_start < a_end:
            new_annotations.append( (current_start, a_end, "unknown Speaker") )

    return new_annotations


def assign_speakers_to_segments(intervals, input_string, starttime ,total_duration):
    # Step 1: Split the string by all possible sentence endings
    segments = re.split(r'(?<=[.!?]) +', input_string)  # Split on . ! ? followed by space

    # Step 2: Calculate the duration per character
    duration_per_character = total_duration / len(input_string) if len(input_string) > 0 else 0

    # Step 3: Prepare to assign speakers
    speaker_assignments = []
    current_time = starttime

    speakerCenterTimes = []
    speakers = []
    for starttime, endtime, speaker in intervals:
        speakerCenterTimes.append( (endtime -starttime)/2. )
        speakers.append(speaker)

    speakerCenterTimes = np.array(speakerCenterTimes)

    speaker_assignments= []
    # Iterate through each segment
    for segment in segments:
        segment_length = len(segment)
        segment_duration = segment_length * duration_per_character
        segment_center = current_time+segment_duration/2.

        closestSpekerIndex = np.argmin(np.abs(speakerCenterTimes - segment_center))
        speaker_assignments.append( (segment,speakers[closestSpekerIndex]))

        # Update the current time
        current_time += segment_duration

    return speaker_assignments

def split_audio_into_chunks(audio, sample_rate, chunk_duration=1):

    chunk_size = sample_rate * chunk_duration
    chunks = []

    # Calculate the size of each part
    parts = len(audio) // chunk_size
    remainder_size = len(audio) % chunk_size

    # Split the tensor into equal parts
    for i in range(parts):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunks.append(audio[start_index:end_index])

    # If there is a remainder, store it in a list
    remainder = audio[parts * chunk_size:]
    if remainder_size > 0:
        chunks.append(remainder)

    return chunks

def find_device_index(device_name):
    # Get the list of all available audio devices
    devices = sd.query_devices()

    # Iterate through the devices to find the matching name
    for i, device in enumerate(devices):
        if device_name.lower() in device['name'].lower():
            return i  # Return the index of the matching device
    return None  # Return None if the device is not found



def remove_leading_silence(audio_data, silence_threshold=0.03):

    # Calculate the absolute values of the audio data
    abs_audio_data = np.abs(audio_data)

    # Determine the silence threshold based on the data type
    if np.issubdtype(audio_data.dtype, np.integer):
        max_value = np.iinfo(audio_data.dtype).max  # Maximum value for integer types
        silence_threshold = silence_threshold * max_value  # Scale threshold for integer types
    else:
        silence_threshold = silence_threshold  # Default threshold for float types
    # Find the index where the audio data exceeds the silence threshold
    first_sound_index = np.argmax(abs_audio_data > silence_threshold)

    # If no sound is detected, return the original audio data
    if first_sound_index == 0 and np.all(abs_audio_data <= silence_threshold):
        print("No sound")
        return []
    # Trim the audio data to remove leading silence
    trimmed_audio_data = audio_data[first_sound_index:]
    # Create a new audio array with only the non-silent parts
    #trimmed_audio = audio[start:end + 1]

    return trimmed_audio_data