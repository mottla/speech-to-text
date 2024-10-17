import os

import torch
from pynput import keyboard
import SpeechToText_OnKlickClass
import pyperclip

def on_press(key):
    global key_pressed
    try:
        # Add the key to the set of pressed keys
        if hasattr(key, 'char') and key.char is not None:
            print(f'Character key pressed: {key.char}')
        else:
            print(f'Special key pressed: {key}')
            if key == keyboard.Key.esc:
                print("shutting down")
                SpeechToText.close()
                #FakeSpeech.close()
                return False
            elif key == keyboard.Key.f2:
                selected_text = pyperclip.paste()
                print("Text to Speech: ", selected_text)
                #FakeSpeech.generate_audio_and_play(text=selected_text, language="en")
            elif key == keyboard.Key.f4:
                if not key_pressed:
                    key_pressed = True
                    print("Start transcribing...")
                    SpeechToText.start_recording()
                else:
                    print("Stop transcribing ...")
                    key_pressed = False
                    SpeechToText.close()
    except AttributeError:
        # Handle special keys (like Shift, Ctrl, etc.)
        print(f'Special key pressed: {key}')

if __name__ == '__main__':
    # Variable to track if the key is pressed
    key_pressed = False

    device1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device2 = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    print("Start")
    SpeechToText = SpeechToText_OnKlickClass.SpeechToText(device1, 6, False)
    # FakeSpeech = FakeSpeechClass.TextToSpeech(device2)
    # SpeechToText.setEmbeddingComputer(FakeSpeech.compute_embedding)
    # SpeechToText.setGPTEmbeddingComputer(FakeSpeech.compute_GPTembedding)

    print("Done Loading AIs: Start")

    # Prompt the user for input
    user_input = input(
        "if you want to transcribe a file, type a valid filepath here, otherwise type anything to continue interactive mode: ")
    # Check if the user provided a file path or pressed 's'
    if os.path.exists(user_input):
        file_path = user_input
        print(f"Transcribing the file at {file_path}...")
        SpeechToText.transcribeFile(file_path)
        # Here you would add the transcription logic
        print("Transcription complete!")
    else:
        print("Continuing with the next steps...")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
