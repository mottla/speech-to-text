import os

import torch
from pynput import keyboard
import SpeechToText_OnKlickClass
import sounddevice as sd

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
                return False
            elif key == keyboard.Key.f4:
                if not key_pressed:
                    key_pressed = True
                    print("Start transcribing...")
                    SpeechToText.start_recording_multiple(10)
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
    SpeechToText = SpeechToText_OnKlickClass.SpeechToText(device1,  True)

    print("Done Loading AIs: Start")

    while True:
        user_input = input(
            """
            if you want to transcribe a soundfile or all soundfiles within a folder, type a valid file/folerpath, \n
            to display and select one or more input devices press 'i'. \n
            Start listener push 'r'.  \n
            close with c
            """)
        if os.path.exists(user_input.strip('"')):
            file_path = user_input.strip('"')
            print(f"Transcribing the file at {file_path}...")
            SpeechToText.transcribeFile(file_path, 15)
            # Here you would add the transcription logic
            print("Transcription complete!")
        elif user_input.lower() == "i":
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                print(
                    f"Device {i}: {device['name']} - Input Channels: {device['max_input_channels']}, Output Channels: {device['max_output_channels']}")
                devices_input = input(
                    "type the device numbers that should jointly record and transcribe (example: '0 3' without ' ) ")
                try:
                    # Split the input string by spaces and convert to a list of integers
                    devicesN = [int(num) for num in devices_input.split()]
                    SpeechToText.set_input_devices(devicesN)
                    print("Selected:", devicesN)
                    break
                except ValueError:
                    print("Invalid input. Please enter numbers separated by spaces.")
        elif user_input.lower() == "r":
            print("Key listener is active. Start/Pause transcription by pressing f4. Close program with ESC")

            # Get the default input device

            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
        elif user_input.lower() == "c":
            print("shutting down")
            break




