import os
from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import librosa
import pretty_midi
from pydub import AudioSegment
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load configurations from environment variables
API_KEY = os.environ.get('OPENAI_API_KEY')
# Initialize the GPT-3 API
openai.api_key = API_KEY

# Constants
TEMP_WAV_FILENAME = "temp.wav"
MIDI_EXTENSION = ".midi"


def save_midi_to_disk(midi, directory, filename):
    filepath = os.path.join(directory, filename)
    midi.write(filepath)
    return filepath


def get_filtered_midi_files(style_keywords):
    folder_path = os.path.join(os.getcwd(), "mids")
    all_files = os.listdir(folder_path)
    return [
        os.path.join(folder_path, f)
        for f in all_files
        if f.lower().endswith(".mid")
           and any(f.lower().startswith(keyword.lower()) for keyword in style_keywords)
    ]


def convert_text_to_midi(text_data):
    note_strings = text_data.strip().split('\n')
    output_midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False)

    for note_string in note_strings:
        note_values = note_string.split()
        if len(note_values) < 3:
            logging.warning(f"Invalid note format: {note_string}. Skipping.")
            continue

        pitch, start_time, end_time = map(int, note_values[:3])

        velocity = int(note_values[3]) if len(note_values) >= 4 else 100

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time / 1000.0,
            end=end_time / 1000.0
        )
        instrument.notes.append(note)

    output_midi.instruments.append(instrument)
    return output_midi


def generate_midi_text_from_files(midi_file_paths):
    combined_note_string = ""

    for midi_file_path in midi_file_paths:
        if not os.path.isfile(midi_file_path):
            continue

        try:
            midi = pretty_midi.PrettyMIDI(midi_file_path)
            for instrument in midi.instruments:
                for note in instrument.notes:
                    pitch = note.pitch
                    start_time = int(note.start * 1000)
                    end_time = int(note.end * 1000)
                    velocity = note.velocity
                    note_string = f"{pitch:03d} {start_time:03d} {end_time:03d} {velocity:03d}\n"
                    combined_note_string += note_string

        except Exception as e:
            logging.error(f"Error processing MIDI file '{midi_file_path}': {str(e)}")

    return combined_note_string


def get_midi_modifications_from_gpt3(bpm, text):
    prompt = f"Generate new MIDI data with a BPM of {bpm} using the following data as a template:\n\n{text}\n\nPlease provide the new MIDI data with the same number of lines."
    try:
        response = openai.Completion.create(
            model="text-davinci-002", prompt=prompt, max_tokens=1000
        )
        modification_instructions = response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Failed to get GPT-3 modifications: {str(e)}")
        modification_instructions = ""
    return modification_instructions


def get_bpm_from_wav(wav_file):
    try:
        y, sr = librosa.load(wav_file)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    except Exception as e:
        logging.error(f"Failed to extract BPM from WAV: {str(e)}")
        tempo = 0
    return tempo



@csrf_exempt
def upload(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

    try:
        uploaded_file = request.FILES["file"]
        style_list = request.POST.get('styleList')
        audio = AudioSegment.from_file(uploaded_file, format="wav")

        # Temporary save for processing
        audio.export(TEMP_WAV_FILENAME, format="wav")

        bpm = get_bpm_from_wav(TEMP_WAV_FILENAME)
        midi_text_sample = generate_midi_text_from_files(get_filtered_midi_files(style_list))
        gpt3_midi_text = get_midi_modifications_from_gpt3(bpm, midi_text_sample)
        midi_data = convert_text_to_midi(gpt3_midi_text)

        saved_filepath = save_midi_to_disk(midi_data, os.getcwd(), os.path.splitext(uploaded_file.name)[0] + MIDI_EXTENSION)

        if os.path.exists(saved_filepath):
            response = FileResponse(open(saved_filepath, "rb"), content_type="audio/x-midi")
            response["Content-Disposition"] = f"attachment; filename='{os.path.basename(saved_filepath)}'"
            return response
        else:
            return HttpResponse("File not found", status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

