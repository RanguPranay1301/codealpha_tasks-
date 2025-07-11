import os
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Step 1: Load MIDI files and extract notes
def get_notes(midi_folder):
    notes = []
    for file in os.listdir(midi_folder):
        if file.endswith(".mid"):
            midi = converter.parse(os.path.join(midi_folder, file))
            notes_to_parse = None
            try:
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Step 2: Prepare sequences
def prepare_sequences(notes, sequence_length=100):
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    network_output = to_categorical(network_output)

    return network_input, network_output, note_to_int, pitchnames

# Step 3: Build the model
def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 4: Train the model
def train_model(model, network_input, network_output):
    filepath = "weights-improvement.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(network_input, network_output, epochs=50, batch_size=64, callbacks=[checkpoint])

# Step 5: Generate music
def generate_music(model, network_input, note_to_int, pitchnames, output_file="output.mid", length=500):
    int_to_note = {number: note for note, number in note_to_int.items()}
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(pitchnames))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, [[index]], axis=0)
        pattern = pattern[1:]

    create_midi(prediction_output, output_file)

# Convert prediction to MIDI file
def create_midi(prediction_output, output_file):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

# MAIN FLOW
if __name__ == "__main__":
    midi_folder = "midi_songs"  # Place MIDI files in this folder
    notes = get_notes(midi_folder)
    network_input, network_output, note_to_int, pitchnames = prepare_sequences(notes)
    model = create_model(network_input, len(pitchnames))
    train_model(model, network_input, network_output)
    generate_music(model, network_input, note_to_int, pitchnames)
