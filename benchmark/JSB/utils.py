import numpy as np
import pretty_midi
from IPython.display import Audio


def chorale_to_midi(piano_roll, min_note=0, tempo=120, time_per_step=0.25):
    """Convert chorale to MIDI."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0, name="Piano")

    piano_roll_np = np.array(piano_roll)
    timesteps, num_keys = piano_roll_np.shape

    # Track which notes are currently active
    # key_index -> start_time
    active_notes = {}

    for t in range(timesteps):
        current_time = t * time_per_step

        # Check each key
        for key_idx in range(num_keys):
            is_active = piano_roll_np[t, key_idx] > 0.5  # Threshold for binary
            pitch = key_idx + min_note

            if is_active:
                if key_idx not in active_notes:
                    # Start new note
                    active_notes[key_idx] = current_time
            else:
                if key_idx in active_notes:
                    # End note - create Note object
                    start_time = active_notes[key_idx]
                    note = pretty_midi.Note(
                        velocity=80, pitch=pitch, start=start_time, end=current_time
                    )
                    instrument.notes.append(note)
                    del active_notes[key_idx]

    # Close any remaining notes at the end
    end_time = timesteps * time_per_step
    for key_idx, start_time in active_notes.items():
        pitch = key_idx + min_note
        note = pretty_midi.Note(
            velocity=80, pitch=pitch, start=start_time, end=end_time
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def play_chorale(chorale_data, tempo=120, sample_rate=44100):
    """
    Convert chorale to audio and return IPython Audio widget.

    Args:
        chorale_data: (timesteps, 4) array with MIDI note numbers
        tempo: BPM
        sample_rate: Audio sample rate

    Returns:
        IPython.display.Audio object
    """
    # Convert to MIDI
    midi = chorale_to_midi(chorale_data, tempo=tempo)

    # Synthesize to audio
    # audio_data = midi.fluidsynth(fs=sample_rate)
    audio_data = midi.synthesize(
        fs=sample_rate, wave=lambda x: np.sin(2 * x) + 0.25 * np.cos(x)
    )

    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Return Audio widget
    return Audio(audio_data, rate=sample_rate)


def play_prediction(model_output, prob=0.5):
    """
    Play model's generated chorale.

    Args:
        model_output: Model predictions (logits or probabilities)
        temperature: Sampling temperature
    """
    # Convert to MIDI notes (depends on your encoding)
    # If output is logits for note classes:
    predicted_notes = model_output > prob

    # Convert to numpy
    predicted_notes = np.array(predicted_notes)

    # Create and play MIDI
    play_chorale(predicted_notes, tempo=120)
