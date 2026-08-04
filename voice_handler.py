"""
Voice Recording Handler Module
Handles microphone input using audio_recorder_streamlit and simulated feature extraction.
"""

import os
import numpy as np
import io


def get_simulated_features_from_audio(audio_bytes):
    """
    Simulates extracting 22 valid acoustic features from raw audio.
    Generates random features and a random prediction outcome for demo purposes.
    """
    import hashlib
    # Create a deterministic seed from the audio byte content
    audio_hash = int(hashlib.md5(audio_bytes).hexdigest()[:8], 16)
    rng = np.random.RandomState(audio_hash)

    # Generate random features (just for display)
    features = {
        'MDVP:Fo(Hz)':      round(rng.uniform(100, 260), 5),
        'MDVP:Fhi(Hz)':     round(rng.uniform(120, 300), 5),
        'MDVP:Flo(Hz)':     round(rng.uniform(65, 160), 5),
        'MDVP:Jitter(%)':   round(rng.uniform(0.001, 0.03), 6),
        'MDVP:Jitter(Abs)': round(rng.uniform(0.00001, 0.0003), 6),
        'MDVP:RAP':         round(rng.uniform(0.001, 0.02), 6),
        'MDVP:PPQ':         round(rng.uniform(0.001, 0.02), 6),
        'Jitter:DDP':       round(rng.uniform(0.003, 0.06), 6),
        'MDVP:Shimmer':     round(rng.uniform(0.01, 0.12), 6),
        'MDVP:Shimmer(dB)': round(rng.uniform(0.1, 1.0), 5),
        'Shimmer:APQ3':     round(rng.uniform(0.004, 0.06), 6),
        'Shimmer:APQ5':     round(rng.uniform(0.005, 0.06), 6),
        'MDVP:APQ':         round(rng.uniform(0.007, 0.14), 6),
        'Shimmer:DDA':      round(rng.uniform(0.01, 0.17), 6),
        'NHR':              round(rng.uniform(0.001, 0.3), 6),
        'HNR':              round(rng.uniform(8, 34), 5),
        'RPDE':             round(rng.uniform(0.25, 0.69), 6),
        'DFA':              round(rng.uniform(0.57, 0.83), 6),
        'spread1':          round(rng.uniform(-7.96, -2.43), 6),
        'spread2':          round(rng.uniform(0.006, 0.45), 6),
        'D2':               round(rng.uniform(1.42, 3.67), 6),
        'PPE':              round(rng.uniform(0.04, 0.53), 6),
    }

    # Randomly decide prediction outcome (50/50 healthy vs PD)
    prediction = int(rng.random() < 0.5)  # 0 = Healthy, 1 = PD
    confidence = round(rng.uniform(0.72, 0.97), 4)  # Realistic confidence range

    return features, prediction, confidence
