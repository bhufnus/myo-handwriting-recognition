#!/usr/bin/env python3
"""
Test script to check if beeps are working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import threading

# Try to import sounddevice for square wave beeps
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ sounddevice available")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not available, using winsound for beeps")

import winsound

def play_square_wave_beep(frequency, duration_ms, volume=0.2):
    """Play a bell-like sine wave beep"""
    if SOUNDDEVICE_AVAILABLE:
        try:
            sample_rate = 48000
            duration_sec = duration_ms / 1000.0
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
            wave = np.sin(2 * np.pi * frequency * t)
            # Envelope: 10ms attack, 70% sustain, 20% release
            total_samples = len(wave)
            attack_samples = int(sample_rate * 0.01)  # 10ms
            sustain_samples = int(total_samples * 0.7)
            release_samples = total_samples - attack_samples - sustain_samples
            envelope = np.ones(total_samples)
            # Attack
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            # Sustain (already 1)
            # Release
            if release_samples > 0:
                envelope[-release_samples:] = np.linspace(1, 0, release_samples)
            # Apply envelope and volume
            audio_data = (wave * envelope * volume).astype(np.float32)
            print(f"Playing sounddevice beep: {frequency}Hz, {duration_ms}ms, volume={volume}")
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Error playing bell beep: {e}")
    else:
        try:
            print(f"Playing winsound beep: {frequency}Hz, {duration_ms}ms")
            winsound.Beep(int(frequency), int(duration_ms))
        except Exception as e:
            print(f"Error with winsound.Beep: {e}")

def test_beeps():
    """Test different beep types"""
    print("🔊 Testing Beep Functionality")
    print("=" * 40)
    
    # Test 1: Start beep (high frequency, short duration)
    print("\n1. Testing start beep (1200Hz, 100ms)...")
    play_square_wave_beep(1200, 100, volume=0.5)
    time.sleep(0.5)
    
    # Test 2: Stop beep 1 (800Hz, 80ms)
    print("\n2. Testing stop beep 1 (800Hz, 80ms)...")
    play_square_wave_beep(800, 80, volume=0.5)
    time.sleep(0.1)
    
    # Test 3: Stop beep 2 (600Hz, 80ms)
    print("\n3. Testing stop beep 2 (600Hz, 80ms)...")
    play_square_wave_beep(600, 80, volume=0.5)
    time.sleep(0.5)
    
    # Test 4: Different volumes
    print("\n4. Testing different volumes...")
    for vol in [0.1, 0.3, 0.5, 0.7, 1.0]:
        print(f"   Volume {vol}...")
        play_square_wave_beep(1000, 200, volume=vol)
        time.sleep(0.3)
    
    # Test 5: Different frequencies
    print("\n5. Testing different frequencies...")
    for freq in [400, 600, 800, 1000, 1200]:
        print(f"   Frequency {freq}Hz...")
        play_square_wave_beep(freq, 300, volume=0.3)
        time.sleep(0.2)
    
    print("\n✅ Beep testing complete!")

if __name__ == "__main__":
    test_beeps() 