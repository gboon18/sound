// ----------------------------
// Indefinitely Running Smooth Piano Chords
// ----------------------------

// ----------------------------
// Tempo & Beat
// ----------------------------
100.0 => float BPM;
100.0 / BPM => float beatDur;

// ----------------------------
// Piano Oscillators Setup
// ----------------------------
4 => int numKeys;
SinOsc pianoOscs[numKeys];
Gain pianoGains[numKeys];
0.15 => float amp; // audible amplitude

for (0 => int i; i < numKeys; i++)
{
    pianoOscs[i] => pianoGains[i] => dac;
    0.0 => pianoGains[i].gain; // start muted
}

// ----------------------------
// Chord Frequencies
// ----------------------------
[261.63, 329.63, 392.00, 523.25] @=> float chord1[];
[349.23, 440.00, 523.25, 659.25] @=> float chord2[];
[392.00, 493.88, 587.33, 698.46] @=> float chord3[];
[440.00, 554.37, 659.25, 783.99] @=> float chord4[];

// ----------------------------
// Detune Offsets for chord3 and chord4
// ----------------------------
[-100.0, -30.0, 0.0, 30.0] @=> float chordDetuneOffsets[];

// ----------------------------
// Empty array for non-detune chords
// ----------------------------
float chordRest[4];
0.0 => chordRest[0];
0.0 => chordRest[1];
0.0 => chordRest[2];
0.0 => chordRest[3];

// ----------------------------
// Detune parameters
// ----------------------------
0.2 => float detuneDelay;      // seconds after chord to start glide
0.5 => float detuneGlideDur;   // glide duration

// ----------------------------
// Smooth glide function
// ----------------------------
fun void glideTo(float startFreq, float targetFreq, float dur, SinOsc osc)
{
    20 => int steps;
    dur / steps => float stepTime;
    (targetFreq - startFreq) / steps => float stepSize;

    for (0 => int i; i < steps; i++)
    {
        (startFreq + stepSize * (i+1)) => float newFreq;
        osc.freq(newFreq);
        (stepTime::second) => now;
    }
}

// ----------------------------
// Simple fade-in/out function
// ----------------------------
fun void fadeInOut(Gain g, float peak, float sustainSec)
{
    0.0 => g.gain;
    0.05::second => dur step;

    // fade in
    for (0 => int i; i < 10; i++)
    {
        g.gain() + (peak/10.0) => g.gain;
        step => now;
    }

    // sustain
    (sustainSec::second) => now;

    // fade out
    for (0 => int i; i < 10; i++)
    {
        g.gain() - (peak/10.0) => g.gain;
        step => now;
    }

    0.0 => g.gain;
}

// ----------------------------
// Play a chord
// ----------------------------
fun void playChord(float freqs[], float detuneOffsets[])
{
    // set initial frequencies
    for (0 => int i; i < numKeys; i++)
        pianoOscs[i].freq(freqs[i]);

    // start fade-in for all keys
    for (0 => int i; i < numKeys; i++)
        spork ~ fadeInOut(pianoGains[i], amp, beatDur);

    // detune if needed
    if (detuneOffsets[0] != 0 || detuneOffsets[1] != 0 || detuneOffsets[2] != 0 || detuneOffsets[3] != 0)
    {
        // wait for detune delay
        (detuneDelay::second) => now;

        for (0 => int i; i < numKeys; i++)
        {
            (freqs[i] + detuneOffsets[i]) => float targetFreq;
            freqs[i] => float startFreq;
            spork ~ glideTo(startFreq, targetFreq, detuneGlideDur, pianoOscs[i]);
        }
    }
}

// ----------------------------
// Main Loop: run indefinitely
// ----------------------------
while (true)
{
    playChord(chord1, chordRest);
    (beatDur::second) => now;

    playChord(chord2, chordRest);
    (beatDur::second) => now;

    playChord(chord3, chordDetuneOffsets);
    (beatDur::second) => now;

    playChord(chord4, chordDetuneOffsets);
    (beatDur::second) => now;
}