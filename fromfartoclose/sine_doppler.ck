// MASTER OUTPUT
Gain master => dac;
0.5 => master.gain;

// SINE OSCILLATOR
SinOsc tone => Pan2 pan => master;
440 => tone.freq;   // base frequency
0.3 => tone.gain;

// ROTATION PARAMETERS
2.0 * Math.PI => float fullCircle;
3::second => dur rotationTime; // fast rotation
440 => float f0;                // base frequency
0.02 => float dopplerAmount;    // max pitch shift ±2%

while(true)
{
    now => time start;
    while(now < start + rotationTime)
    {
        (now - start) / rotationTime => float p;

        // ---- AZIMUTH ROTATION ----
        Math.sin(p * fullCircle) => pan.pan;

        // ---- DOPPLER EFFECT ----
        // approximate radial velocity as derivative of pan sine
        Math.cos(p * fullCircle) * dopplerAmount => float shift;
        f0 * (1 + shift) => tone.freq;

        5::ms => now;
    }
}