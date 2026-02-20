// MASTER OUTPUT
Gain master => dac;
0.5 => master.gain;

// BASE SINE
SinOsc tone => Pan2 pan => master;
440 => float f0;
0.3 => tone.gain;

// HARMONICS
SinOsc tone2 => Pan2 pan2 => master;
SinOsc tone3 => Pan2 pan3 => master;
0.025 => tone2.gain;
0.01 => tone3.gain;

// ROTATION PARAMETERS
2.0 * Math.PI => float fullCircle;
3::second => dur rotationTime;
0.02 => float dopplerAmount;

while(true)
{
    now => time start;
    while(now < start + rotationTime)
    {
        (now - start) / rotationTime => float p;

        // rotation
        Math.sin(p * fullCircle) => pan.pan;
        Math.sin(p * fullCircle) => pan2.pan;
        Math.sin(p * fullCircle) => pan3.pan;

        // Doppler shift
        Math.cos(p * fullCircle) * dopplerAmount => float shift;
        f0 * (1 + shift) => tone.freq;
        f0*2 * (1 + shift) => tone2.freq;
        f0*3 * (1 + shift) => tone3.freq;

        5::ms => now;
    }
}