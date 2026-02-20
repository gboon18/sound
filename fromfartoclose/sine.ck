// MASTER OUTPUT
Gain master => dac;
0.5 => master.gain;

// SINE OSCILLATOR
SinOsc tone => Pan2 pan => master;
440 => tone.freq;

// ROTATION PARAMETERS
2.0 * Math.PI => float fullCircle;
3::second => dur rotationTime; // faster, more noticeable

while(true)
{
    now => time start;
    while(now < start + rotationTime)
    {
        (now - start) / rotationTime => float p;

        // Full left-right swing
        Math.sin(p * fullCircle) => pan.pan;

        // Slight frequency modulation
        440 + Math.sin(p * fullCircle) * 10 => tone.freq;

        // Slight volume modulation for distance effect
        0.3 + 0.1 * Math.cos(p * fullCircle) => tone.gain;

        5::ms => now;
    }
}