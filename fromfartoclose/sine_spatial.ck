// MASTER OUTPUT
Gain master => dac;
0.5 => master.gain;

// BASE SINE + HARMONICS
SinOsc s1 => Pan2 pan1 => master;  // 440 Hz
SinOsc s2 => Pan2 pan2 => master;  // 880 Hz
SinOsc s3 => Pan2 pan3 => master;  // 1320 Hz
SinOsc s4 => Pan2 pan4 => master;  // 1760 Hz

440 => s1.freq;
880 => s2.freq;
1320 => s3.freq;
1760 => s4.freq;

0.3 => s1.gain;
0.15 => s2.gain;
0.075 => s3.gain;
0.0375 => s4.gain;

// ROTATION PARAMETERS
2.0 * Math.PI => float fullCircle;
3::second => dur rotationTime;  // one full rotation
now => time startRotation;

// MAIN LOOP: smooth left/right rotation
while(true)
{
    ((now - startRotation) % rotationTime) / rotationTime => float r;

    // Azimuth rotation
    Math.sin(r * fullCircle) => pan1.pan;
    Math.sin(r * fullCircle) => pan2.pan;
    Math.sin(r * fullCircle) => pan3.pan;
    Math.sin(r * fullCircle) => pan4.pan;

    5::ms => now;
}