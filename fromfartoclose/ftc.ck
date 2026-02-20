// FAR TO CLOSE MOVEMENT

// sound source
SawOsc osc => LPF tone => Gain dry => Pan2 pan => dac;

// reverb path
dry => JCRev rev => dac;

// initial settings (far away)
0.1 => dry.gain;        // quiet
0.8 => rev.mix;         // lots of reverb
800 => tone.freq;       // darker
-0.5 => pan.pan;        // slightly off-center

220 => osc.freq;
0.6 => osc.gain;

// movement duration
10::second => dur moveTime;
now + moveTime => time end;

// gradual movement
while(now < end)
{
    // progress from 0 to 1
    (now - (end - moveTime)) / moveTime => float progress;

    // increase volume
    0.1 + (progress * 0.6) => dry.gain;

    // reduce reverb
    0.8 - (progress * 0.7) => rev.mix;

    // brighten sound
    800 + (progress * 4200) => tone.freq;

    // move toward center
    -0.5 + (progress * 0.5) => pan.pan;

    10::ms => now;
}

// hold close position
while(true)
{
    10::ms => now;
}