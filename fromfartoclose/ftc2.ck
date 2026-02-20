// ------------------------------------------------------------
// CLEAN APPROACHING BEAT (NO CLICK / NO CLIP)
// ------------------------------------------------------------

// ---------- MASTER WITH LIMITER ----------
Gain mix => Dyno limiter => JCRev rev => dac;

0.9 => mix.gain;          // lower overall gain
0.9 => limiter.thresh;    // soft limiting
0.5 => limiter.ratio;
0.25 => rev.mix;          // start moderate (not extreme)

// ---------- MAIN PULSE ----------
SawOsc pulse => LPF pulseTone => ADSR pulseEnv => mix;

60 => pulse.freq;
600 => pulseTone.freq;
0.4 => pulse.gain;

// slower, smoother envelope
pulseEnv.set(10::ms, 60::ms, 0.0, 120::ms);

// ---------- SUB ----------
SinOsc sub => ADSR subEnv => mix;

40 => sub.freq;
0.3 => sub.gain;
subEnv.set(10::ms, 80::ms, 0.0, 150::ms);

// ---------- WHISPER ----------
// Noise whisper => HPF air => Gain whisperGain => Pan2 whisperPan => mix;

// 6000 => air.freq;
// 0.15 => whisper.gain;
// 0.0 => whisperGain.gain;
// -0.6 => whisperPan.pan;

// ---------- APPROACH MOVEMENT ----------
12::second => dur total;
now + total => time end;

900::ms => dur beatInterval;

while(now < end)
{
    (now - (end - total)) / total => float p;

    // distance simulation
    600 + (p * 3500) => pulseTone.freq;
    0.25 - (p * 0.20) => rev.mix;
    // p * 0.3 => whisperGain.gain;
    // -0.6 + (p * 0.6) => whisperPan.pan;

    // heartbeat speeds up (but never too fast)
    900::ms - (p * 500::ms) => beatInterval;

    // trigger cleanly
    pulseEnv.keyOn();
    subEnv.keyOn();

    140::ms => now;   // allow attack & body

    pulseEnv.keyOff();
    subEnv.keyOff();

    // WAIT LONGER than release to avoid overlap clicks
    beatInterval => now;
}

// ---------- FINAL CLOSE INTENSE ----------
0.05 => rev.mix;

while(true)
{
    pulseEnv.keyOn();
    subEnv.keyOn();
    140::ms => now;
    pulseEnv.keyOff();
    subEnv.keyOff();
    350::ms => now;
}