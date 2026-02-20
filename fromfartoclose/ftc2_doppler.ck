// ------------------------------------------------------------
// CINEMATIC APPROACHING BEAT WITH SMOOTH ROTATION + DOPPLER
// ------------------------------------------------------------

// ---------- MASTER OUTPUT ----------
Gain mix => Dyno limiter => JCRev rev => dac;

0.9 => mix.gain;
0.9 => limiter.thresh;
0.5 => limiter.ratio;
0.25 => rev.mix;

// ---------- MAIN PULSE ----------
SawOsc pulse => Pan2 pulsePan => LPF pulseTone => ADSR pulseEnv => mix;

60 => float basePulseFreq;
600 => pulseTone.freq;
0.4 => pulse.gain;
pulseEnv.set(10::ms, 60::ms, 0.0, 120::ms);

// ---------- SUB ----------
SinOsc sub => Pan2 subPan => ADSR subEnv => mix;

40 => float baseSubFreq;
0.3 => sub.gain;
subEnv.set(10::ms, 80::ms, 0.0, 150::ms);

// ---------- TIMERS ----------
24::second => dur approachTime;       // distance approach
3::second  => dur rotationTime;       // full rotation duration
900::ms    => dur beatInterval;       // pulse heartbeat

now => time startApproach;
now => time startRotation;
now => time lastBeat;

0.02 => float dopplerAmount;          // ±2% Doppler effect
2.0 * Math.PI => float fullCircle;

// ---------- MAIN LOOP ----------
while(now < startApproach + approachTime)
{
    // -------- APPROACH PROGRESS 0 → 1 --------
    (now - startApproach) / approachTime => float p;

    // -------- ROTATION PROGRESS (continuous) --------
    ((now - startRotation) % rotationTime) / rotationTime => float r;

    // -------- DISTANCE SIMULATION --------
    600 + (p * 3500) => pulseTone.freq;    // main pulse moves up as it approaches
    0.25 - (p * 0.20) => rev.mix;         // reverb decreases

    // -------- AZIMUTHAL ROTATION --------
    Math.sin(r * fullCircle) => pulsePan.pan;
    Math.cos(r * fullCircle) => subPan.pan;

    // -------- DOPPLER EFFECT --------
    Math.cos(r * fullCircle) * dopplerAmount => float shift;
    basePulseFreq * (1 + shift) => pulse.freq;
    baseSubFreq * (1 + shift) => sub.freq;

    // -------- HEARTBEAT / PULSE --------
    if(now >= lastBeat)
    {
        pulseEnv.keyOn();
        subEnv.keyOn();

        140::ms => now;      // envelope body
        pulseEnv.keyOff();
        subEnv.keyOff();

        lastBeat + beatInterval => lastBeat;  // keeps heartbeat period constant
    }

    5::ms => now;  // small time step for smooth rotation
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