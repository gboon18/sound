// ------------------------------------------------------------
// SPATIAL PLUCKY PULSE SYSTEM - FIXED BEAT & SMOOTH APPROACH
// ------------------------------------------------------------

// ---------- MASTER OUTPUT ----------
Gain master => Dyno limiter => JCRev rev => dac;
0.8 => master.gain;
0.25 => rev.mix;

// ---------- UGens ----------
SawOsc pulse;
ADSR pulseEnv;
Pan2 pulsePan;

SinOsc sub;
ADSR subEnv;
Pan2 subPan;

// ---------- INITIALIZE UGens ----------
60 => pulse.freq;
0.4 => pulse.gain;
pulse => pulsePan => pulseEnv => master;
pulseEnv.set(5::ms, 40::ms, 0.0, 80::ms);

40 => sub.freq;
0.25 => sub.gain;
sub => subPan => subEnv => master;
subEnv.set(10::ms, 60::ms, 0.0, 120::ms);

// ---------- FUNCTIONS ----------

// Smooth approach progress 0 → 1 with easing
fun float smoothApproach(time startTime, dur approachDur)
{
    (now - startTime) / approachDur => float p;
    if(p > 1.0) 1.0 => p;
    // ease-in-out (sinusoidal)
    Math.sin(p * Math.PI/2) => float easedP;
    return easedP;
}

// Azimuth rotation pan value -1 → 1
fun float azimuthPan(time startTime, dur rotDur, float phaseOffset)
{
    ((now - startTime) % rotDur) / rotDur => float r;
    Math.sin(r * 2.0 * Math.PI + phaseOffset) => float panValue;
    return panValue;
}

// ---------- TIMERS ----------
12::second => dur approachTime;    // approach duration
3::second  => dur rotationTime;    // rotation period
500::ms    => dur beatInterval;    // heartbeat interval

now => time startApproach;
now => time startRotation;
now => time lastBeat;

// ---------- MAIN LOOP: APPROACH + ROTATION + HEARTBEAT ----------
while(now < startApproach + approachTime)
{
    // -------- APPROACH PROGRESS --------
    smoothApproach(startApproach, approachTime) => float p;

    // -------- DISTANCE / GAIN SIMULATION --------
    0.2 + p * 0.6 => pulse.gain;    // pulse fades in
    0.1 + p * 0.3 => sub.gain;      // sub fades in
    60 + p * 200 => pulse.freq;      // separate "approach pitch" layer
    40 + p * 50 => sub.freq;

    // -------- ROTATION --------
    azimuthPan(startRotation, rotationTime, 0.0) => pulsePan.pan;
    azimuthPan(startRotation, rotationTime, Math.PI/2) => subPan.pan;

    // -------- HEARTBEAT / PULSE --------
    if(now - lastBeat >= beatInterval)
    {
        pulseEnv.keyOn();
        subEnv.keyOn();
        80::ms => now;        // envelope duration
        pulseEnv.keyOff();
        subEnv.keyOff();
        now => lastBeat;      // heartbeat interval is now independent
    }

    5::ms => now;               // small step for smooth rotation
}

// ---------- FINAL CLOSE INTENSE ----------
0.05 => rev.mix;
while(true)
{
    pulseEnv.keyOn();
    subEnv.keyOn();
    80::ms => now;
    pulseEnv.keyOff();
    subEnv.keyOff();
    400::ms => now;
}