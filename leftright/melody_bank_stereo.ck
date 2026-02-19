// ------------------------------------------------------------
// melody_bank_stereo.ck
// Left channel: major harmony/melody
// Right channel: minor harmony/melody
// Independent filter + reverb per channel
// Real-time control via OSC + (optional) keyboard
// ------------------------------------------------------------

// ------------------------- GLOBALS --------------------------
120.0 => float gBPM;
0 => int gBank;          // 0 = pop, 1 = jazz
0 => int gPaused;        // 0 running, 1 paused
48 => int gBaseMidi;     // C3 = 48 (sets overall register)
Event gChordTick;        // conductor -> players sync each chord

// current chord root offset in semitones from C (0..11), updated by conductor
0 => int gChordRoot;

// for jitter/humanization
2.0 => float gHumanMs;

// ------------------------- SCALES ---------------------------
[0,2,4,5,7,9,11] @=> int C_MAJOR_SCALE[];
[0,2,3,5,7,8,10] @=> int C_NAT_MINOR_SCALE[];

// ------------------------- CHORD UTILS ----------------------
// quality: 0 = maj, 1 = min, 2 = dim
fun void buildTriad(int rootMidi, int quality, int notes[])
{
    rootMidi => notes[0];
    if(quality == 0) { rootMidi + 4 => notes[1]; rootMidi + 7 => notes[2]; }
    else if(quality == 1) { rootMidi + 3 => notes[1]; rootMidi + 7 => notes[2]; }
    else { rootMidi + 3 => notes[1]; rootMidi + 6 => notes[2]; }
}

fun int pickFromArray(int a[])
{
    return a[Math.random2(0, a.size()-1)];
}

fun int clampMidi(int m)
{
    if(m < 0) return 0;
    if(m > 127) return 127;
    return m;
}

// choose mostly chord tones, sometimes scale tones
// biasChord: 0.0..1.0
fun int choosePitchMajor(int rootMidi, float biasChord)
{
    int triad[3];
    buildTriad(rootMidi, 0, triad);

    if(Math.randomf() < biasChord)
    {
        return triad[Math.random2(0,2)];
    }
    else
    {
        // pick a scale degree then snap near root's octave region
        pickFromArray(C_MAJOR_SCALE) => int deg;
        // center around rootMidi's octave
        (rootMidi/12)*12 + deg => int cand;
        // sometimes jump octave
        if(Math.randomf() < 0.25) { 12 * Math.random2(-1, 1) +=> cand; }
        return cand;
    }
}

fun int choosePitchMinor(int rootMidi, float biasChord)
{
    int triad[3];
    buildTriad(rootMidi, 1, triad);

    if(Math.randomf() < biasChord)
    {
        return triad[Math.random2(0,2)];
    }
    else
    {
        pickFromArray(C_NAT_MINOR_SCALE) => int deg;
        (rootMidi/12)*12 + deg => int cand;
        if(Math.randomf() < 0.25) { 12 * Math.random2(-1, 1) +=> cand; }
        return cand;
    }
}

// rhythm: choose from a small set based on BPM
fun dur chooseStepDur()
{
    // base 1/8 note
    (60.0/gBPM)::second => dur q; // quarter note
    q/2 => dur e;                 // eighth note
    q/4 => dur s;                 // sixteenth note

    Math.randomf() => float r;
    if(r < 0.60) return e;
    else if(r < 0.90) return s;
    else return q; // occasional longer note
}

// humanize timing slightly
fun dur humanize(dur d)
{
    (Math.random2f(-gHumanMs, gHumanMs))::ms => dur j;
    return d + j;
}

// --------------------- PROGRESSION BANK ---------------------
// Each bank is a sequence of chord roots (as semitone offsets from C).
// Players decide major/minor quality per channel; conductor provides root.
[0, 7, 9, 5] @=> int BANK_POP[];        // I - V - vi - IV  (C-G-A-F)
[2, 7, 0, 0] @=> int BANK_JAZZ[];       // ii - V - I - I   (D-G-C-C)

// chord length in beats
4 => int gChordBeats;

// ------------------------ CONDUCTOR -------------------------
fun void conductor()
{
    0 => int i;

    while(true)
    {
        if(gPaused)
        {
            50::ms => now;
            continue;
        }

        // pick current bank chord root
        if(gBank == 0)
        {
            BANK_POP[i % BANK_POP.size()] => gChordRoot;
            i++;
        }
        else
        {
            BANK_JAZZ[i % BANK_JAZZ.size()] => gChordRoot;
            i++;
        }

        // notify players "new chord"
        gChordTick.broadcast();

        // wait chord duration
        (60.0/gBPM)::second => dur q;
        gChordBeats * q => dur chordDur;
        chordDur => now;
    }
}

// ---------------------- LEFT CHANNEL VOICE ------------------
fun void leftPlayer()
{
    // signal chain (left only)
    TriOsc osc => ADSR env => LPF filt => JCRev rev => Gain g => dac.left;

    0.20 => g.gain;
    2000.0 => filt.freq;
    1.0 => filt.Q;
    0.10 => rev.mix;

    // envelope
    5::ms => dur A;
    30::ms => dur D;
    0.6 => float S;
    60::ms => dur R;
    env.set(A, D, S, R);

    // timbre
    0.6 => osc.width;
    0.15 => osc.gain;

    while(true)
    {
        gChordTick => now;

        // play notes until next chord tick arrives
        now => time t0;

        (60.0/gBPM)::second => dur q;
        gChordBeats * q => dur chordDur;
        t0 + chordDur => time tEnd;

        // root midi for this chord in a mid register
        clampMidi(gBaseMidi + gChordRoot) => int rootMidi;

        while(now < tEnd)
        {
            if(gPaused) { 20::ms => now; continue; }

            chooseStepDur() => dur step;
            choosePitchMajor(rootMidi, 0.75) => int m;

            Std.mtof(m) => osc.freq;
            env.keyOn();
            (step * 0.85) => now;
            env.keyOff();
            humanize(step * 0.15) => now;
        }
    }
}

// --------------------- RIGHT CHANNEL VOICE ------------------
fun void rightPlayer()
{
    // signal chain (right only) - different "space/distance"
    SawOsc osc => ADSR env => BPF filt => NRev rev => Gain g => dac.right;

    0.20 => g.gain;
    900.0 => filt.freq;
    3.0 => filt.Q;
    0.25 => rev.mix;

    10::ms => dur A;
    60::ms => dur D;
    0.5 => float S;
    120::ms => dur R;
    env.set(A, D, S, R);

    0.12 => osc.gain;

    while(true)
    {
        gChordTick => now;

        now => time t0;
        (60.0/gBPM)::second => dur q;
        gChordBeats * q => dur chordDur;
        t0 + chordDur => time tEnd;

        // same root, but interpreted as minor channel
        clampMidi(gBaseMidi + gChordRoot) => int rootMidi;

        while(now < tEnd)
        {
            if(gPaused) { 20::ms => now; continue; }

            chooseStepDur() => dur step;
            choosePitchMinor(rootMidi, 0.70) => int m;

            Std.mtof(m) => osc.freq;
            env.keyOn();
            (step * 0.80) => now;
            env.keyOff();
            humanize(step * 0.20) => now;
        }
    }
}

// -------------------------- OSC CONTROL ---------------------
// Send OSC to port 9000 from any controller (Max, TouchOSC, Python, etc.)
// Addresses (float unless noted):
//  /tempo            (BPM)
//  /bank             (int: 0 pop, 1 jazz)
//  /pause            (int: 0/1)
//  /chordBeats       (int)
//  /left/gain        (0..1)
//  /left/rev         (0..1)
//  /left/cutoff      (Hz)
//  /left/q           (Q)
//  /right/gain       (0..1)
//  /right/rev        (0..1)
//  /right/cutoff     (Hz)
//  /right/q          (Q)
//
// NOTE: We keep references to UGens by storing them in global holders.
// To keep this single-file, we use global "control variables" and apply
// them in small control loops below.

0.20 => float L_gain;
0.10 => float L_rev;
2000.0 => float L_cutoff;
1.0 => float L_q;

0.20 => float R_gain;
0.25 => float R_rev;
900.0 => float R_cutoff;
3.0 => float R_q;

// OSC receiver
OscRecv recv;
9000 => recv.port;
recv.listen();

// register OSC events
recv.event("/tempo, f") @=> OscEvent eTempo;
recv.event("/bank, i") @=> OscEvent eBank;
recv.event("/pause, i") @=> OscEvent ePause;
recv.event("/chordBeats, i") @=> OscEvent eChordBeats;

recv.event("/left/gain, f") @=> OscEvent eLGain;
recv.event("/left/rev, f") @=> OscEvent eLRev;
recv.event("/left/cutoff, f") @=> OscEvent eLCut;
recv.event("/left/q, f") @=> OscEvent eLQ;

recv.event("/right/gain, f") @=> OscEvent eRGain;
recv.event("/right/rev, f") @=> OscEvent eRRev;
recv.event("/right/cutoff, f") @=> OscEvent eRCut;
recv.event("/right/q, f") @=> OscEvent eRQ;

// control variable clamp
fun float clamp01(float x)
{
    if(x < 0.0) return 0.0;
    if(x > 1.0) return 1.0;
    return x;
}

fun void oscControl()
{
    while(true)
    {
        // wait for any OSC message (poll-style with short sleep)
        10::ms => now;

        while(eTempo.nextMsg() != 0)
        {
            eTempo.getFloat() => float bpm;
            if(bpm >= 20.0 && bpm <= 300.0) bpm => gBPM;
        }

        while(eBank.nextMsg() != 0)
        {
            eBank.getInt() => int b;
            if(b == 0 || b == 1) b => gBank;
        }

        while(ePause.nextMsg() != 0)
        {
            ePause.getInt() => int p;
            (p != 0) => gPaused;
        }

        while(eChordBeats.nextMsg() != 0)
        {
            eChordBeats.getInt() => int cb;
            if(cb >= 1 && cb <= 16) cb => gChordBeats;
        }

        while(eLGain.nextMsg() != 0) clamp01(eLGain.getFloat()) => L_gain;
        while(eLRev.nextMsg()  != 0) clamp01(eLRev.getFloat())  => L_rev;
        while(eLCut.nextMsg()  != 0) eLCut.getFloat()           => L_cutoff;
        while(eLQ.nextMsg()    != 0) eLQ.getFloat()             => L_q;

        while(eRGain.nextMsg() != 0) clamp01(eRGain.getFloat()) => R_gain;
        while(eRRev.nextMsg()  != 0) clamp01(eRRev.getFloat())  => R_rev;
        while(eRCut.nextMsg()  != 0) eRCut.getFloat()           => R_cutoff;
        while(eRQ.nextMsg()    != 0) eRQ.getFloat()             => R_q;
    }
}

// Apply control variables to the actual UGens by "tapping" them in each player.
// To keep it simple, we run separate appliers that search the shred tree is not possible,
// so we implement "proxy" control: each channel has a hidden bus and we control it here.

Gain Lbus => dac.left;
Gain Rbus => dac.right;

// Rebuild players to route into buses (so we can control gain globally)
fun void leftPlayer_bus()
{
    TriOsc osc => ADSR env => LPF filt => JCRev rev => Gain g => Lbus;

    // initial
    1.0 => Lbus.gain;

    5::ms => dur A;
    30::ms => dur D;
    0.6 => float S;
    60::ms => dur R;
    env.set(A, D, S, R);

    0.6 => osc.width;
    0.15 => osc.gain;

    while(true)
    {
        // apply current params continuously
        L_gain => g.gain;
        L_rev => rev.mix;
        L_cutoff => filt.freq;
        L_q => filt.Q;

        gChordTick => now;

        now => time t0;
        (60.0/gBPM)::second => dur q;
        gChordBeats * q => dur chordDur;
        t0 + chordDur => time tEnd;

        clampMidi(gBaseMidi + gChordRoot) => int rootMidi;

        while(now < tEnd)
        {
            // refresh params in-note too
            L_gain => g.gain;
            L_rev => rev.mix;
            L_cutoff => filt.freq;
            L_q => filt.Q;

            if(gPaused) { 20::ms => now; continue; }

            chooseStepDur() => dur step;
            choosePitchMajor(rootMidi, 0.75) => int m;

            Std.mtof(m) => osc.freq;
            env.keyOn();
            (step * 0.85) => now;
            env.keyOff();
            humanize(step * 0.15) => now;
        }
    }
}

fun void rightPlayer_bus()
{
    SawOsc osc => ADSR env => BPF filt => NRev rev => Gain g => Rbus;

    1.0 => Rbus.gain;

    10::ms => dur A;
    60::ms => dur D;
    0.5 => float S;
    120::ms => dur R;
    env.set(A, D, S, R);

    0.12 => osc.gain;

    while(true)
    {
        R_gain => g.gain;
        R_rev => rev.mix;
        R_cutoff => filt.freq;
        R_q => filt.Q;

        gChordTick => now;

        now => time t0;
        (60.0/gBPM)::second => dur q;
        gChordBeats * q => dur chordDur;
        t0 + chordDur => time tEnd;

        clampMidi(gBaseMidi + gChordRoot) => int rootMidi;

        while(now < tEnd)
        {
            R_gain => g.gain;
            R_rev => rev.mix;
            R_cutoff => filt.freq;
            R_q => filt.Q;

            if(gPaused) { 20::ms => now; continue; }

            chooseStepDur() => dur step;
            choosePitchMinor(rootMidi, 0.70) => int m;

            Std.mtof(m) => osc.freq;
            env.keyOn();
            (step * 0.80) => now;
            env.keyOff();
            humanize(step * 0.20) => now;
        }
    }
}

// ----------------------- KEYBOARD CONTROL -------------------
// Space: pause toggle
// 1: pop bank, 2: jazz bank
// +/-: tempo down/up (5 BPM)
// Optional: comment out if you don't want HID.
fun void keyboardControl()
{
    Hid hi;
    HidMsg msg;

    if(!hi.openKeyboard(0))
    {
        // no keyboard available; just idle
        while(true) 1::second => now;
    }

    while(true)
    {
        hi => now;
        while(hi.recv(msg))
        {
            if(msg.isButtonDown())
            {
                if(msg.ascii == 32) // space
                {
                    (gPaused == 0) => gPaused;
                }
                else if(msg.ascii == 49) // '1'
                {
                    0 => gBank;
                }
                else if(msg.ascii == 50) // '2'
                {
                    1 => gBank;
                }
                else if(msg.ascii == 45) // '-'
                {
                    (gBPM - 5.0) => gBPM;
                    if(gBPM < 20.0) 20.0 => gBPM;
                }
                else if(msg.ascii == 61 || msg.ascii == 43) // '=' or '+'
                {
                    (gBPM + 5.0) => gBPM;
                    if(gBPM > 300.0) 300.0 => gBPM;
                }
            }
        }
    }
}

// --------------------------- START --------------------------
spork ~ conductor();
spork ~ oscControl();
spork ~ keyboardControl();

// Use bus-based players (so gain control is clean)
spork ~ leftPlayer_bus();
spork ~ rightPlayer_bus();

// keep alive
while(true) 1::second => now;
