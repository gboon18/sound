// drum.ck
// Drum machine: Kick, Snare, Hi-Hat (closed + open)
// OSC port 9001
//
// Run standalone:      chuck drum.ck
// Run with synth:      chuck gm.ck drum.ck
//
// GUI control:         python gm_gui.py   (handles both ports 9000 + 9001)

// ──────────────────────── KickDrum ────────────────────────────────────────────
// Sine oscillator + pitch sweep (freqStart → freqEnd) + short noise click.
// LFO samples on each hit to vary the sweep start pitch.

public class KickDrum extends Chugraph
{
    SinOsc  osc => ADSR ampEnv => Gain drv => Dyno lim => outlet;
    Noise   clk => ADSR clkEnv => drv;

    // LFO runs continuously; sampled at hit() time for per-hit pitch variation
    SinOsc lfo => blackhole;

    150.0  => float freqStart;   // nominal pitch sweep start (Hz)
    40.0   => float freqEnd;     // pitch sweep end (Hz)
    55::ms => dur   pitchTime;   // sweep duration
    0      => int   _sweeping;
    0.0    => float lfoAmt;      // LFO depth in Hz (0 = off)
    150.0  => float _hitFreq;    // actual start freq used by _sweep()

    fun void init()
    {
        1::ms   => ampEnv.attackTime;
        320::ms => ampEnv.decayTime;
        0.0     => ampEnv.sustainLevel;
        60::ms  => ampEnv.releaseTime;

        1::ms  => clkEnv.attackTime;
        12::ms => clkEnv.decayTime;
        0.0    => clkEnv.sustainLevel;
        5::ms  => clkEnv.releaseTime;
        0.12   => clk.gain;

        lim.limit();
        0::ms  => lim.attackTime;
        20::ms => lim.releaseTime;
        0.85   => lim.thresh;
        0.0    => lim.slopeAbove;

        freqStart => osc.freq;
        1.0 => drv.gain;

        // LFO default: slow, off
        0.3 => lfo.freq;
    }

    fun void hit()
    {
        // sample LFO for this hit's pitch offset
        freqStart + lfoAmt * lfo.last() => _hitFreq;
        if (_hitFreq < 20.0) 20.0 => _hitFreq;

        1 => ampEnv.keyOn;
        1 => clkEnv.keyOn;
        if (!_sweeping) spork ~ _sweep();
    }

    fun void _sweep()
    {
        1 => _sweeping;
        _hitFreq => osc.freq;
        50 => int N;
        pitchTime / N => dur dt;
        (_hitFreq - freqEnd) / N => float df;
        for (0 => int i; i < N; i++)
        {
            osc.freq() - df => osc.freq;
            dt => now;
        }
        freqEnd => osc.freq;
        0 => _sweeping;
    }

    fun void setDecay(float t)      { t::ms => ampEnv.decayTime; }
    fun void setTune(float hz)      { hz => freqStart; }
    fun void setDrive(float x)      { 0.8 + 5.2 * (x / 100.0) => drv.gain; }
    fun void setPitchDecay(float t) { t::ms => pitchTime; }
    fun void setLfoRate(float hz)   { (hz < 0.01 ? 0.01 : hz) => lfo.freq; }
    fun void setLfoAmt(float hz)    { hz => lfoAmt; }
}


// ──────────────────────── SnareDrum ───────────────────────────────────────────
// Tuned sine (body) + bandpass-filtered noise (rattle).
// LFO varies the body tone pitch on each hit.

public class SnareDrum extends Chugraph
{
    SinOsc tone  => ADSR tEnv => Gain mix => Dyno lim => outlet;
    Noise  body  => BPF  bpf  => ADSR nEnv => mix;

    SinOsc lfo => blackhole;

    200.0 => float baseTone;   // nominal tone frequency (Hz)
    0.0   => float lfoAmt;     // LFO depth in Hz

    fun void init()
    {
        1::ms   => tEnv.attackTime;
        90::ms  => tEnv.decayTime;
        0.0     => tEnv.sustainLevel;
        30::ms  => tEnv.releaseTime;
        baseTone => tone.freq;
        0.35    => tone.gain;

        1::ms   => nEnv.attackTime;
        110::ms => nEnv.decayTime;
        0.0     => nEnv.sustainLevel;
        40::ms  => nEnv.releaseTime;
        3500.0  => bpf.freq;
        0.8     => bpf.Q;
        0.70    => body.gain;

        lim.limit();
        0::ms  => lim.attackTime;
        20::ms => lim.releaseTime;
        0.85   => lim.thresh;
        0.0    => lim.slopeAbove;

        0.3 => lfo.freq;
    }

    fun void hit()
    {
        // sample LFO for per-hit tone pitch variation
        baseTone + lfoAmt * lfo.last() => float f;
        if (f < 20.0) 20.0 => f;
        f => tone.freq;

        1 => tEnv.keyOn;
        1 => nEnv.keyOn;
    }

    fun void setDecay(float t)
    {
        t::ms         => tEnv.decayTime;
        (t * 1.1)::ms => nEnv.decayTime;
    }

    fun void setTune(float hz)    { hz => baseTone; }
    fun void setSnap(float x)     { x / 100.0 => body.gain; 1.0 - x / 200.0 => tone.gain; }
    fun void setLfoRate(float hz) { (hz < 0.01 ? 0.01 : hz) => lfo.freq; }
    fun void setLfoAmt(float hz)  { hz => lfoAmt; }
}


// ──────────────────────── HiHat ───────────────────────────────────────────────
// White noise through HPF. Closed = short decay, open = long decay.
// LFO varies the HPF cutoff on each hit for tonal shimmer.

public class HiHat extends Chugraph
{
    Noise nsrc => HPF hpf => ADSR env => Dyno lim => outlet;

    SinOsc lfo => blackhole;

    35::ms  => dur   closedDec;
    260::ms => dur   openDec;
    7800.0  => float baseHpf;   // nominal HPF cutoff (Hz)
    0.0     => float lfoAmt;    // LFO depth in Hz

    fun void init()
    {
        1::ms      => env.attackTime;
        closedDec  => env.decayTime;
        0.0        => env.sustainLevel;
        8::ms      => env.releaseTime;

        baseHpf => hpf.freq;
        0.7     => hpf.Q;

        lim.limit();
        0::ms  => lim.attackTime;
        10::ms => lim.releaseTime;
        0.85   => lim.thresh;
        0.0    => lim.slopeAbove;

        0.3 => lfo.freq;
    }

    // open == 0: closed,  open == 1: open
    fun void hit(int open)
    {
        // sample LFO for per-hit HPF cutoff variation
        baseHpf + lfoAmt * lfo.last() => float f;
        if (f < 200.0)  200.0  => f;
        if (f > 20000.0) 20000.0 => f;
        f => hpf.freq;

        if (open) openDec   => env.decayTime;
        else      closedDec => env.decayTime;
        1 => env.keyOn;
    }

    fun void setClosedDecay(float t) { t::ms => closedDec; }
    fun void setOpenDecay(float t)   { t::ms => openDec; }
    fun void setLfoRate(float hz)    { (hz < 0.01 ? 0.01 : hz) => lfo.freq; }
    fun void setLfoAmt(float hz)     { hz => lfoAmt; }
}


// ──────────────────────── Instantiate + wire ─────────────────────────────────
// Each voice has its own per-voice hard limiter (thresh 0.85).
// Master Dyno limiter catches residual inter-voice summing peaks.

KickDrum  kick  => Gain master => Dyno masterLim => dac;
SnareDrum snare => master;
HiHat     hat   => master;

kick.init();
snare.init();
hat.init();

0.55 => master.gain;   // headroom before master limiter

masterLim.limit();
0::ms  => masterLim.attackTime;
30::ms => masterLim.releaseTime;
0.92   => masterLim.thresh;
0.0    => masterLim.slopeAbove;


// ──────────────────────── Default 16-step patterns ───────────────────────────

[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0] @=> int kickPat[];
[0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0] @=> int snarePat[];
[1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0] @=> int chatPat[];
[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0] @=> int ohatPat[];


// ──────────────────────── Sequencer state ────────────────────────────────────

120.0 => float bpm;
0     => int   seqRunning;
-1    => int   curStep;


// ──────────────────────── Sequencer loop ─────────────────────────────────────

fun void seqLoop()
{
    0 => int step;
    while (true)
    {
        if (seqRunning)
        {
            step => curStep;
            if (kickPat[step])  kick.hit();
            if (snarePat[step]) snare.hit();
            if (chatPat[step])  hat.hit(0);
            if (ohatPat[step])  hat.hit(1);

            (60.0 / bpm / 4.0) => float stepSec;
            stepSec::second => now;
            (step + 1) % 16 => step;
        }
        else
        {
            -1 => curStep;
            10::ms => now;
        }
    }
}
spork ~ seqLoop();


// ──────────────────────── OSC listener (port 9001) ───────────────────────────

fun void oscLoop()
{
    OscIn  oin;
    OscMsg omsg;
    9001 => oin.port;

    oin.addAddress("/drum/bpm, f");
    oin.addAddress("/drum/play, i");
    oin.addAddress("/drum/vol, f");

    oin.addAddress("/drum/step/kick, i i");
    oin.addAddress("/drum/step/snare, i i");
    oin.addAddress("/drum/step/chat, i i");
    oin.addAddress("/drum/step/ohat, i i");

    oin.addAddress("/drum/kick/decay, f");
    oin.addAddress("/drum/kick/tune, f");
    oin.addAddress("/drum/kick/drive, f");
    oin.addAddress("/drum/kick/pitchdecay, f");
    oin.addAddress("/drum/kick/lfo/rate, f");
    oin.addAddress("/drum/kick/lfo/amt, f");

    oin.addAddress("/drum/snare/decay, f");
    oin.addAddress("/drum/snare/tune, f");
    oin.addAddress("/drum/snare/snap, f");
    oin.addAddress("/drum/snare/lfo/rate, f");
    oin.addAddress("/drum/snare/lfo/amt, f");

    oin.addAddress("/drum/hat/decay/c, f");
    oin.addAddress("/drum/hat/decay/o, f");
    oin.addAddress("/drum/hat/lfo/rate, f");
    oin.addAddress("/drum/hat/lfo/amt, f");

    while (true)
    {
        oin => now;
        while (oin.recv(omsg))
        {
            if (omsg.address == "/drum/bpm")  omsg.getFloat(0) => bpm;
            if (omsg.address == "/drum/play") omsg.getInt(0)   => seqRunning;
            if (omsg.address == "/drum/vol")  omsg.getFloat(0) => master.gain;

            if (omsg.address == "/drum/step/kick")  { omsg.getInt(0) => int s; omsg.getInt(1) => kickPat[s]; }
            if (omsg.address == "/drum/step/snare") { omsg.getInt(0) => int s; omsg.getInt(1) => snarePat[s]; }
            if (omsg.address == "/drum/step/chat")  { omsg.getInt(0) => int s; omsg.getInt(1) => chatPat[s]; }
            if (omsg.address == "/drum/step/ohat")  { omsg.getInt(0) => int s; omsg.getInt(1) => ohatPat[s]; }

            if (omsg.address == "/drum/kick/decay")      kick.setDecay(omsg.getFloat(0));
            if (omsg.address == "/drum/kick/tune")       kick.setTune(omsg.getFloat(0));
            if (omsg.address == "/drum/kick/drive")      kick.setDrive(omsg.getFloat(0));
            if (omsg.address == "/drum/kick/pitchdecay") kick.setPitchDecay(omsg.getFloat(0));
            if (omsg.address == "/drum/kick/lfo/rate")   kick.setLfoRate(omsg.getFloat(0));
            if (omsg.address == "/drum/kick/lfo/amt")    kick.setLfoAmt(omsg.getFloat(0));

            if (omsg.address == "/drum/snare/decay")     snare.setDecay(omsg.getFloat(0));
            if (omsg.address == "/drum/snare/tune")      snare.setTune(omsg.getFloat(0));
            if (omsg.address == "/drum/snare/snap")      snare.setSnap(omsg.getFloat(0));
            if (omsg.address == "/drum/snare/lfo/rate")  snare.setLfoRate(omsg.getFloat(0));
            if (omsg.address == "/drum/snare/lfo/amt")   snare.setLfoAmt(omsg.getFloat(0));

            if (omsg.address == "/drum/hat/decay/c")     hat.setClosedDecay(omsg.getFloat(0));
            if (omsg.address == "/drum/hat/decay/o")     hat.setOpenDecay(omsg.getFloat(0));
            if (omsg.address == "/drum/hat/lfo/rate")    hat.setLfoRate(omsg.getFloat(0));
            if (omsg.address == "/drum/hat/lfo/amt")     hat.setLfoAmt(omsg.getFloat(0));
        }
    }
}
spork ~ oscLoop();


while (true) { 1::second => now; }
