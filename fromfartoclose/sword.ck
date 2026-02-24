// ============================================================
// SwordSwish.ck  (modular Chugraph effect)
// - Accepts any audio input via inlet
// - Outputs processed swish via outlet
// - Two overlapped swings:
//     1) left->right  : [t .. t + swingDur]
//     2) right->left  : [t+offsetDur .. t+offsetDur+swingDur]
//   repeating every swingDur
// ============================================================

// ---------------------------
// Voice helper class (must be global scope in ChucK)
// ---------------------------
class Voice
{
    Gain   inTap;
    Noise  nz;
    Gain   nzG;
    Gain   sum;
    BPF    bpf;
    HPF    hpf;
    ADSR   env;
    Pan2   pan;
    Gain   outG;

    // init chain: (input + noise) -> BPF -> HPF -> ADSR -> Pan2 -> Gain
    fun void init(Gain src, Gain wetBus)
    {
        src => inTap => sum;
        nz => nzG => sum;

        sum => bpf => hpf => env => pan => outG => wetBus;

        // defaults for safety
        0.0 => nzG.gain;
        0.0 => outG.gain;
    }

    fun void setStatic(float q, float hpfHz)
    {
        q => bpf.Q;
        hpfHz => hpf.freq;
    }

    fun void setGains(float inG, float nzG0, float outG0)
    {
        inG => inTap.gain;
        nzG0 => nzG.gain;
        outG0 => outG.gain;
    }

    fun void setEnv(dur a, dur d, float s, dur r)
    {
        env.set(a, d, s, r);
    }
}

// ---------------------------
// Swish module (Chugraph)
// ---------------------------
public class SwordSwish extends Chugraph
{
    // patch points
    inlet => Gain _dryTap;

    // wet mix bus -> reverb -> outlet
    Gain _wetBus => JCRev _rev => Gain _out => outlet;

    // user parameters (human input variables)
    dur   swingDur;        // duration of one swing (e.g., 0.5::second)
    dur   offsetDur;       // offset between opposite swing starts (e.g., 0.25::second)
    dur   ctrlDt;          // control update interval (e.g., 1::ms)

    float panDepth;        // 0..1 (how wide in stereo)
    float inGain;          // dry input sent into swish chain
    float noiseGain;       // extra air/noise for the swish
    float wetGain;         // overall wet output level

    // filter shaping
    float bpfLo;           // sweep start frequency (Hz)
    float bpfHi;           // sweep end frequency (Hz)
    float bpfQ;            // resonance/Q for bandpass
    float hpfCut;          // highpass cutoff (Hz)

    // envelope
    dur   atk;
    dur   dec;
    float sus;
    dur   rel;

    // reverb
    float revMix;          // JCRev mix 0..1

    // voice count (poly) for overlap safety
    int   nVoices;

    // voices
    Voice _v[0];
    int   _rr;

    // ---------------------------------
    // constructor-like init
    // ---------------------------------
    fun void init(int voices)
    {
        voices => nVoices;

        // defaults (can be overridden via setters)
        0.5::second => swingDur;
        0.25::second => offsetDur;
        1::ms => ctrlDt;

        1.0 => panDepth;
        1.0 => inGain;
        0.20 => noiseGain;
        0.90 => wetGain;

        900.0 => bpfLo;
        4500.0 => bpfHi;
        6.0 => bpfQ;
        250.0 => hpfCut;

        5::ms => atk;
        60::ms => dec;
        0.0 => sus;
        120::ms => rel;

        0.08 => revMix;

        // output stages
        wetGain => _wetBus.gain;
        1.0 => _out.gain;
        revMix => _rev.mix;

        // allocate and patch voices
        _v.clear();
        for (0 => int i; i < nVoices; i++)
        {
            Voice vv;
            vv.init(_dryTap, _wetBus);
            vv.setStatic(bpfQ, hpfCut);
            vv.setGains(inGain, noiseGain, 1.0);
            vv.setEnv(atk, dec, sus, rel);
            _v << vv;
        }

        0 => _rr;

        // start the two overlapped swing schedulers
        spork ~ _swingLR();
        spork ~ _swingRL();
    }

    // ---------------------------------
    // parameter setters (human inputs)
    // ---------------------------------
    fun void setSwingDur(dur d)      { d => swingDur; }
    fun void setOffsetDur(dur d)     { d => offsetDur; }
    fun void setCtrlDt(dur d)        { d => ctrlDt; }

    fun void setPanDepth(float x)    { x => panDepth; }
    fun void setInGain(float x)
    {
        x => inGain;
        for (0 => int i; i < nVoices; i++) { inGain => _v[i].inTap.gain; }
    }
    fun void setNoiseGain(float x)
    {
        x => noiseGain;
        for (0 => int i; i < nVoices; i++) { noiseGain => _v[i].nzG.gain; }
    }
    fun void setWetGain(float x)     { x => wetGain; wetGain => _wetBus.gain; }

    fun void setBpf(float loHz, float hiHz, float q)
    {
        loHz => bpfLo;
        hiHz => bpfHi;
        q    => bpfQ;
        for (0 => int i; i < nVoices; i++) { bpfQ => _v[i].bpf.Q; }
    }
    fun void setHpf(float hz)
    {
        hz => hpfCut;
        for (0 => int i; i < nVoices; i++) { hpfCut => _v[i].hpf.freq; }
    }

    fun void setEnv(dur a, dur d, float s, dur r)
    {
        a => atk; d => dec; s => sus; r => rel;
        for (0 => int i; i < nVoices; i++) { _v[i].env.set(atk, dec, sus, rel); }
    }

    fun void setReverbMix(float x)   { x => revMix; revMix => _rev.mix; }

    // ---------------------------------
    // voice selection
    // ---------------------------------
    fun Voice _nextVoice()
    {
        _rr => int idx;
        (_rr + 1) % nVoices => _rr;
        return _v[idx];
    }

    // ---------------------------------
    // core "play one swing" routine
    // dir = -1 for L->R, +1 for R->L start (we compute endpoints)
    // ---------------------------------
    fun void _playSwing(int dir)
    {
        Voice @ v;
        _nextVoice() @=> v;

        // reset env (keyOff then keyOn) for a clean transient
        v.env.keyOff();
        1::ms => now;

        // start concurrent pan + filter sweeps
        spork ~ _panSweep(v, dir);
        spork ~ _bpfSweep(v);

        v.env.keyOn();
        (swingDur - ctrlDt) => now;
        v.env.keyOff();
    }

    // pan sweep: overlap-friendly, independent per voice
    fun void _panSweep(Voice @ v, int dir)
    {
        // L->R: start -panDepth to +panDepth
        // R->L: start +panDepth to -panDepth
        float p0;
        float p1;

        if (dir < 0) { -panDepth => p0;  panDepth => p1; }
        else         {  panDepth => p0; -panDepth => p1; }

        // do a linear ramp over swingDur
        0::second => dur t;
        while (t < swingDur)
        {
            t / swingDur => float u;
            (p0 + (p1 - p0) * u) => v.pan.pan;

            ctrlDt => now;
            t + ctrlDt => t;
        }
        p1 => v.pan.pan;
    }

    // bandpass sweep: gives the "air cut" whoosh; independent per voice
    fun void _bpfSweep(Voice @ v)
    {
        0::second => dur t;
        while (t < swingDur)
        {
            t / swingDur => float u;

            // perceptually nicer if frequency moves exponentially:
            // f = lo * (hi/lo)^u
            (bpfHi / bpfLo) => float ratio;
            (bpfLo * Math.pow(ratio, u)) => float f;

            f => v.bpf.freq;

            ctrlDt => now;
            t + ctrlDt => t;
        }
        bpfHi => v.bpf.freq;
    }

    // ---------------------------------
    // schedulers: overlapped swing pattern
    // ---------------------------------
    fun void _swingLR()
    {
        // left->right starts at t=0, repeats every swingDur
        while (true)
        {
            spork ~ _playSwing(-1);
            swingDur => now;
        }
    }

    fun void _swingRL()
    {
        // right->left starts at offsetDur, repeats every swingDur
        offsetDur => now;
        while (true)
        {
            spork ~ _playSwing(1);
            swingDur => now;
        }
    }
}

// ============================================================
// Example usage (replace the input chain with anything)
// ============================================================

// Human input variables (edit these)
0.5::second => dur   SWING_DUR;
0.25::second => dur  OFFSET_DUR;

900.0  => float BPF_LO;
4500.0 => float BPF_HI;
6.0    => float BPF_Q;
250.0  => float HPF_CUT;

0.08   => float REV_MIX;
0.90   => float WET_GAIN;
0.20   => float NOISE_GAIN;
1.0    => float PAN_DEPTH;

// Example source (any input can go in)
SawOsc src => Gain srcG => SwordSwish sw => dac;
0.12 => srcG.gain;
220.0 => src.freq;

// init + set params
sw.init(6);                       // voices (poly) for overlap
sw.setSwingDur(SWING_DUR);
sw.setOffsetDur(OFFSET_DUR);
sw.setCtrlDt(1::ms);

sw.setBpf(BPF_LO, BPF_HI, BPF_Q);
sw.setHpf(HPF_CUT);

sw.setReverbMix(REV_MIX);
sw.setWetGain(WET_GAIN);
sw.setNoiseGain(NOISE_GAIN);
sw.setPanDepth(PAN_DEPTH);

// envelope shaping (fast attack, short decay, no sustain, smooth release)
sw.setEnv(5::ms, 60::ms, 0.0, 120::ms);

// run
while (true) { 1::second => now; }