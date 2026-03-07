// grandmother_like.ck
// Grandmother-inspired mono synth voice using Chugraph (Chubgraph is deprecated).
// Run: chuck grandmother_like.ck

public class GrandMotherVoice extends Chugraph
{
    // --------- patch points ---------
    // external audio can be patched into inlet
    inlet => Gain extIn => Gain mix => Gain drive => LPF f1 => LPF f2 => LPF f3 => LPF f4
          => ADSR ampEnv => Gain velGain => Dyno outLim => NRev rev => outlet;

    // oscillators + noise into mixer
    TriOsc  vco1_tri => Gain vco1_gain => mix;
    SawOsc  vco1_saw => Gain vco1s_gain => mix;
    SqrOsc  vco1_sqr => Gain vco1q_gain => mix;

    TriOsc  vco2_tri => Gain vco2_gain => mix;
    SawOsc  vco2_saw => Gain vco2s_gain => mix;
    SqrOsc  vco2_sqr => Gain vco2q_gain => mix;

    Noise   nsrc => Gain noiseGain => mix;

    // LFO routing (to blackhole so it runs even if not directly audible)
    SinOsc lfo => Gain lfoPitch => blackhole;
    SinOsc lfo2 => Gain lfoCutoff => blackhole;

    // filter envelope (separate ADSR used as a control-rate signal)
    // note: ADSR is audio-rate; we sample it periodically for cutoff updates
    ADSR filtEnv;

    // --------- internal state ---------
    48 => int baseOffset;           // MIDI note offset (48 = C3-ish)
    0 => int lastNote;
    0 => int gateIsOn;

    0.0 => float velAmp;            // velocity scaling 0..1

    0.0 => float targetFreq;
    0.0 => float currentFreq;

    30::ms => dur glideTime;        // portamento
    1 => int doGlide;

    800.0 => float cutoffBaseHz;    // base cutoff
    0.7 => float resonanceQ;        // Q for each LPF stage
    1500.0 => float envAmtHz;       // filter env amount in Hz
    0.0 => float lfoPitchAmt;       // Hz-ish via osc freq modulation
    0.0 => float lfoCutoffAmt;      // Hz modulation on cutoff

    // --------- init defaults ---------
    fun void init()
    {
        // kill all osc outputs until selected
        0 => vco1_gain.gain => vco1s_gain.gain => vco1q_gain.gain;
        0 => vco2_gain.gain => vco2s_gain.gain => vco2q_gain.gain;

        0 => extIn.gain;
        0 => noiseGain.gain;

        // drive
        1.0 => drive.gain;

        // hard brick-wall limiter — must call limit() or Dyno is passthrough
        outLim.limit();
        0::ms  => outLim.attackTime;   // instantaneous attack: no transient slips through
        20::ms => outLim.releaseTime;  // fast recovery
        0.80   => outLim.thresh;       // ceiling
        0.0    => outLim.slopeAbove;   // CRUCIAL: infinite ratio (true hard limit, not 10:1)
        1.0    => outLim.gain;         // neutral output gain (velocity handled by velGain)

        // initialize velGain now so no 1-sample window at default gain=1.0
        0.35 => velGain.gain;

        0.08 => rev.mix;

        // amp envelope (fast by default, tweak via setters)
        5::ms   => ampEnv.attackTime;
        120::ms => ampEnv.decayTime;
        0.65    => ampEnv.sustainLevel;
        120::ms => ampEnv.releaseTime;

        // filter envelope
        3::ms   => filtEnv.attackTime;
        140::ms => filtEnv.decayTime;
        0.0     => filtEnv.sustainLevel;
        180::ms => filtEnv.releaseTime;

        // filter base
        setRes(35);
        setCutoff(40);

        // LFO defaults
        5.0 => lfo.freq => lfo2.freq;
        0 => lfoPitch.gain => lfoCutoff.gain;

        // start modulation/control shreds
        spork ~ controlLoop();
        spork ~ glideLoop();
    }

    // --------- oscillator selection ---------
    // 0 = off, 1 = tri, 2 = saw, 3 = square
    fun void setVco1Wave(int t)
    {
        0 => vco1_gain.gain => vco1s_gain.gain => vco1q_gain.gain;
        if(t == 1) 0.30 => vco1_gain.gain;
        if(t == 2) 0.30 => vco1s_gain.gain;
        if(t == 3) 0.30 => vco1q_gain.gain;
    }

    fun void setVco2Wave(int t)
    {
        0 => vco2_gain.gain => vco2s_gain.gain => vco2q_gain.gain;
        if(t == 1) 0.30 => vco2_gain.gain;
        if(t == 2) 0.30 => vco2s_gain.gain;
        if(t == 3) 0.30 => vco2q_gain.gain;
    }

    // --------- mixer levels ---------
    fun void setVco1Level(float x) { clamp01(x) => float y; y => vco1_gain.gain => vco1s_gain.gain => vco1q_gain.gain; }
    fun void setVco2Level(float x) { clamp01(x) => float y; y => vco2_gain.gain => vco2s_gain.gain => vco2q_gain.gain; }

    fun void setNoise(float amt01)
    {
        clamp01(amt01) => noiseGain.gain;
    }

    fun void setExtIn(float amt01)
    {
        clamp01(amt01) => extIn.gain;
    }

    // --------- tuning / detune ---------
    fun void setDetune(float cents)
    {
        // cents -> ratio
        Math.pow(2.0, cents / 1200.0) => float r;
        // store as a multiplier by reusing vco2_* freq updates in glideLoop()
        // we’ll apply this ratio there via vco2Ratio
        r => vco2Ratio;
    }
    1.0 => float vco2Ratio;

    fun void setBaseOffset(int semis) { semis => baseOffset; }

    // --------- glide ---------
    fun void setGlideMs(float glideMs)
    {
        if(glideMs <= 0.0) { 0 => doGlide; 1::ms => glideTime; return; }
        1 => doGlide;
        glideMs::ms => glideTime;
    }

    // --------- filter controls ---------
    // amount 0..100 mapped roughly to 40..8000 Hz
    fun void setCutoff(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        // exponential-ish mapping
        40.0 * Math.pow(200.0, amount / 100.0) => cutoffBaseHz;
    }

    // amount 0..100 -> Q ~ 0.4..4.0 (per stage; cascaded LPFs can get intense)
    fun void setRes(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        0.4 + 3.6 * (amount / 100.0) => resonanceQ;
        resonanceQ => f1.Q => f2.Q => f3.Q => f4.Q;
    }

    // env amount 0..100 -> 0..6000 Hz
    fun void setEnvAmt(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        6000.0 * (amount / 100.0) => envAmtHz;
    }

    // --------- drive / reverb ---------
    // drive 0..100 -> gain 0.8..6
    fun void setDrive(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        0.8 + 5.2 * (amount / 100.0) => drive.gain;
    }

    fun void setReverb(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        0.25 * (amount / 100.0) => rev.mix;
    }

    // --------- LFO ---------
    fun void setLfoRate(float hz)
    {
        if(hz < 0.05) 0.05 => hz;
        hz => lfo.freq => lfo2.freq;
    }

    // amounts 0..100
    fun void setLfoPitch(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        // small pitch modulation in Hz
        8.0 * (amount / 100.0) => lfoPitchAmt;
        lfoPitchAmt => lfoPitch.gain;
    }

    fun void setLfoCutoff(float amount)
    {
        if(amount < 0) 0 => amount;
        if(amount > 100) 100 => amount;
        900.0 * (amount / 100.0) => lfoCutoffAmt;
        lfoCutoffAmt => lfoCutoff.gain;
    }

    // --------- envelopes ---------
    fun void setAmpADSR(dur a, dur d, float s, dur r)
    {
        a => ampEnv.attackTime;
        d => ampEnv.decayTime;
        s => ampEnv.sustainLevel;
        r => ampEnv.releaseTime;
    }

    fun void setFiltADSR(dur a, dur d, float s, dur r)
    {
        a => filtEnv.attackTime;
        d => filtEnv.decayTime;
        s => filtEnv.sustainLevel;
        r => filtEnv.releaseTime;
    }

    // --------- note events ---------
    fun void noteOn(int midiNote, int velocity)
    {
        midiNote => lastNote;
        1 => gateIsOn;

        // velocity 1..127 -> 0..1
        (velocity $ float) / 127.0 => velAmp;
        if(velAmp < 0.0) 0.0 => velAmp;
        if(velAmp > 1.0) 1.0 => velAmp;

        Std.mtof(baseOffset + midiNote) => targetFreq;
        if(currentFreq <= 0.0) targetFreq => currentFreq;

        // envelopes
        1 => ampEnv.keyOn;
        1 => filtEnv.keyOn;
    }

    fun void noteOff(int midiNote)
    {
        // mono behavior: only release if it matches the held note
        if(midiNote == lastNote)
        {
            0 => gateIsOn;
            1 => ampEnv.keyOff;
            1 => filtEnv.keyOff;
        }
    }

    // --------- loops ---------
    fun void glideLoop()
    {
        1::ms => dur dt;
        while(true)
        {
            if(doGlide == 0)
            {
                targetFreq => currentFreq;
            }
            else
            {
                // simple 1-pole smoothing based on glideTime
                (dt / 1::samp) / (glideTime / 1::samp) => float alpha;
                if(alpha > 1.0) 1.0 => alpha;
                currentFreq + (targetFreq - currentFreq) * alpha => currentFreq;
            }

            // apply oscillator frequencies
            // pitch LFO is applied as additive Hz (small)
            (currentFreq + lfo.last() * lfoPitchAmt) => float f1hz;
            if(f1hz < 0.01) 0.01 => f1hz;

            f1hz => vco1_tri.freq => vco1_saw.freq => vco1_sqr.freq;

            (f1hz * vco2Ratio) => float f2hz;
            if(f2hz < 0.01) 0.01 => f2hz;
            f2hz => vco2_tri.freq => vco2_saw.freq => vco2_sqr.freq;

            dt => now;
        }
    }

    fun void controlLoop()
    {
        2::ms => dur dt;
        while(true)
        {
            // filter cutoff = base + env + LFO
            cutoffBaseHz
              + (filtEnv.value() * envAmtHz)
              + (lfo2.last() * lfoCutoffAmt)
              => float cf;

            // guard rails
            if(cf < 20.0) 20.0 => cf;
            if(cf > 16000.0) 16000.0 => cf;

            cf => f1.freq => f2.freq => f3.freq => f4.freq;

            // velocity scaling (simple)
            // keep envelope as the main shaper; just scale overall output level slightly
            (0.35 + 0.65 * velAmp) => velGain.gain;

            dt => now;
        }
    }

    // --------- helpers ---------
    fun float clamp01(float x)
    {
        if(x < 0.0) return 0.0;
        if(x > 1.0) return 1.0;
        return x;
    }
}

// ------------------- demo / usage -------------------
GrandMotherVoice gm => dac;

// default patch: saw + square, some drive, env sweep
gm.init();
gm.setVco1Wave(2);       // saw
gm.setVco2Wave(3);       // square
gm.setDetune(7.0);       // cents
gm.setDrive(35);
gm.setCutoff(35);
gm.setRes(40);
gm.setEnvAmt(55);

gm.setLfoRate(5.5);
gm.setLfoPitch(10);
gm.setLfoCutoff(18);

gm.setNoise(0.02);
gm.setReverb(10);
gm.setGlideMs(35);

// ─ Recording tap (WvOut off by default; enabled via /gm/record/start) ─
WvOut recOut;
Gain  recTap;
gm => recTap => recOut => blackhole;
0.0 => recTap.gain;

// ---- OSC listener (receives from gm_gui.py on port 9000) ----
fun void oscLoop()
{
    OscIn oin;
    OscMsg omsg;
    9000 => oin.port;
    oin.addAddress("/gm/cutoff, f");
    oin.addAddress("/gm/res, f");
    oin.addAddress("/gm/envamt, f");
    oin.addAddress("/gm/drive, f");
    oin.addAddress("/gm/reverb, f");
    oin.addAddress("/gm/glidems, f");
    oin.addAddress("/gm/lforate, f");
    oin.addAddress("/gm/lfopitch, f");
    oin.addAddress("/gm/lfocutoff, f");
    oin.addAddress("/gm/vco1wave, i");
    oin.addAddress("/gm/vco2wave, i");
    oin.addAddress("/gm/vco1level, f");
    oin.addAddress("/gm/vco2level, f");
    oin.addAddress("/gm/detune, f");
    oin.addAddress("/gm/noise, f");
    oin.addAddress("/gm/extin, f");
    oin.addAddress("/gm/ampA, f");
    oin.addAddress("/gm/ampD, f");
    oin.addAddress("/gm/ampS, f");
    oin.addAddress("/gm/ampR, f");
    oin.addAddress("/gm/filtA, f");
    oin.addAddress("/gm/filtD, f");
    oin.addAddress("/gm/filtS, f");
    oin.addAddress("/gm/filtR, f");
    oin.addAddress("/gm/seq/noteon, i");
    oin.addAddress("/gm/seq/noteoff, i");
    oin.addAddress("/gm/record/start, s");
    oin.addAddress("/gm/record/stop, i");

    while(true)
    {
        oin => now;
        while(oin.recv(omsg))
        {
            if(omsg.address == "/gm/cutoff")   gm.setCutoff(omsg.getFloat(0));
            if(omsg.address == "/gm/res")       gm.setRes(omsg.getFloat(0));
            if(omsg.address == "/gm/envamt")    gm.setEnvAmt(omsg.getFloat(0));
            if(omsg.address == "/gm/drive")     gm.setDrive(omsg.getFloat(0));
            if(omsg.address == "/gm/reverb")    gm.setReverb(omsg.getFloat(0));
            if(omsg.address == "/gm/glidems")   gm.setGlideMs(omsg.getFloat(0));
            if(omsg.address == "/gm/lforate")   gm.setLfoRate(omsg.getFloat(0));
            if(omsg.address == "/gm/lfopitch")  gm.setLfoPitch(omsg.getFloat(0));
            if(omsg.address == "/gm/lfocutoff") gm.setLfoCutoff(omsg.getFloat(0));
            if(omsg.address == "/gm/vco1wave")  gm.setVco1Wave(omsg.getInt(0));
            if(omsg.address == "/gm/vco2wave")  gm.setVco2Wave(omsg.getInt(0));
            if(omsg.address == "/gm/vco1level") gm.setVco1Level(omsg.getFloat(0));
            if(omsg.address == "/gm/vco2level") gm.setVco2Level(omsg.getFloat(0));
            if(omsg.address == "/gm/detune")    gm.setDetune(omsg.getFloat(0));
            if(omsg.address == "/gm/noise")     gm.setNoise(omsg.getFloat(0));
            if(omsg.address == "/gm/extin")     gm.setExtIn(omsg.getFloat(0));
            if(omsg.address == "/gm/ampA")  { omsg.getFloat(0) => float t; t::ms => gm.ampEnv.attackTime; }
            if(omsg.address == "/gm/ampD")  { omsg.getFloat(0) => float t; t::ms => gm.ampEnv.decayTime; }
            if(omsg.address == "/gm/ampS")  { omsg.getFloat(0) => gm.ampEnv.sustainLevel; }
            if(omsg.address == "/gm/ampR")  { omsg.getFloat(0) => float t; t::ms => gm.ampEnv.releaseTime; }
            if(omsg.address == "/gm/filtA") { omsg.getFloat(0) => float t; t::ms => gm.filtEnv.attackTime; }
            if(omsg.address == "/gm/filtD") { omsg.getFloat(0) => float t; t::ms => gm.filtEnv.decayTime; }
            if(omsg.address == "/gm/filtS") { omsg.getFloat(0) => gm.filtEnv.sustainLevel; }
            if(omsg.address == "/gm/filtR") { omsg.getFloat(0) => float t; t::ms => gm.filtEnv.releaseTime; }
            // sequencer: GUI sends absolute MIDI note; adjust for baseOffset (48)
            if(omsg.address == "/gm/seq/noteon")  gm.noteOn(omsg.getInt(0) - gm.baseOffset, 100);
            if(omsg.address == "/gm/seq/noteoff") gm.noteOff(gm.lastNote);
            // recording
            if(omsg.address == "/gm/record/start") {
                omsg.getString(0) => string rpath;
                rpath => recOut.wavFilename;
                1.0 => recTap.gain;
            }
            if(omsg.address == "/gm/record/stop") {
                recOut.closeFile();
                0.0 => recTap.gain;
            }
        }
    }
}
spork ~ oscLoop();

// keep main shred alive; sequencer is driven from gm_gui.py via OSC
while(true) { 1::second => now; }