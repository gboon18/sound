

---

# Time-varying `mix` and `lifter_q` (linear interpolation)

Instead of constant `--mix` and `--lifter_q`, you can provide time-dependent control curves using JSON dicts:

- `--mix_map` : time (seconds) → mix value
- `--lifter_q_map` : time (seconds) → lifter_q value

Values are **linearly interpolated** between keys and **clamped** outside the range to the nearest endpoint.

Example (fade in timbre strength over 5 seconds):

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio input.wav --out_wav out.wav \
  --mix_map '{"0.0": 0.0, "5.0": 1.0}'
```

Example (smoothen envelope over time):

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio input.wav --out_wav out.wav \
  --lifter_q_map '{"0.0": 60, "5.0": 20}'
```

Use both together:

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio input.wav --out_wav out.wav \
  --mix_map '{"0.0": 0.2, "3.0": 1.0}' \
  --lifter_q_map '{"0.0": 50, "3.0": 25}'
```

Notes:
- If `--mix_map` is not provided, `--mix` is used.
- If `--lifter_q_map` is not provided, `--lifter_q` is used.
