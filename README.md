# Synthetic ECG Generator based on Inverse Fractional Derivatives

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible synthetic electrocardiogram (ECG) generator that uses **inverse fractional calculus** to accurately control the morphology of P, Q, R, S, and T waves. Unlike classical models (e.g., sum of Gaussians), this approach allows independent tuning of each wave’s frequency characteristics via the fractional derivative order α.

---

##  Mathematical foundation

### Forward fractional derivative (Grünwald–Letnikov)

For a given order α ∈ (0,2], the discrete fractional derivative is computed as a convolution with weighting coefficients:

``` math
D^\alpha[x](n) = \sum_{k=0}^{M} w_k(\alpha) \cdot x(n-k)
```

```math
w_k(\alpha) = \frac{(-1)^k \Gamma(\alpha+1)}{\Gamma(k+1)\Gamma(\alpha-k+1)}, \quad w_0=1
```

The coefficients are normalized so that $\(\sum |w_k| = 1\)$.

### Inverse problem

The generator solves the equation:

``` math
D^\alpha[s](t) = \text{target}(t)
```

where `target(t)` is a desired complex envelope containing peaks at specified temporal positions (e.g., a sharp Gaussian for the R wave, broader bell‑shaped functions for P and T). The solution s(t) is obtained via regularised least squares using a Toeplitz matrix:

```math
\mathbf{s} = \left( \mathbf{T}^H \mathbf{T} + \lambda \mathbf{I} \right)^{-1} \mathbf{T}^H \mathbf{z}
```

The real part of $\( \mathbf{s} \)$ becomes the output ECG signal.

### Hybrid synthesis

Different waves are synthesised with **different α orders**, mimicking their frequency characteristics:

| Wave | Typical α | Characteristic |
|------|-----------|----------------|
| R    | 1.6–2.0   | Sharp high‑frequency peak |
| Q, S | 1.2–1.8   | Short negative deflections |
| P    | 0.5–0.9   | Low‑frequency smooth wave |
| T    | 0.8–1.1   | Broad mid‑frequency wave |

Each wave is generated independently, after which their real parts are summed. Controlled white noise and low‑frequency baseline drift are added to the final signal.

---

## Installation

```bash
git clone https://github.com/yourusername/inverse-fractional-ecg-synth.git
cd inverse-fractional-ecg-synth
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0   # for plotting (optional)
```

---

## Usage

### Command line

Generate a 10‑second ECG with default parameters:

```bash
python inverse_ecg_synth.py --output normal_ecg.wav
```

Generate a 30‑second signal with custom heart rate and α orders:

```bash
python inverse_ecg_synth.py --duration 30 --hr 90 --r_alpha 1.9 --p_alpha 0.6 --output custom_ecg.wav
```

### As a Python module

```python
from inverse_ecg_synth import InverseFractionalECGSynthesizer

synth = InverseFractionalECGSynthesizer(fs=1000)
ecg, time, info = synth.generate_ecg_from_peaks(
    duration=10,
    hr_bpm=75,
    r_alpha=1.8,
    p_alpha=0.7,
    t_alpha=0.9,
    noise_level=0.02
)

# Save to WAV
from scipy.io import wavfile
wavfile.write("my_ecg.wav", 1000, (ecg * 32767).astype(np.int16))

# Information about R‑peaks
print(f"Generated {len(info['r_positions'])} R peaks")
```

---

## Parameters

### Signal parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--duration` | 10.0 | Signal length (seconds) |
| `--fs` | 1000 | Sampling frequency (Hz) |
| `--hr` | 75.0 | Mean heart rate (beats per minute) |
| `--noise` | 0.02 | Standard deviation of additive white Gaussian noise (mV) |
| `--drift_amp` | 0.05 | Amplitude of sinusoidal baseline drift (mV) |
| `--drift_freq` | 0.15 | Frequency of baseline drift (Hz) |

### Wave parameters (can be changed via function arguments)

| Wave | Amplitude (mV) | α (order) | Offset from R (s) |
|------|----------------|-----------|-------------------|
| P    | 0.15           | 0.7       | -0.18 (before R)  |
| Q    | -0.05          | 1.8       | -0.05             |
| R    | 1.0            | 1.8       | 0.0               |
| S    | -0.1           | 1.8       | 0.05              |
| T    | 0.3            | 0.9       | 0.25              |

All wave parameters can be customised by passing the corresponding arguments to `generate_ecg_from_peaks` (see the source code).

---

## Example output

Below is a 10‑second synthetic ECG (HR = 75 bpm, α_R = 1.8, α_P = 0.7, α_T = 0.9). Red dashed lines mark the true R‑peak positions.

---

## Customising wave morphology

The shape of each wave is determined by a target envelope (Gaussian by default). You can change it by editing the appropriate section in the `generate_ecg_from_peaks` method. For example, to make the T‑wave asymmetric:

```python
# Instead of a Gaussian, use a skewed function
left = np.exp(-((t_center - idx)/sigma)**2)
right = np.exp(-((t_center - idx)/(1.5*sigma))**2)
gauss = np.where(t_center >= idx, left, right)
```

The inverse solver will automatically compute the required ECG contribution.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

**Issue: The generated signal looks very noisy / has large spikes.**  
**Solution:** Reduce `--noise` and increase the `regularization` parameter in the call to `solve_inverse_fractional` (default is 1e-5). You can also reduce `drift_amp`.

**Issue: Memory error for long durations (> 30 s).**  
**Solution:** Direct matrix inversion requires large memory for long signals. For durations longer than 30 seconds, consider implementing an iterative solver (e.g., conjugate gradient). This is a planned improvement.

**Issue: Not enough heart rate variability.**  
**Solution:** The `--hr_std` parameter (relative standard deviation of RR intervals) defaults to 0.03 (3%). Increase it to 0.05–0.1 for more physiological variability.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For questions or suggestions, please open an issue on GitHub.
