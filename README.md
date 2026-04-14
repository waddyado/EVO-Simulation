# EVO 3D Simulation Lab

This is a program to simulate my mathematical Exotic Vacuum Object theory that I based upon Ken Shoulders’ foundational EV/EVO (Electrum Validum / Exotic Vacuum Object) experiments from the 1980s–2000s


Interactive desktop simulator for EV Field Theory v1, implemented in pure Python with:

- `numpy`
- `scipy`
- `sympy`
- `matplotlib`

Main script: `evo_simulation_3d.py`

---

## 1) What This Program Does

The app launches a live 3D EV simulation and shows:

- EV state vector behavior (`Q`, `R`, `X`, `V`, `Psi`, `sigma`, `chi`, `eta`)
- Energy decomposition (`EQ`, `EC`, `Eint`, `Ekin`, `Ebdy`)
- Radius quantization view (Eq. 13-style curves)
- Boundary/guide interaction visuals
- Emission model toggling
- Monte-Carlo stability estimation
- Finite-difference container potential slice overlay

It is designed as an interactive research tool where you can tune model parameters with sliders and trigger actions with buttons.

---

## 2) Requirements

Use Python 3.9+ (3.10+ recommended).

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

- `numpy`
- `scipy`
- `sympy`
- `matplotlib`

---

## 3) Run Instructions

From the project root folder:

```bash
python evo_simulation_3d.py
```

If your system maps `python` differently, use:

```bash
py evo_simulation_3d.py
```

The app opens a desktop matplotlib window with:

- Left: primary 3D EV scene
- Right: diagnostic plots
- Bottom: controls (sliders and buttons)
- Bottom status line: current state and simulation information

---

## 4) UI Overview and How To Use

## 4.1 Sliders (bottom area)

Use sliders to set model parameters in real time:

- `a`, `b`, `beta`: container-law terms
- `R0`, `gamma`, `kC`: radius quantization controls
- `Ne`: electron count scale
- `Ez`, `By`: external electric and magnetic field components
- `ck0`, `dk0`: modal stiffness/nonlinearity terms

Tip: move one slider at a time and watch the energy + stability updates.

## 4.2 Buttons

- `Minimize EEV`  
  Runs numerical minimization to seek a locally stable radius/mode state.

- `Launch EV`  
  Reinitializes position/velocity and starts a fresh trajectory.

- `Run Monte-Carlo`  
  Runs 500+ random formation trials and estimates stable creation probability.

- `Reset Simulation`  
  Resets state/history to defaults.

- `Toggle Emission`  
  Turns emission law contribution on/off.

- `Save Data`  
  Writes `evo_simulation_data.npz` in the project root.

## 4.3 Plots and Indicators

- **3D Scene**: EV sphere, velocity arrow, modal field lines, guide surfaces, potential overlay.
- **sigma/chi/eta panels**: interaction channels over time.
- **Radius quantization panel**: family curves and highlighted ~1 um scale.
- **Energy panel**: bar plot of `EQ`, `EC`, `Eint`, `Ekin`, `Ebdy`.
- **MC panel**: stable/unstable formation probability after Monte-Carlo run.
- **Status bar**: current `R*`, total `EEV`, stability classification.
- **FPS(sim)**: smoothed simulation FPS estimate.

---

## 5) Suggested Workflow

1. Start app with defaults.
2. Click `Launch EV`.
3. Sweep `Ne`, `a`, `beta`, and `Ez`.
4. Click `Minimize EEV` after parameter changes.
5. Run `Run Monte-Carlo` to compare formation probability.
6. Toggle emission and inspect changes.
7. Save result snapshots with `Save Data`.

---

## 6) Performance Notes

This version is tuned for better interactivity:

- lower-cost 3D meshes for faster redraw
- finite-difference solve updated less frequently
- heavy panel recomputation at reduced cadence
- lower-overhead ODE integrator settings

If you still see low FPS:

1. Close other heavy apps/GPU workloads.
2. Keep figure window size moderate.
3. Avoid very frequent slider drags.
4. Use an up-to-date Python + matplotlib build.

---

## 7) Output Files

- `evo_simulation_data.npz`: saved state/history and Monte-Carlo results.

Load quickly in Python:

```python
import numpy as np
d = np.load("evo_simulation_data.npz", allow_pickle=True)
print(d.files)
```

---

## 8) Troubleshooting

- **Window does not appear**  
  Ensure desktop GUI support is available for matplotlib on your Python install.

- **`ModuleNotFoundError`**  
  Reinstall deps: `pip install -r requirements.txt`.

- **Very slow simulation**  
  Use the performance notes above; update GPU/display drivers if needed.

