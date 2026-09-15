# Project 04 — Dehazing: complete workflow

The [README](README.md) states the findings. This states how they were reached,
in what order, and what each decision cost.

---

## 1 · Why this question, and not "which dehazing method is best"

Haze has something almost no other image degradation has: **a physical model with
every unknown written down.**

```
I(x) = J(x)·t(x) + A·(1 − t(x))          t(x) = exp(−β·d(x))
```

So the question can be sharper than "which output looks clearest":

> **Did the method recover the physics, or did it just add contrast?**

That is answerable because generating the haze retains `J`, `t` **and** `A`. Most
projects in this repo know the clean image; this one also knows the intermediate
quantity every method is trying to estimate.

---

## 2 · The complete workflow

```mermaid
flowchart TD
    subgraph GEN["A · Generate, keeping three truths"]
        A1[Clear image J]
        A2[Depth map d<br/>far plane at the TOP]
        A3["t = exp(−β·d)"]
        A4["I = J·t + A·(1−t)"]
    end

    subgraph EST["B · What every method must estimate"]
        B1[Airlight A<br/>brightest within the dark channel]
        B2[Transmission t<br/>dark channel prior]
        B3[Refine t<br/>guided filter]
    end

    subgraph INV["C · Invert"]
        C1["J = (I−A)/max(t, t_min) + A"]
        C2[ORACLE: invert with the TRUE t and A]
    end

    subgraph SCORE["D · Score two different things"]
        D1[Output quality<br/>PSNR, SSIM vs J]
        D2[Physics recovery<br/>MAE vs the true t, error vs the true A]
        D3[No reference<br/>RMS contrast]
    end

    subgraph SWEEP["E · The experiments"]
        E1[162-combination parameter sweep]
        E2[4 airlight estimators]
        E3[beta 0.2 - 3.0]
    end

    A1 --> A4
    A2 --> A3 --> A4
    A4 --> B1 --> B2 --> B3 --> C1
    A3 -.true t.-> C2
    A1 -.retained as truth.-> D1
    C1 --> D1 & D3
    B1 & B3 -.-> D2
    C2 --> D1
    D1 --> E1 & E2 & E3
    D2 --> E2

    style A3 fill:#fef3c7,stroke:#d97706
    style D2 fill:#fee2e2,stroke:#dc2626
    style GEN fill:#eff6ff
```

**The two highlighted boxes are the project.** The transmission map (amber) is
the quantity being estimated, and scoring it directly (red) is what separates
"removed haze" from "added contrast".

---

## 3 · Build order

| Step | What | Why here |
|---:|---|---|
| 1 | The scattering model in the generator | Without exact `t`, the central question is unaskable. |
| 2 | The oracle | Converts "18.9 dB" into "31 dB below what was achievable". |
| 3 | DCP + guided refine | The real method. |
| 4 | **CLAHE as a control** | Added early, deliberately: something that *looks* like dehazing and models nothing. |
| 5 | Retinex | A model that provably cannot fit — multiplicative applied to additive. |
| 6 | Parameter sweep | The paper's constants were 0.86 dB worse on this data. |
| 7 | Airlight comparison | Added after noticing `A = 1.000`. Produced the project's finding. |
| 8 | UI, figures, docs | Last. |

---

## 4 · Decisions, with reasoning

### Why the far plane is at the top of the frame

`add_haze` puts the deepest haze at the **top**. That is how an outdoor scene
recedes: the horizon is distant and hazy, the ground at your feet is close and
clear. Getting it upside down puts the densest haze on the nearest object, which
no method can then explain — and every result would be a measurement of that bug.

### Why CLAHE is in the table

Because it wins a column. CLAHE reaches **0.181 RMS contrast**, higher than the
physical method's 0.156, while scoring **12.93 dB** against 19.50. If it were not
in the table, a reader could reasonably assume contrast tracks quality here. It
does not, and the only way to show that is to include something that games it.

### Why ω is 0.80 and not the paper's 0.95

Measured. 162 combinations over six images:

```
omega  patch  radius   eps    tmin    PSNR   SSIM
 0.95     15      40  1e-03   0.10   18.64  0.852   <- He et al.
 0.80      7      40  1e-02   0.05   19.50  0.872   <- measured best
```

Copying constants from a paper written for different images, at a different
resolution, is not reproduction — it is cargo-culting. The values are in the
source with the measurement that chose them.

### Why the *worse* airlight estimator is the default

This was the hard call. Four estimators:

| | Airlight error | Transmission MAE | PSNR |
|---|---:|---:|---:|
| Brightest (default) | 0.0631 | 0.1059 | **19.496** |
| Median | **0.0508** | **0.0967** | 18.495 |

The default is worse at estimating both intermediate quantities and better at
producing the image. The recovery `J = (I−A)/t + A` is over-estimating `A` and
over-estimating `t`; the errors partially cancel, and fixing one alone breaks the
cancellation.

The project optimises **output quality**, because that is what a dehazing method
is for — and the losing configuration stays in the table, because the comparison
*is* the result.

---

## 5 · What each stage costs

Six images at β = 1.4:

| Stage | Time |
|---|---:|
| CLAHE | 0.88 ms |
| Gamma control | 10.6 ms |
| Oracle inversion | 10.2 ms |
| Dark channel prior | 30.9 ms |
| DCP + guided refine | 39.8 ms |
| Multi-scale Retinex | **426 ms** |
| **Full `run.py`** | **~90 s** |

Retinex is **11× slower than the best method and 10.6 dB worse.** Reporting the
time column is what makes that visible.

---

## 6 · Reproducing it

```bash
python run.py                 # regenerates every number and figure
streamlit run ui/app.py       # the interactive app
python infer.py photo.jpg     # your own hazy image
```

`run.py` rewrites `results/results.json` and `results/tables.md`; the README's
numbers are copied from those files rather than typed.

---

## 7 · What would come next

* **A non-monotonic depth map.** A near object against a distant sky is the case
  where the dark channel prior's assumption is most likely to break.
* **Spatially varying airlight.** A single global `A` is wrong near the sun.
* **Project 26** takes the contrast-versus-PSNR disagreement seen here as its own
  subject — which metric is right, and how would you know.
* **The cancellation finding deserves its own study.** It appeared twice
  independently here (once with a quadtree airlight, once with a median one), and
  a proper treatment would map the error surface rather than sample four points
  on it.
