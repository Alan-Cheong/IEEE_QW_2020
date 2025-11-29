#!/usr/bin/env python3
"""
Map-inspired multi-slit simulation using Feynman's path integral (free propagator).

- Prudential Tower: point source
- North–south streets: treated as parallel slits on a single aperture plane
- Maxwell Road: detection screen (line) where intensity is measured. The
  origin (0, 0) is set at the junction of Wallich Road and Maxwell Road.

The program computes the complex amplitude at each point on the screen by
integrating the free-particle propagator from the source to each point within
every street-slit, then from the slit to the screen. Interference arises by
summing over all allowed intermediate points (i.e. paths constrained to pass
through the streets), which is the path-integral picture with one intermediate
slice.

Two unit systems are supported:
  1) electron: uses physical electron mass and energy (eV) to get time of flight
  2) dimensionless: sets m = hbar = 1 and uses distances as times (good for
     qualitative patterns at human scales)

Example (origin at Wallich×Maxwell, screen at y=0):
  python map_path_integral.py --preset sg --units dimensionless \
      --screen-y 0 --aperture-y 400 --source-y 800 \
      --output maxwell_intensity.png

To tweak streets manually:
  python map_path_integral.py --slits "Cecil:200:20,Tras:-80:18,TgPagar:-180:22,SouthBridge:280:20,Neil:-260:20" \
      --units dimensionless --output maxwell_intensity.png

Where each slit is NAME:X_CENTER:WIDTH in meters (or arbitrary units if
--units=dimensionless).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib

# Use a non-interactive backend so this runs headless without display servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants (SI)
HBAR = 1.054_571_817e-34
ELECTRON_MASS = 9.109_383_7015e-31
ELEMENTARY_CHARGE = 1.602_176_634e-19


@dataclass
class Slit:
    name: str
    x_center: float  # meters (or arbitrary units in dimensionless mode)
    width: float     # meters (or arbitrary units in dimensionless mode)

    def sample_points(self, num_samples: int) -> Tuple[np.ndarray, float]:
        half = self.width / 2.0
        xs = np.linspace(self.x_center - half, self.x_center + half, num_samples)
        dx = xs[1] - xs[0] if num_samples > 1 else self.width
        return xs, dx


def free_kernel_prefactor(dt: float, mass: float, hbar: float) -> complex:
    """Return the 1D free-particle propagator prefactor.

    K(x, t) ~ sqrt(m / (2π i ħ t)) * exp(i m (Δx)^2 / (2 ħ t))
    The overall constant does not affect the normalized intensity, but the
    square-root form is standard and numerically more stable.
    """
    return np.sqrt(mass / (2.0 * np.pi * 1j * hbar * dt))


def free_kernel_phase(dist2: np.ndarray, dt: float, mass: float, hbar: float) -> np.ndarray:
    return np.exp(1j * mass * dist2 / (2.0 * hbar * dt))


def compute_velocity_for_electron(energy_ev: float) -> float:
    energy_j = energy_ev * ELEMENTARY_CHARGE
    return np.sqrt(2.0 * energy_j / ELECTRON_MASS)


def parse_slits_arg(slits_arg: str) -> List[Slit]:
    slits: List[Slit] = []
    for part in slits_arg.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            name, x_str, w_str = part.split(':', 2)
        except ValueError:
            raise ValueError(f"Invalid --slits segment '{part}'. Expected NAME:X:WIDTH")
        slits.append(Slit(name=name, x_center=float(x_str), width=float(w_str)))
    if not slits:
        raise ValueError("No valid slits parsed from --slits")
    return slits


def streets_preset_sg() -> List[Slit]:
    """A rough, map-inspired set of north–south streets near Maxwell Road.

    Positions are heuristic and only intended to give a visually interesting
    interference pattern; they are not geodetically accurate.
    """
    # x positions in meters (arbitrary scale); widths ~ street widths
    return [
        Slit("Neil Rd", x_center=-260.0, width=20.0),
        Slit("Tg Pagar Rd", x_center=-180.0, width=22.0),
        Slit("Tras St", x_center=-80.0, width=18.0),
        Slit("Cecil St", x_center=160.0, width=22.0),
        Slit("South Bridge Rd", x_center=260.0, width=20.0),
    ]


def preset_to_slits(preset: str) -> List[Slit]:
    key = (preset or '').strip().lower()
    if key in ("sg", "singapore", "telok-ayer"):
        return streets_preset_sg()
    raise ValueError(f"Unknown preset '{preset}'. Use 'sg' or pass --slits explicitly.")


def compute_screen_intensity(
    slits: List[Slit],
    source_x: float,
    source_y: float,
    aperture_y: float,
    screen_y: float,
    screen_width: float,
    screen_samples: int,
    slit_samples: int,
    units: str,
    energy_ev: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute intensity on the screen via one-slice path integral through slits.

    Returns (x_screen, intensity_normalized).
    """
    # No strict ordering is required; we use |Δy| to define time slices

    # Screen sampling points
    x_screen = np.linspace(-screen_width / 2.0, screen_width / 2.0, screen_samples)

    # Unit handling
    units_key = units.strip().lower()
    if units_key == "electron":
        v = compute_velocity_for_electron(energy_ev)
        dt1 = abs(aperture_y - source_y) / v
        dt2 = abs(screen_y - aperture_y) / v
        mass = ELECTRON_MASS
        hbar = HBAR
    elif units_key == "dimensionless":
        # Handy qualitative mode: m=hbar=1, and time equals distance along y.
        mass = 1.0
        hbar = 1.0
        dt1 = float(abs(aperture_y - source_y))
        dt2 = float(abs(screen_y - aperture_y))
    else:
        raise ValueError("units must be 'electron' or 'dimensionless'")

    # Precompute constants
    c1 = free_kernel_prefactor(dt1, mass, hbar)
    c2 = free_kernel_prefactor(dt2, mass, hbar)
    y1 = aperture_y
    y2 = screen_y

    amplitude = np.zeros_like(x_screen, dtype=np.complex128)

    # Accumulate contributions slit by slit to control memory use
    for slit in slits:
        xs, dx = slit.sample_points(slit_samples)

        # Source -> slit (vector over slit samples)
        dist2_src = (xs - source_x) ** 2 + (y1 - source_y) ** 2
        k1 = c1 * free_kernel_phase(dist2_src, dt1, mass, hbar)

        # Slit -> screen (matrix: screen_samples x slit_samples)
        # Avoid allocating huge matrices by chunking if needed
        # Here, for typical sizes, direct allocation is fine.
        xb = x_screen[:, None]
        dist2_scr = (xb - xs[None, :]) ** 2 + (y2 - y1) ** 2
        k2 = c2 * free_kernel_phase(dist2_scr, dt2, mass, hbar)

        # Path integral over the slit width (Riemann sum)
        amplitude += (k2 @ k1) * dx

    intensity = np.abs(amplitude) ** 2
    # Normalize for visualization
    if intensity.max() > 0:
        intensity /= intensity.max()
    return x_screen, intensity


def plot_results(
    x_screen: np.ndarray,
    intensity: np.ndarray,
    slits: List[Slit],
    aperture_y: float,
    screen_y: float,
    source_x: float,
    out_path: str,
    title: str,
):
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

    # Geometry panel
    ax0 = fig.add_subplot(gs[0])
    for i, slit in enumerate(slits):
        ax0.add_patch(
            plt.Rectangle(
                (slit.x_center - slit.width / 2.0, aperture_y - 0.5),
                slit.width,
                1.0,
                color="#4682b4",
                alpha=0.6,
                label="Street (slit)" if i == 0 else None,
            )
        )
    ax0.axhline(screen_y, color="#2f4f4f", linestyle="--", label="Maxwell Rd (screen)")
    ax0.plot([source_x], [0.0], marker="*", color="#d62728", markersize=12, label="Prudential Tower (source)")
    ax0.set_ylabel("y (distance)")
    ax0.set_xlabel("x (across streets)")
    ax0.set_title("Geometry")
    ax0.legend(loc="upper right", ncol=3, fontsize=8)

    # Intensity panel
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(x_screen, intensity, color="#111111")
    ax1.set_xlabel("x on Maxwell Rd (screen)")
    ax1.set_ylabel("Relative intensity")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    fig.suptitle("Map-inspired electron path integral (one intermediate slice)")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Feynman path-integral multi-slit over city streets")
    presets = p.add_mutually_exclusive_group()
    presets.add_argument("--preset", default="sg", help="Street preset: sg")
    p.add_argument(
        "--slits",
        help="Custom slits as 'NAME:X:WIDTH,...' (overrides --preset)",
    )
    p.add_argument("--units", default="dimensionless", choices=["dimensionless", "electron"], help="Unit system")
    p.add_argument("--energy-eV", type=float, default=50.0, help="Electron energy (eV) if --units=electron")
    p.add_argument("--source-y", type=float, default=800.0, help="y of Prudential Tower (origin at Wallich×Maxwell)")
    p.add_argument("--aperture-y", type=float, default=400.0, help="y of streets/aperture line (origin at Wallich×Maxwell)")
    p.add_argument("--screen-y", type=float, default=0.0, help="y of Maxwell Rd screen (origin at Wallich×Maxwell)")
    p.add_argument("--screen-width", type=float, default=800.0, help="Extent of screen plotted across x")
    p.add_argument("--screen-samples", type=int, default=1401, help="Number of x samples on screen")
    p.add_argument("--slit-samples", type=int, default=301, help="Samples across each street width")
    p.add_argument("--source-x", type=float, default=0.0, help="x position of the Prudential Tower source")
    p.add_argument("--output", default="maxwell_intensity.png", help="Output image path")

    args = p.parse_args()

    if args.slits:
        slits = parse_slits_arg(args.slits)
    else:
        slits = preset_to_slits(args.preset)

    x_screen, intensity = compute_screen_intensity(
        slits=slits,
        source_x=args.source_x,
        source_y=args.source_y,
        aperture_y=args.aperture_y,
        screen_y=args.screen_y,
        screen_width=args.screen_width,
        screen_samples=args.screen_samples,
        slit_samples=args.slit_samples,
        units=args.units,
        energy_ev=args.energy_eV,
    )

    title = (
        f"Origin at Wallich×Maxwell; Screen y={args.screen_y}, Streets y={args.aperture_y}, Source y={args.source_y}; "
        f"units={args.units}, energy={args.energy_eV} eV"
    )
    plot_results(
        x_screen=x_screen,
        intensity=intensity,
        slits=slits,
        aperture_y=args.aperture_y,
        screen_y=args.screen_y,
        source_x=args.source_x,
        out_path=args.output,
        title=title,
    )

    # Basic CLI printout
    print(f"Saved intensity plot to {args.output}")


if __name__ == "__main__":
    main()
