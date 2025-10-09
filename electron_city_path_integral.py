#!/usr/bin/env python3
"""
Electron interference across city streets modeled via Feynman-style path summation.

We approximate:
- Source (Prudential Tower) at y = -L1
- Barrier made of buildings with open 'slits' where streets run: Telok Ayer, Amoy, Stanley at y = 0
- Maxwell Road as the detection screen at y = +L2

Geometry uses an abstract meter-like unit; pick values below to vaguely match the map scale.
Wavelength is an adjustable effective de Broglie wavelength; we pick a larger-than-physical value so
interference fringes are visible over city-block scales.

Outputs:
- outputs/geometry.png
- outputs/intensity.png

Run:
  python electron_city_path_integral.py --show
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class StreetSlit:
    name: str
    center_x: float  # horizontal position of the street center (units)
    width: float     # open width of the street (units)


@dataclass
class CityInterferenceConfig:
    # Distances along y
    source_y: float = -200.0  # Prudential Tower relative to barrier (units)
    barrier_y: float = 0.0
    screen_y: float = 300.0   # Maxwell Road distance from barrier (units)

    # Effective wavelength (units)
    wavelength: float = 50.0

    # Streets treated as slits along the barrier line (y = 0)
    streets: Tuple[StreetSlit, ...] = (
        StreetSlit("Telok Ayer St", center_x=-60.0, width=12.0),
        StreetSlit("Amoy St",        center_x=0.0,   width=10.0),
        StreetSlit("Stanley St",     center_x=60.0,  width=10.0),
    )

    # Sampling resolution across each slit and along the screen
    points_per_meter_slit: float = 1.0  # increase for higher accuracy
    screen_span: float = 400.0          # total x-span of Maxwell Road sampled (units)
    screen_points: int = 1201           # resolution along the screen

    # Amplitude decay options
    use_1_over_r_decay: bool = True


class CityPathIntegralSimulator:
    def __init__(self, config: CityInterferenceConfig) -> None:
        self.cfg = config
        self.k = 2.0 * np.pi / self.cfg.wavelength

    def _slit_samples(self, slit: StreetSlit) -> np.ndarray:
        half_w = slit.width / 2.0
        # Ensure at least a few points per slit
        num_samples = max(3, int(np.ceil(slit.width * self.cfg.points_per_meter_slit)))
        xs = np.linspace(slit.center_x - half_w, slit.center_x + half_w, num_samples)
        return xs

    def _segment_amplitude(self, source_x: float, slit_x: np.ndarray, screen_x: np.ndarray) -> np.ndarray:
        # Distances: source->slit and slit->screen for each screen point
        r1 = np.sqrt((slit_x - source_x) ** 2 + (self.cfg.barrier_y - self.cfg.source_y) ** 2)
        r2 = np.sqrt((screen_x[:, None] - slit_x[None, :]) ** 2 + (self.cfg.screen_y - self.cfg.barrier_y) ** 2)

        phase = self.k * (r1[None, :] + r2)
        if self.cfg.use_1_over_r_decay:
            decay = 1.0 / (r1[None, :] * r2)
        else:
            decay = 1.0

        # Integrate over slit_x by summing and multiplying by dx
        dx = (slit_x[-1] - slit_x[0]) / max(1, (len(slit_x) - 1))
        contribution = np.sum(np.exp(1j * phase) * decay, axis=1) * dx
        return contribution

    def simulate(self, source_x: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        screen_x = np.linspace(-self.cfg.screen_span / 2.0, self.cfg.screen_span / 2.0, self.cfg.screen_points)
        total_amp = np.zeros_like(screen_x, dtype=np.complex128)

        for slit in self.cfg.streets:
            slit_x = self._slit_samples(slit)
            total_amp += self._segment_amplitude(source_x=source_x, slit_x=slit_x, screen_x=screen_x)

        intensity = np.abs(total_amp) ** 2
        # Normalize for visualization
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)
        return screen_x, total_amp, intensity

    def plot_geometry(self, source_x: float = 0.0, save_path: str | None = None) -> None:
        fig, ax = plt.subplots(figsize=(9, 6))

        # Draw source, barrier with slits, and screen line
        ax.scatter([source_x], [self.cfg.source_y], c="red", label="Prudential Tower (source)")

        # Barrier line segments excluding slits for visibility
        barrier_x_min = -self.cfg.screen_span / 2.0
        barrier_x_max = +self.cfg.screen_span / 2.0
        slit_intervals = [(s.center_x - s.width/2, s.center_x + s.width/2) for s in self.cfg.streets]
        slit_intervals.sort()
        # Draw barrier as a set of opaque segments between slits
        segments: List[Tuple[float, float]] = []
        cursor = barrier_x_min
        for a, b in slit_intervals:
            if a > cursor:
                segments.append((cursor, a))
            cursor = b
        if cursor < barrier_x_max:
            segments.append((cursor, barrier_x_max))
        for a, b in segments:
            ax.plot([a, b], [self.cfg.barrier_y, self.cfg.barrier_y], color="black", linewidth=3, alpha=0.6)

        # Draw slits (streets)
        for s in self.cfg.streets:
            ax.plot([s.center_x - s.width/2, s.center_x + s.width/2], [self.cfg.barrier_y, self.cfg.barrier_y],
                    color="royalblue", linewidth=6, solid_capstyle='butt')
            ax.text(s.center_x, self.cfg.barrier_y + 8.0, s.name, ha='center', va='bottom', fontsize=9, color='royalblue')

        # Screen (Maxwell Road)
        ax.plot([-self.cfg.screen_span/2, self.cfg.screen_span/2], [self.cfg.screen_y, self.cfg.screen_y],
                color="green", linewidth=2, label="Maxwell Road (screen)")

        ax.set_title("City electron interference: geometry")
        ax.set_xlabel("x (abstract units)")
        ax.set_ylabel("y (abstract units)")
        ax.legend(loc="upper right")
        ax.set_xlim(-self.cfg.screen_span/2, self.cfg.screen_span/2)
        ax.set_ylim(self.cfg.source_y - 40, self.cfg.screen_y + 60)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=160, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    def plot_intensity(self, screen_x: np.ndarray, intensity: np.ndarray, save_path: str | None = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(screen_x, intensity, color="purple")
        ax.set_xlabel("x along Maxwell Road (abstract units)")
        ax.set_ylabel("Normalized intensity |A|^2")
        ax.set_title("Interference pattern on Maxwell Road")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1.05)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=160, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Electron interference across city streets via path summation")
    parser.add_argument("--wavelength", type=float, default=None, help="Effective wavelength (same units as geometry)")
    parser.add_argument("--source-x", type=float, default=20.0, help="Source x offset (units)")
    parser.add_argument("--screen-span", type=float, default=None, help="Total x-span for sampling the screen")
    parser.add_argument("--screen-points", type=int, default=None, help="Number of points along the screen")
    parser.add_argument("--slit-resolution", type=float, default=None, help="Samples per unit width across each street slit")
    parser.add_argument("--no-1overr", action="store_true", help="Disable 1/r amplitude decay")
    parser.add_argument("--show", action="store_true", help="Show plots instead of saving only")
    args = parser.parse_args()

    cfg = CityInterferenceConfig()
    if args.wavelength is not None:
        cfg.wavelength = args.wavelength
    if args.screen_span is not None:
        cfg.screen_span = args.screen_span
    if args.screen_points is not None:
        cfg.screen_points = args.screen_points
    if args.slit_resolution is not None:
        cfg.points_per_meter_slit = args.slit_resolution
    if args.no_1overr:
        cfg.use_1_over_r_decay = False

    simulator = CityPathIntegralSimulator(cfg)

    screen_x, amp, intensity = simulator.simulate(source_x=args.source_x)

    outputs_dir = os.path.join("outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    geom_path = os.path.join(outputs_dir, "geometry.png")
    intensity_path = os.path.join(outputs_dir, "intensity.png")

    simulator.plot_geometry(source_x=args.source_x, save_path=None if args.show else geom_path)
    simulator.plot_intensity(screen_x, intensity, save_path=None if args.show else intensity_path)

    if not args.show:
        print(f"Saved: {geom_path}")
        print(f"Saved: {intensity_path}")


if __name__ == "__main__":
    main()
