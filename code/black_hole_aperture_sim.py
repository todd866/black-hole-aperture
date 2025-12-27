"""
Black Hole Aperture Simulation

Demonstrates that observer-relative dimensional apertures produce
black hole phenomenology (time dilation, horizon effects) without
invoking GR explicitly.

Key insight: The same high-dimensional dynamics appear differently
to observers with different apertures. External observers see time
freeze at the horizon; infalling observers see nothing special.

This connects to LIGO: gravitational waves are perturbations in
the aperture structure. The ringdown waveform encodes how the
merged system's aperture stabilizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import welch
from dataclasses import dataclass
from typing import Tuple, List
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)


@dataclass
class ObserverState:
    """State variables for an observer."""
    k_eff: float  # Effective dimension (participation ratio)
    s_acc: float  # Accessible entropy (log det C)
    tau_rate: float  # Correlation accumulation rate
    q_cumulative: float  # Thermodynamic cost of erasure
    tau_accumulated: float  # Total accumulated proper time


class CoupledOscillatorSystem:
    """
    High-dimensional dynamical system: N coupled harmonic oscillators.

    This serves as the "underlying reality" that both observers watch.
    The dynamics are the same; only the apertures differ.
    """

    def __init__(self, n_oscillators: int = 50, coupling: float = 0.3,
                 damping: float = 0.01):
        self.n = n_oscillators
        self.coupling = coupling
        self.damping = damping

        # Initialize with random positions and velocities
        self.reset()

    def reset(self):
        """Reset to random initial conditions."""
        self.positions = np.random.randn(self.n) * 0.5
        self.velocities = np.random.randn(self.n) * 0.3

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """Compute derivatives for ODE integration."""
        x = state[:self.n]
        v = state[self.n:]

        # Spring forces (harmonic)
        forces = -x

        # Coupling to neighbors
        forces[:-1] += self.coupling * (x[1:] - x[:-1])
        forces[1:] += self.coupling * (x[:-1] - x[1:])

        # Damping
        forces -= self.damping * v

        return np.concatenate([v, forces])

    def step(self, dt: float = 0.02):
        """Advance system by one timestep."""
        state = np.concatenate([self.positions, self.velocities])
        t_span = [0, dt]
        result = odeint(self.derivatives, state, t_span)
        self.positions = result[-1, :self.n]
        self.velocities = result[-1, self.n:]

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current positions and velocities."""
        return self.positions.copy(), self.velocities.copy()


class Observer:
    """
    An observer with a specific aperture (access to degrees of freedom).

    The aperture determines which modes the observer can see.
    External observers lose access to high-frequency modes near the horizon.
    Infalling observers maintain full access.
    """

    def __init__(self, n_modes: int, observer_type: str = 'infalling'):
        self.n = n_modes
        self.observer_type = observer_type
        self.prev_s_acc = 0.0
        self.q_cumulative = 0.0
        self.tau_accumulated = 0.0

    def get_aperture_weights(self, radius: float) -> np.ndarray:
        """
        Compute aperture weights based on radius.

        For external observer: weights decrease for high-frequency modes
        as radius approaches 0 (horizon).

        For infalling observer: full access always.
        """
        if self.observer_type == 'infalling':
            return np.ones(self.n)

        # External observer: high-frequency modes suppressed near horizon
        mode_freqs = np.arange(1, self.n + 1) / self.n
        weights = np.power(radius, mode_freqs * 3)
        return weights

    def compute_k_eff(self, weights: np.ndarray) -> float:
        """Compute effective dimension via participation ratio."""
        total = np.sum(weights)
        total_sq = np.sum(weights ** 2)
        if total_sq < 1e-10:
            return 0.0
        return (total ** 2) / total_sq

    def compute_s_acc(self, positions: np.ndarray, velocities: np.ndarray,
                      weights: np.ndarray) -> float:
        """
        Compute accessible entropy: S_acc = (1/2) log det C

        Using weighted variance as diagonal approximation.
        """
        epsilon = 1e-10
        variances = (positions ** 2 + velocities ** 2) * weights + epsilon
        log_det = np.sum(np.log(variances))
        return 0.5 * log_det

    def compute_tau_rate(self, velocities: np.ndarray,
                         weights: np.ndarray) -> float:
        """
        Compute correlation accumulation rate (Fisher speed proxy).
        """
        weighted_v_sq = np.sum(weights * velocities ** 2)
        total_weight = np.sum(weights)
        if total_weight < 1e-10:
            return 0.0
        return np.sqrt(weighted_v_sq / total_weight)

    def observe(self, positions: np.ndarray, velocities: np.ndarray,
                radius: float, dt: float = 1.0) -> ObserverState:
        """
        Make an observation and compute all state variables.
        """
        weights = self.get_aperture_weights(radius)

        k_eff = self.compute_k_eff(weights)
        s_acc = self.compute_s_acc(positions, velocities, weights)
        tau_rate = self.compute_tau_rate(velocities, weights)

        # Thermodynamic cost: erasure when S_acc drops
        delta_s = max(0, self.prev_s_acc - s_acc)
        self.q_cumulative += delta_s
        self.prev_s_acc = s_acc

        # Accumulate proper time
        self.tau_accumulated += tau_rate * dt

        return ObserverState(
            k_eff=k_eff,
            s_acc=s_acc,
            tau_rate=tau_rate,
            q_cumulative=self.q_cumulative,
            tau_accumulated=self.tau_accumulated
        )


def run_simulation(n_steps: int = 1000, n_oscillators: int = 50,
                   radius_profile: str = 'static') -> dict:
    """
    Run the full simulation with both observers.

    Args:
        n_steps: Number of timesteps
        n_oscillators: Number of coupled oscillators
        radius_profile: 'static', 'infall', or 'merger'

    Returns:
        Dictionary with all time series data
    """
    system = CoupledOscillatorSystem(n_oscillators)
    external = Observer(n_oscillators, 'external')
    infalling = Observer(n_oscillators, 'infalling')

    # Storage
    data = {
        'time': [],
        'radius': [],
        'external': {'k_eff': [], 's_acc': [], 'tau_rate': [],
                     'q': [], 'tau': []},
        'infalling': {'k_eff': [], 's_acc': [], 'tau_rate': [],
                      'q': [], 'tau': []},
    }

    for t in range(n_steps):
        # Determine radius based on profile
        if radius_profile == 'static':
            radius = 0.5  # Fixed at mid-distance
        elif radius_profile == 'infall':
            # Gradual approach to horizon
            radius = max(0.01, 1.0 - t / n_steps)
        elif radius_profile == 'merger':
            # Simulate merger: rapid infall then ringdown
            if t < n_steps // 3:
                radius = 1.0 - 0.5 * (t / (n_steps // 3))
            elif t < 2 * n_steps // 3:
                # Rapid plunge
                phase = (t - n_steps // 3) / (n_steps // 3)
                radius = 0.5 - 0.45 * phase
            else:
                # Ringdown oscillations
                phase = (t - 2 * n_steps // 3) / (n_steps // 3)
                radius = 0.05 + 0.1 * np.exp(-3 * phase) * np.sin(20 * phase)
                radius = max(0.01, radius)
        else:
            radius = 0.5

        # Step the system
        system.step()
        pos, vel = system.get_state()

        # Both observers watch
        ext_state = external.observe(pos, vel, radius)
        inf_state = infalling.observe(pos, vel, 1.0)  # Infalling always at r=1

        # Store
        data['time'].append(t)
        data['radius'].append(radius)

        for key in ['k_eff', 's_acc', 'tau_rate', 'q', 'tau']:
            attr = key if key != 'q' else 'q_cumulative'
            attr = attr if key != 'tau' else 'tau_accumulated'
            data['external'][key].append(getattr(ext_state, attr))
            data['infalling'][key].append(getattr(inf_state, attr))

    # Convert to numpy
    for key in data:
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
        elif isinstance(data[key], dict):
            for subkey in data[key]:
                data[key][subkey] = np.array(data[key][subkey])

    return data


def generate_figure_1(data: dict, filename: str = '../figures/fig1_time_dilation.pdf'):
    """
    Figure 1: Time dilation demonstration.

    Shows accumulated proper time for both observers during infall.
    External observer's clock asymptotes; infalling continues linearly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Radius profile
    ax = axes[0, 0]
    ax.plot(data['time'], data['radius'], 'k-', linewidth=2)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Near-horizon')
    ax.set_xlabel('Wall time')
    ax.set_ylabel('Radius')
    ax.set_title('A. Infall trajectory')
    ax.legend()

    # Panel B: Effective dimension
    ax = axes[0, 1]
    ax.plot(data['time'], data['external']['k_eff'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['k_eff'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$k_{eff}$')
    ax.set_title('B. Effective dimension')
    ax.legend()

    # Panel C: Correlation rate (τ rate)
    ax = axes[1, 0]
    ax.plot(data['time'], data['external']['tau_rate'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['tau_rate'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$\\dot{\\tau}$ (correlation rate)')
    ax.set_title('C. Time flow rate')
    ax.legend()

    # Panel D: Accumulated proper time
    ax = axes[1, 1]
    ax.plot(data['time'], data['external']['tau'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['tau'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$\\tau$ (accumulated proper time)')
    ax.set_title('D. Proper time accumulation')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def generate_figure_2(data: dict, filename: str = '../figures/fig2_thermodynamics.pdf'):
    """
    Figure 2: Thermodynamic cost of aperture squeezing.

    Shows accessible entropy and erasure cost (Landauer).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Accessible entropy
    ax = axes[0]
    ax.plot(data['time'], data['external']['s_acc'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['s_acc'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$S_{acc}$ (accessible entropy)')
    ax.set_title('A. Accessible entropy')
    ax.legend()

    # Panel B: Cumulative erasure cost
    ax = axes[1]
    ax.plot(data['time'], data['external']['q'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['q'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$Q$ (cumulative erasure cost)')
    ax.set_title('B. Thermodynamic cost (Landauer)')
    ax.legend()

    # Panel C: τ vs Q (time-cost tradeoff)
    ax = axes[2]
    ax.plot(data['external']['q'], data['external']['tau'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['infalling']['q'], data['infalling']['tau'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Cumulative erasure cost $Q$')
    ax.set_ylabel('Accumulated proper time $\\tau$')
    ax.set_title('C. Time-cost tradeoff')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def generate_figure_3(filename: str = '../figures/fig3_ligo_connection.pdf'):
    """
    Figure 3: Connection to gravitational waves (LIGO).

    Shows how aperture dynamics during merger produce
    characteristic ringdown waveforms.
    """
    # Run merger simulation
    data = run_simulation(n_steps=1500, radius_profile='merger')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Merger trajectory
    ax = axes[0, 0]
    ax.plot(data['time'], data['radius'], 'k-', linewidth=2)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    ax.text(250, 0.9, 'Inspiral', ha='center')
    ax.text(750, 0.9, 'Merger', ha='center')
    ax.text(1250, 0.9, 'Ringdown', ha='center')
    ax.set_xlabel('Wall time')
    ax.set_ylabel('Effective radius')
    ax.set_title('A. Merger trajectory')

    # Panel B: k_eff during merger (aperture dynamics)
    ax = axes[0, 1]
    ax.plot(data['time'], data['external']['k_eff'], 'r-', linewidth=1.5)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$k_{eff}$ (external)')
    ax.set_title('B. Aperture collapse during merger')

    # Panel C: τ rate as "strain" proxy
    ax = axes[1, 0]
    # Compute derivative of τ_rate as proxy for GW strain
    tau_rate = data['external']['tau_rate']
    strain_proxy = np.gradient(tau_rate)
    ax.plot(data['time'], strain_proxy, 'purple', linewidth=1)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wall time')
    ax.set_ylabel('$d\\dot{\\tau}/dt$ (strain proxy)')
    ax.set_title('C. Aperture perturbation (GW proxy)')

    # Panel D: Power spectrum of ringdown
    ax = axes[1, 1]
    ringdown_start = 1000
    ringdown_signal = strain_proxy[ringdown_start:]
    if len(ringdown_signal) > 10:
        freqs, psd = welch(ringdown_signal, fs=1.0, nperseg=min(256, len(ringdown_signal)//2))
        ax.semilogy(freqs, psd, 'purple', linewidth=2)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power spectral density')
    ax.set_title('D. Ringdown spectrum')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def generate_figure_4(filename: str = '../figures/fig4_k_vs_radius.pdf'):
    """
    Figure 4: k_eff vs radius showing the dimensional collapse.

    Analogous to the Schwarzschild time dilation factor.
    """
    radii = np.linspace(0.01, 1.0, 100)
    n_oscillators = 50

    k_effs = []
    for r in radii:
        mode_freqs = np.arange(1, n_oscillators + 1) / n_oscillators
        weights = np.power(r, mode_freqs * 3)
        total = np.sum(weights)
        total_sq = np.sum(weights ** 2)
        k_eff = (total ** 2) / total_sq
        k_effs.append(k_eff)

    # Schwarzschild comparison
    # Time dilation factor: sqrt(1 - r_s/r), with r_s = 0.1 (horizon at r=0.1)
    r_s = 0.1
    schwarzschild = np.sqrt(np.maximum(0, 1 - r_s / radii))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(radii, np.array(k_effs) / max(k_effs), 'r-',
            linewidth=2, label='$k_{eff}(r) / k_{max}$ (aperture model)')
    ax.plot(radii, schwarzschild, 'k--',
            linewidth=2, label='$\\sqrt{1 - r_s/r}$ (Schwarzschild)')

    ax.axvline(x=r_s, color='gray', linestyle=':', alpha=0.7)
    ax.text(r_s + 0.02, 0.5, 'Horizon', rotation=90, va='center')

    ax.set_xlabel('Radius $r$')
    ax.set_ylabel('Normalized time dilation factor')
    ax.set_title('Dimensional collapse vs Schwarzschild time dilation')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


if __name__ == '__main__':
    print("Running black hole aperture simulations...")

    # Generate main infall simulation
    print("\n1. Running infall simulation...")
    data_infall = run_simulation(n_steps=1000, radius_profile='infall')

    print("\n2. Generating Figure 1: Time dilation...")
    generate_figure_1(data_infall)

    print("\n3. Generating Figure 2: Thermodynamics...")
    generate_figure_2(data_infall)

    print("\n4. Generating Figure 3: LIGO connection...")
    generate_figure_3()

    print("\n5. Generating Figure 4: k vs radius...")
    generate_figure_4()

    print("\nAll figures generated successfully!")
    print("\nKey results:")
    print(f"  - External observer τ accumulated: {data_infall['external']['tau'][-1]:.1f}")
    print(f"  - Infalling observer τ accumulated: {data_infall['infalling']['tau'][-1]:.1f}")
    print(f"  - Ratio (time dilation): {data_infall['infalling']['tau'][-1] / max(1, data_infall['external']['tau'][-1]):.2f}x")
