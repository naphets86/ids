"""
Integrierte Zustandsdichte (IDS) von Archimedean-Gittergraphen
==============================================================

Dieses Modul implementiert die numerische Berechnung der integrierten Zustandsdichte
(Integrated Density of States, IDS) für Archimedean-Tessellationen der euklidischen
Ebene mittels Floquet-Bloch-Theorie.

Hauptalgorithmus: O(N³) Komplexität basierend auf Eigenvalue-Berechnung
- Phase 1: Konstruktion der Brillouin-Zone [O(N_k²)]
- Phase 2: Vorbereitung der Datenstrukturen [O(1)]
- Phase 3: Eigenvalue-Berechnung für alle k-Punkte [O(N_k² × N_band³) = O(N³)]
- Phase 4: Integration und Normalisierung [O(N_k² × N_band)]

Autoren: Stephan Epp
Datum: 31. Juli 2026
"""

import numpy as np
from scipy import linalg, optimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings

# ============================================================================
# KLASSEN FÜR GITTERSTRUKTUREN
# ============================================================================

class ArchimideanLattice:
    """
    Basis-Klasse für Archimedean-Tessellationen der euklidischen Ebene.
    
    Attributes:
        name: Name der Tesselation (z.B. "(6,6,6)", "(4,8,8)")
        lattice_type: Gittertyp-Klassifikation
        positions: Atompositionen in der Einheitszelle
        hopping: Hüpf-Matrix-Elemente zwischen Atomen
        reciprocal_vectors: Reziproke Gittervektoren
    """
    
    def __init__(self, lattice_type: str):
        """
        Initialisiere ein Archimedean-Gitter.
        
        Args:
            lattice_type: Gittertyp, z.B. "(6,6,6)", "(4,8,8)", "(3,12,12)"
        """
        self.lattice_type = lattice_type
        self.positions = None
        self.hopping = None
        self.num_sites_per_cell = None
        self.reciprocal_vectors = None
        self.brillouin_zone = None
        
        self._setup_lattice()
    
    def _setup_lattice(self):
        """Konfiguriere Gitterstruktur basierend auf Gittertyp."""
        
        if self.lattice_type == "(6,6,6)":
            # Hexagonales Gitter (2 Atome pro Einheitszelle)
            self.num_sites_per_cell = 2
            a = 1.0  # Gitterkonstante
            
            self.positions = np.array([
                [0.0, 0.0],
                [a/2, a*np.sqrt(3)/2]
            ])
            
            # Reziproke Gittervektoren (für hexagonales Gitter)
            self.reciprocal_vectors = np.array([
                [2*np.pi/a, 0],
                [-np.pi/a, np.pi*np.sqrt(3)/a]
            ])
            
            # Hüpf-Parameter
            self.hopping = np.array([
                [0.0, -1.0],
                [-1.0, 0.0]
            ], dtype=complex)
        
        elif self.lattice_type == "(4,8,8)":
            # Truncated Square (2 Atome pro Einheitszelle)
            self.num_sites_per_cell = 2
            a = 1.0
            
            self.positions = np.array([
                [0.0, 0.0],
                [a/np.sqrt(2), a/np.sqrt(2)]
            ])
            
            self.reciprocal_vectors = np.array([
                [2*np.pi/a, 0],
                [0, 2*np.pi/a]
            ])
            
            self.hopping = np.array([
                [0.0, -0.8],
                [-0.8, 0.0]
            ], dtype=complex)
        
        elif self.lattice_type == "(3,12,12)":
            # Truncated Hexagon (3 Atome pro Einheitszelle)
            self.num_sites_per_cell = 3
            a = 1.0
            
            self.positions = np.array([
                [0.0, 0.0],
                [a/2, 0.0],
                [a/2, a*np.sqrt(3)/2]
            ])
            
            self.reciprocal_vectors = np.array([
                [2*np.pi/a, 0],
                [-np.pi/a, np.pi*np.sqrt(3)/a]
            ])
            
            self.hopping = np.array([
                [0.0, -1.0, -0.5],
                [-1.0, 0.0, -1.0],
                [-0.5, -1.0, 0.0]
            ], dtype=complex)
        
        else:
            raise ValueError(f"Unbekannter Gittertyp: {self.lattice_type}")
    
    def has_bond(self, i: int, j: int) -> bool:
        """Überprüfe ob es eine Verbindung zwischen Atomen i und j gibt."""
        return np.abs(self.hopping[i, j]) > 1e-10
    
    def get_hopping(self, i: int, j: int) -> complex:
        """Hole Hüpf-Parameter zwischen Atomen i und j."""
        return self.hopping[i, j]


# ============================================================================
# BRILLOUIN-ZONE UND K-GITTER
# ============================================================================

def construct_brillouin_zone(lattice: ArchimideanLattice) -> Tuple[np.ndarray, np.ndarray]:
    """
    Konstruiere die erste Brillouin-Zone aus den reziproken Gittervektoren.
    
    Args:
        lattice: Archimedean-Gitter-Objekt
    
    Returns:
        (b1, b2): Reziproke Gittervektoren der ersten Brillouin-Zone
    """
    b1, b2 = lattice.reciprocal_vectors[0], lattice.reciprocal_vectors[1]
    
    # Normalisiere
    b1 = b1 / (2 * np.pi)
    b2 = b2 / (2 * np.pi)
    
    return b1, b2


def create_k_grid(b1: np.ndarray, b2: np.ndarray, N_k: int) -> np.ndarray:
    """
    Erstelle ein Gitter von Bloch-Vektoren in der Brillouin-Zone.
    
    Args:
        b1, b2: Reziproke Gittervektoren
        N_k: Anzahl der Punkte pro Raumrichtung
    
    Returns:
        k_grid: Array von Bloch-Vektoren, Form (N_k, N_k, 2)
    """
    k_grid = np.zeros((N_k, N_k, 2))
    
    for i in range(N_k):
        for j in range(N_k):
            # Normalisierte Koordinaten: [-0.5, 0.5]
            s = (i / N_k) - 0.5
            t = (j / N_k) - 0.5
            
            k_grid[i, j, :] = s * b1 + t * b2
    
    return k_grid


# ============================================================================
# FLOQUET-OPERATOR UND EIGENVALUE-BERECHNUNG
# ============================================================================

def construct_floquet_operator(lattice: ArchimideanLattice, k: np.ndarray) -> np.ndarray:
    """
    Konstruiere den quasi-periodischen Floquet-Operator H(k).
    
    Der Operator hat die Form:
        H(k)_{ij} = exp(i k·r_ij) × t_ij
    
    wobei r_ij der Versatz zwischen Atomen i und j ist.
    
    Args:
        lattice: Archimedean-Gitter
        k: Bloch-Vektor [k_x, k_y]
    
    Returns:
        H_k: Floquet-Operator-Matrix (komplexe n×n Matrix)
    """
    n = lattice.num_sites_per_cell
    H_k = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            if i != j and lattice.has_bond(i, j):
                # Berechne räumlichen Versatz
                r_ij = lattice.positions[j] - lattice.positions[i]
                
                # Berechne Bloch-Phase: exp(i k·r_ij)
                phase = np.exp(1j * np.dot(k, r_ij))
                
                # Hole Hüpf-Parameter
                t_ij = lattice.get_hopping(i, j)
                
                # Setze Matrixelement
                H_k[i, j] = phase * t_ij
    
    return H_k


def compute_eigenvalues_at_k(H_k: np.ndarray) -> np.ndarray:
    """
    Berechne die Eigenwerte des Floquet-Operators mittels QR-Zerlegung.
    
    Dies ist die O(n³)-intensive Schleife des Algorithmus.
    
    Args:
        H_k: Floquet-Operator-Matrix
    
    Returns:
        eigenvalues: Vektor von Eigenwerten
    """
    # Verwende scipy's effiziente Eigenvalue-Berechnung
    eigenvalues = np.linalg.eigvalsh(H_k)
    return np.real(eigenvalues)


# ============================================================================
# HAUPTALGORITHMUS: IDS-BERECHNUNG [O(N³)]
# ============================================================================

def compute_IDS_floquet(lattice: ArchimideanLattice, 
                        N_k: int, 
                        E_values: np.ndarray, 
                        sigma: float = 0.01,
                        verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Berechne die integrierte Zustandsdichte (IDS) mittels Floquet-Bloch-Theorie.
    
    Laufzeitkomplexität: O(N_k² × N_band³) = O(N³)
    
    Die Berechnung erfolgt in 4 Phasen:
    
    Phase 1: Konstruktion der Brillouin-Zone [O(N_k²)]
        - Konstruiere die erste Brillouin-Zone
        - Erstelle ein uniformes Gitter von N_k × N_k Bloch-Vektoren
    
    Phase 2: Vorbereitung der Datenstrukturen [O(1)]
        - Allokiere Speicher für Eigenwerte und IDS
    
    Phase 3: Hauptschleife - Eigenvalue-Berechnung [O(N_k² × N_band³)]
        Für jeden Bloch-Vektor k:
            - Konstruiere Floquet-Operator H(k)
            - Berechne Eigenwerte mittels QR-Zerlegung [O(N_band³) pro k]
        Gesamtzeit: N_k² Bloch-Vektoren × O(N_band³) = O(N³)
    
    Phase 4: Integration über Brillouin-Zone [O(N_k² × N_band × m)]
        Für jede Energie E in E_values:
            - Zähle Eigenwerte unterhalb E
            - Normalisiere durch Anzahl k-Punkte und Bänder
    
    Args:
        lattice: Archimedean-Gitter-Objekt
        N_k: Diskretisierungsauflösung pro Raumrichtung (z.B. 50, 100, 500)
        E_values: Array von Energiewerten für IDS-Berechnung
        sigma: Regularisierungsbreite für Heaviside-Approx. (default: 0.01)
        verbose: Gebe Fortschritts-Informationen aus
    
    Returns:
        (N_E, metadata): 
            - N_E: Array mit IDS-Werten für jede Energie in E_values
            - metadata: Dictionary mit Berechnung-Metadaten
    """
    
    # Berechne adaptive Regularisierungsbreite basierend auf Energieauflösung
    if len(E_values) > 1:
        E_range = E_values[-1] - E_values[0]
        dE_avg = E_range / (len(E_values) - 1)
        # σ sollte etwa 2-3 mal die durchschnittliche Energieauflösung sein
        sigma_adaptive = 3.0 * dE_avg
        sigma = max(sigma, sigma_adaptive)
    
    print(f"Starte IDS-Berechnung für {lattice.lattice_type}-Gitter")
    print(f"  Diskretisierungsauflösung: N_k = {N_k} (insgesamt {N_k*N_k} k-Punkte)")
    print(f"  Energiewerte: {len(E_values)}")
    print(f"  Regularisierungsbreite σ = {sigma:.6f} (adaptiv berechnet)")
    
    # ========================================================================
    # PHASE 1: Konstruktion der Brillouin-Zone [O(N_k²)]
    # ========================================================================
    if verbose:
        print("\nPhase 1: Konstruktion der Brillouin-Zone...")
    
    b1, b2 = construct_brillouin_zone(lattice)
    k_grid = create_k_grid(b1, b2, N_k)  # Shape: (N_k, N_k, 2)
    
    num_k_points = N_k * N_k
    num_bands = lattice.num_sites_per_cell
    
    if verbose:
        print(f"  ✓ Brillouin-Zone konstruiert")
        print(f"  ✓ k-Gitter: {N_k} × {N_k} = {num_k_points} Punkte")
    
    # ========================================================================
    # PHASE 2: Vorbereitung der Datenstrukturen [O(1)]
    # ========================================================================
    if verbose:
        print("\nPhase 2: Vorbereitung der Datenstrukturen...")
    
    eigenvalues_dict = {}  # Dictionary: k -> [E_1(k), E_2(k), ...]
    eigenvalue_data = np.zeros((N_k, N_k, num_bands))
    
    if verbose:
        print(f"  ✓ Speicher allokiert für {num_k_points} k-Punkte × {num_bands} Bänder")
    
    # ========================================================================
    # PHASE 3: Hauptschleife - Eigenvalue-Berechnung [O(N_k² × N_band³)]
    # ========================================================================
    if verbose:
        print("\nPhase 3: Eigenvalue-Berechnung (Hauptschleife)...")
        print(f"  Dies ist der O(N³)-intensive Teil!")
    
    for i in range(N_k):
        for j in range(N_k):
            k = k_grid[i, j, :]  # Bloch-Vektor
            
            # Schritt 3a: Konstruiere Floquet-Operator H(k)
            H_k = construct_floquet_operator(lattice, k)
            
            # Schritt 3b: Berechne Eigenwerte [O(N_band³) Zeit pro k]
            evals = compute_eigenvalues_at_k(H_k)
            
            # Speichere Eigenwerte
            eigenvalue_data[i, j, :] = evals
            k_tuple = (k[0], k[1])
            eigenvalues_dict[k_tuple] = evals
        
        if verbose and (i + 1) % max(1, N_k // 10) == 0:
            print(f"    {100 * (i + 1) / N_k:.1f}% - {i + 1}/{N_k} Reihen berechnet")
    
    if verbose:
        print(f"  ✓ Alle {num_k_points} Eigenwertprobleme gelöst")
    
    # ========================================================================
    # PHASE 4: Integration über Brillouin-Zone [O(N_k² × N_band × m)]
    # ========================================================================
    if verbose:
        print("\nPhase 4: Integration und Normalisierung...")
    
    N_E = np.zeros(len(E_values))
    
    for idx_E, E in enumerate(E_values):
        count = 0.0
        
        # Iteriere über alle Eigenwerte
        for i in range(N_k):
            for j in range(N_k):
                for n in range(num_bands):
                    E_n_k = eigenvalue_data[i, j, n]
                    
                    # Regulierte Heaviside-Funktion (glatte Approximation)
                    # Θ_σ(x) = 1/2 * (1 + tanh(x/σ))
                    theta_approx = 0.5 * (1.0 + np.tanh((E - E_n_k) / sigma))
                    count += theta_approx
        
        # Normalisiere durch Anzahl k-Punkte und Bänder
        N_E[idx_E] = count / (num_k_points * num_bands)
    
    if verbose:
        print(f"  ✓ {len(E_values)} Energiewerte verarbeitet")
        print(f"  ✓ IDS-Werte im Bereich [{N_E.min():.4f}, {N_E.max():.4f}]")
    
    # ========================================================================
    # Zusammenfassung und Metadata
    # ========================================================================
    metadata = {
        'lattice_type': lattice.lattice_type,
        'N_k': N_k,
        'num_k_points': num_k_points,
        'num_bands': num_bands,
        'sigma': sigma,
        'eigenvalues': eigenvalue_data,
        'eigenvalue_dict': eigenvalues_dict,
        'complexity': f'O(N_k³) with N_k={N_k}',
        'total_flops': N_k**2 * (num_bands**3 + 100)  # Grobe Schätzung
    }
    
    print(f"\n✓ IDS-Berechnung abgeschlossen!")
    print(f"  Gesamte Rechenzeit: O(N_k² × N_band³) = O({N_k}² × {num_bands}³)")
    
    return N_E, metadata


# ============================================================================
# HILFSFUNKTIONEN FÜR ANALYSE UND VISUALISIERUNG
# ============================================================================

def compute_DOS(N_E: np.ndarray, E_values: np.ndarray, dE: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechne die Density of States (DOS) aus der IDS durch numerische Differentiation.
    
    ρ(E) = dN/dE
    
    Args:
        N_E: Integrierte Zustandsdichte
        E_values: Energie-Werte
        dE: Differential-Schrittweite
    
    Returns:
        (DOS, E_dos): DOS-Werte und korrespondierende Energien
    """
    dos = np.gradient(N_E, E_values)
    return dos, E_values


def compute_spectral_gap(eigenvalue_data: np.ndarray) -> float:
    """
    Berechne die spektrale Lücke (Bandgap) aus den Eigenwerten.
    
    Args:
        eigenvalue_data: Array von Eigenwerten, Form (N_k, N_k, num_bands)
    
    Returns:
        gap: Spektrale Lücke (Minimum zwischen Bändern)
    """
    sorted_evals = np.sort(eigenvalue_data.flatten())
    
    # Finde Lücke zwischen benachbarten Eigenwerten
    gaps = np.diff(sorted_evals)
    max_gap_idx = np.argmax(gaps)
    
    return gaps[max_gap_idx]
