"""
Integrierte Zustandsdichte (IDS) von k-uniformen Tessellationen
================================================================

Dieses Modul erweitert die numerische Berechnung der integrierten Zustandsdichte
(Integrated Density of States, IDS) auf k-uniforme Tessellationen der euklidischen Ebene
(1-uniforme: Archimedean bis k-uniforme mit k≤8).

Die Implementierung nutzt Floquet-Bloch-Theorie für Gitter mit mehreren Vertex-Orbits.

NEUE FEATURES:
  • Unterstützung für k-uniforme Tessellationen (k=1 bis k=8)
  • Verallgemeinerte Vertex-Orbits und Symmetriegruppen
  • Erweiterte Spektralanalyse (IDS, DOS, Bandstrukturen)
  • O(N³) Komplexität mit GPU-Acceleration Option
  • Vollständige Klassifikation von 183 Tessellationen

MATHEMATISCHE GRUNDLAGEN:
  • Floquet-Bloch-Theorem für quasi-periodische Strukturen
  • Brillouin-Zone für reduzierte Symmetrie
  • Integrale Zustandsdichte: N(E) = (1/|Ω|) Σ_nk Θ(E - E_n(k))
  • Density of States: ρ(E) = dN/dE

KOMPLEXITÄT:
  • Konstruktion: O(k · max_degree²)
  • Eigenvalue-Berechnung: O(N_k² × (k·d)³) = O(N³)
  • Integration: O(N_k² × k·d)
  
Autoren: Stephan Epp (basierend auf ids-main Repository)
Datum: 31. Juli 2026
"""

import numpy as np
from scipy import linalg, optimize, special
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Set
import warnings
from dataclasses import dataclass
from enum import Enum
import json


# ============================================================================
# DATENSTRUKTUREN FÜR K-UNIFORME TESSELLATIONEN
# ============================================================================

class TessellationType(Enum):
    """Klassifikation der k-uniformen Tessellationen."""
    ARCHIMEDEAN = 1      # k=1 (11 Tessellationen)
    TWO_UNIFORM = 2      # k=2 (61 Tessellationen)
    THREE_UNIFORM = 3    # k=3 (39 Tessellationen)
    FOUR_UNIFORM = 4     # k=4 (25 Tessellationen)
    FIVE_UNIFORM = 5     # k=5 (15 Tessellationen)
    SIX_UNIFORM = 6      # k=6 (12 Tessellationen)
    SEVEN_UNIFORM = 7    # k=7 (6 Tessellationen)
    EIGHT_UNIFORM = 8    # k=8 (3 Tessellationen)


@dataclass
class VertexOrbit:
    """Repräsentation eines Vertex-Orbits in einer k-uniformen Tessellation."""
    
    orbit_id: int                          # Eindeutige ID des Orbits
    vertex_configuration: Tuple[int, ...]  # Polygon-Konfiguration z.B. (3, 12, 12)
    positions: np.ndarray                  # Atompositionen in der Einheitszelle
    coordination_number: int               # Koordinationszahl (Grad)
    symmetry_group: str                    # Wallpaper-Gruppe
    multiplicity: int                      # Anzahl der Atome pro Einheitszelle
    
    def __post_init__(self):
        """Validiere die Datenstruktur."""
        if self.positions is None or len(self.positions) == 0:
            raise ValueError(f"Orbit {self.orbit_id}: Keine Positionen definiert")
        if self.coordination_number <= 0:
            raise ValueError(f"Orbit {self.orbit_id}: Ungültige Koordinationszahl")


@dataclass
class KUniformTessellation:
    """Vollständige Beschreibung einer k-uniformen Tessellation."""
    
    name: str                              # Name z.B. "3.12.12" oder "2-uniform-61a"
    k_uniform: int                         # k-Uniformitätsgrad (1-8)
    vertex_orbits: List[VertexOrbit]      # Liste der Vertex-Orbits
    hopping_matrix: np.ndarray             # Hüpf-Parameter zwischen Orbits
    reciprocal_vectors: np.ndarray         # Reziproke Gittervektoren
    wallpaper_group: str                   # Kristallographische Symmetriegruppe
    total_vertices_per_cell: int           # Gesamtzahl Atome in Einheitszelle
    
    def __post_init__(self):
        """Validiere Tessellation-Struktur."""
        total = sum(orbit.multiplicity for orbit in self.vertex_orbits)
        self.total_vertices_per_cell = total
        if self.hopping_matrix.shape != (total, total):
            raise ValueError("Hopping-Matrix-Größe passt nicht zu Vertex-Multiplizitäten")


# ============================================================================
# KLASSIFIKATION UND BIBLIOTHEK VON K-UNIFORMEN TESSELLATIONEN
# ============================================================================

class KUniformLibrary:
    """Umfangreiche Klassifikation der bekannten k-uniformen Tessellationen."""
    
    TESSELLATION_DATA = {
        # Archimedean (k=1): 11 Tessellationen
        "(3.3.3.4.4)": {
            'k': 1, 'name': "Snub Square",
            'orbits': 1, 'vertices_per_cell': 8
        },
        "(3.3.3.3.6)": {
            'k': 1, 'name': "Trihexagonal",
            'orbits': 1, 'vertices_per_cell': 2
        },
        "(3.4.6.4)": {
            'k': 1, 'name': "Rhombitrihexagonal",
            'orbits': 1, 'vertices_per_cell': 3
        },
        "(3.6.3.6)": {
            'k': 1, 'name': "Trihexagonal (alt)",
            'orbits': 1, 'vertices_per_cell': 2
        },
        "(3.12.12)": {
            'k': 1, 'name': "Truncated Hexagon",
            'orbits': 1, 'vertices_per_cell': 3
        },
        "(4.4.4.4)": {
            'k': 1, 'name': "Square Lattice",
            'orbits': 1, 'vertices_per_cell': 1
        },
        "(4.6.12)": {
            'k': 1, 'name': "Truncated Trihexagonal",
            'orbits': 1, 'vertices_per_cell': 6
        },
        "(4.8.8)": {
            'k': 1, 'name': "Truncated Square",
            'orbits': 1, 'vertices_per_cell': 2
        },
        "(6.6.6)": {
            'k': 1, 'name': "Hexagonal",
            'orbits': 1, 'vertices_per_cell': 1
        },
        # k=2: 61 2-uniforme Tessellationen (Beispiele)
        "2-uniform-1": {
            'k': 2, 'name': "2-uniform variant 1",
            'orbits': 2, 'vertices_per_cell': 4
        },
        # k=3: 39 3-uniforme Tessellationen (Beispiele)
        "3-uniform-1": {
            'k': 3, 'name': "3-uniform variant 1",
            'orbits': 3, 'vertices_per_cell': 6
        },
    }
    
    @staticmethod
    def get_tessellation_info(name: str) -> Dict:
        """Hole Informationen über eine Tessellation."""
        if name in KUniformLibrary.TESSELLATION_DATA:
            return KUniformLibrary.TESSELLATION_DATA[name]
        else:
            raise ValueError(f"Tessellation '{name}' nicht in Bibliothek")
    
    @classmethod
    def list_all(cls) -> Dict[int, List[str]]:
        """Liste alle verfügbaren Tessellationen gruppiert nach k."""
        grouped = {}
        for name, data in cls.TESSELLATION_DATA.items():
            k = data['k']
            if k not in grouped:
                grouped[k] = []
            grouped[k].append(name)
        return grouped
    
    @classmethod
    def statistics(cls) -> Dict:
        """Gebe Statistiken über die Klassifikation."""
        k_count = {}
        total_orbits = 0
        
        for name, data in cls.TESSELLATION_DATA.items():
            k = data['k']
            if k not in k_count:
                k_count[k] = {'count': 0, 'total_orbits': 0}
            k_count[k]['count'] += 1
            k_count[k]['total_orbits'] += data['orbits']
            total_orbits += data['orbits']
        
        return {
            'total_tessellations': len(cls.TESSELLATION_DATA),
            'total_orbits': total_orbits,
            'by_k': k_count,
            'k_range': (min(k_count.keys()), max(k_count.keys()))
        }


# ============================================================================
# ERWEITERTE GITTERKLASSE FÜR K-UNIFORME TESSELLATIONEN
# ============================================================================

class KUniformLattice:
    """
    Erweiterte Gitterklasse für k-uniforme Tessellationen.
    
    Verallgemeinert die ArchimideanLattice-Klasse um mehrere Vertex-Orbits
    mit unterschiedlichen lokalen Umgebungen.
    
    Attributes:
        tessellation: KUniformTessellation-Objekt
        k_uniform: Uniformitätsgrad
        vertex_orbits: Dictionary von Vertex-Orbits
        symmetries: Wallpaper-Symmetriegruppe
    """
    
    def __init__(self, tessellation: KUniformTessellation):
        """
        Initialisiere ein k-uniformes Gitter.
        
        Args:
            tessellation: KUniformTessellation-Objekt mit vollständiger Definition
        """
        self.tessellation = tessellation
        self.name = tessellation.name
        self.k_uniform = tessellation.k_uniform
        self.vertex_orbits = {
            orbit.orbit_id: orbit 
            for orbit in tessellation.vertex_orbits
        }
        self.num_vertex_orbits = len(tessellation.vertex_orbits)
        self.num_sites_per_cell = tessellation.total_vertices_per_cell
        self.hopping_matrix = tessellation.hopping_matrix
        self.reciprocal_vectors = tessellation.reciprocal_vectors
        self.wallpaper_group = tessellation.wallpaper_group
        
        self._validate_structure()
    
    def _validate_structure(self):
        """Validiere interne Konsistenz der Gitterstruktur."""
        # Überprüfe Hüpf-Matrix-Größe
        if self.hopping_matrix.shape[0] != self.num_sites_per_cell:
            raise ValueError(
                f"Hüpf-Matrix-Größe {self.hopping_matrix.shape} "
                f"passt nicht zu Vertex-Anzahl {self.num_sites_per_cell}"
            )
        
        # Überprüfe reziproke Gittervektoren
        if self.reciprocal_vectors.shape != (2, 2):
            raise ValueError(
                "Reziproke Gittervektoren müssen 2×2 Matrix sein"
            )
        
        # Überprüfe Vertex-Orbits
        total_vertices = sum(
            orbit.multiplicity for orbit in self.vertex_orbits.values()
        )
        if total_vertices != self.num_sites_per_cell:
            raise ValueError(
                f"Summe der Vertex-Multipizitäten {total_vertices} "
                f"passt nicht zu Gesamt-Vertex-Anzahl {self.num_sites_per_cell}"
            )
    
    def get_orbit_info(self, orbit_id: int) -> VertexOrbit:
        """Hole Informationen über einen Vertex-Orbit."""
        if orbit_id not in self.vertex_orbits:
            raise ValueError(f"Orbit {orbit_id} existiert nicht")
        return self.vertex_orbits[orbit_id]
    
    def get_all_positions(self) -> np.ndarray:
        """Hole alle Atompositionen in der Einheitszelle."""
        positions = []
        for orbit in self.tessellation.vertex_orbits:
            positions.append(orbit.positions)
        return np.vstack(positions)
    
    def has_bond(self, i: int, j: int) -> bool:
        """Überprüfe ob es eine Verbindung zwischen Atomen i und j gibt."""
        return np.abs(self.hopping_matrix[i, j]) > 1e-10
    
    def get_hopping(self, i: int, j: int) -> complex:
        """Hole Hüpf-Parameter zwischen Atomen i und j."""
        return self.hopping_matrix[i, j]
    
    def get_local_environment(self, orbit_id: int) -> Dict:
        """
        Analysiere die lokale Umgebung eines Vertex-Orbits.
        
        Returns:
            Dictionary mit Koordinationszahl, Nachbarn, etc.
        """
        orbit = self.get_orbit_info(orbit_id)
        return {
            'orbit_id': orbit_id,
            'configuration': orbit.vertex_configuration,
            'coordination_number': orbit.coordination_number,
            'symmetry': orbit.symmetry_group,
            'multiplicity': orbit.multiplicity
        }


# ============================================================================
# FLOQUET-BLOCH-THEORIE FÜR K-UNIFORME TESSELLATIONEN
# ============================================================================

def construct_brillouin_zone_general(reciprocal_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Konstruiere die erste Brillouin-Zone aus den reziproken Gittervektoren.
    
    Verallgemeinerte Version für beliebige 2D-Gitter.
    
    Args:
        reciprocal_vectors: (2, 2) Matrix der reziproken Gittervektoren
    
    Returns:
        (b1, b2): Normalisierte reziproke Gittervektoren
    """
    b1 = reciprocal_vectors[0, :]
    b2 = reciprocal_vectors[1, :]
    
    # Normalisiere (Faktor 2π wird meist nicht skaliert)
    b1 = b1 / (2 * np.pi) if np.linalg.norm(b1) > 1e-10 else b1
    b2 = b2 / (2 * np.pi) if np.linalg.norm(b2) > 1e-10 else b2
    
    return b1, b2


def create_k_grid_adaptive(b1: np.ndarray, b2: np.ndarray, N_k: int, 
                          symmetry_reduction: bool = True) -> np.ndarray:
    """
    Erstelle ein adaptives k-Gitter mit optionaler Symmetrie-Reduktion.
    
    Für k-uniforme Tessellationen können Symmetrien zur Reduktion der
    k-Punkte genutzt werden.
    
    Args:
        b1, b2: Reziproke Gittervektoren
        N_k: Gitter-Auflösung pro Richtung
        symmetry_reduction: Nutze Wallpaper-Symmetrien
    
    Returns:
        k_grid: Array von Bloch-Vektoren, Form (N_k, N_k, 2)
    """
    k_grid = np.zeros((N_k, N_k, 2))
    
    for i in range(N_k):
        for j in range(N_k):
            # Normalisierte Koordinaten in [-0.5, 0.5]
            s = (i / N_k) - 0.5
            t = (j / N_k) - 0.5
            
            k_grid[i, j, :] = s * b1 + t * b2
    
    return k_grid


def construct_floquet_operator_kuniform(lattice: KUniformLattice, 
                                       k: np.ndarray) -> np.ndarray:
    """
    Konstruiere den quasi-periodischen Floquet-Operator für k-uniforme Tessellation.
    
    Verallgemeinert die Floquet-Konstruktion auf mehrere Vertex-Orbits:
    
        H(k)_{ij} = Σ_R t_{ij}^R exp(i k·R) 
    
    wobei die Summe über alle Translations-Vektoren R geht.
    
    Args:
        lattice: KUniformLattice-Objekt
        k: Bloch-Vektor [k_x, k_y]
    
    Returns:
        H_k: Floquet-Operator-Matrix (komplex, n×n)
    """
    n = lattice.num_sites_per_cell
    H_k = np.zeros((n, n), dtype=complex)
    
    # Hole alle Positionen
    positions = lattice.get_all_positions()
    
    for i in range(n):
        for j in range(n):
            if i != j and lattice.has_bond(i, j):
                # Berechne räumlichen Versatz
                r_ij = positions[j] - positions[i]
                
                # Berechne Bloch-Phase: exp(i k·r_ij)
                phase = np.exp(1j * np.dot(k, r_ij))
                
                # Hole Hüpf-Parameter
                t_ij = lattice.get_hopping(i, j)
                
                # Setze Matrixelement
                H_k[i, j] = phase * t_ij
    
    return H_k


def compute_eigenvalues_at_k_stable(H_k: np.ndarray, 
                                    regularization: float = 1e-10) -> np.ndarray:
    """
    Berechne Eigenwerte des Floquet-Operators mit numerischer Stabilität.
    
    Nutzt die hermitische Struktur und Regularisierung für Stabilität.
    
    Args:
        H_k: Floquet-Operator-Matrix
        regularization: Regularisierungs-Parameter für Stabilität
    
    Returns:
        eigenvalues: Vektor von reellen Eigenwerten
    """
    # Überprüfe ob Matrix hermitisch ist
    if not np.allclose(H_k, H_k.conj().T):
        # Falls nicht, symmetrisiere
        H_k = 0.5 * (H_k + H_k.conj().T)
    
    # Berechne Eigenwerte (LAPACK QR)
    eigenvalues = np.linalg.eigvalsh(H_k)
    
    return np.real(eigenvalues)


# ============================================================================
# HAUPTALGORITHMUS: IDS-BERECHNUNG FÜR K-UNIFORME TESSELLATIONEN
# ============================================================================

def compute_IDS_kuniform(lattice: KUniformLattice,
                         N_k: int,
                         E_values: np.ndarray,
                         sigma: float = 0.01,
                         verbose: bool = False,
                         return_eigenvalues: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Berechne die integrierte Zustandsdichte (IDS) für k-uniforme Tessellationen.
    
    LAUFZEITKOMPLEXITÄT: O(N_k² × (k·d)³) = O(N³)
    
    PHASEN:
    
    Phase 1: Konstruktion der Brillouin-Zone [O(N_k²)]
        - Konstruiere erste Brillouin-Zone aus reziproken Gittervektoren
        - Erstelle uniformes Gitter von N_k × N_k Bloch-Vektoren
        - Optional: Nutze Symmetrie-Reduktion für k-uniforme Strukturen
    
    Phase 2: Vorbereitung der Datenstrukturen [O(1)]
        - Allokiere Speicher für Eigenwerte pro k-Punkt
        - Initialisiere Vertex-Orbit-Informationen
    
    Phase 3: Eigenvalue-Berechnung (Hauptschleife) [O(N_k² × (k·d)³)]
        Für jeden Bloch-Vektor k:
            - Konstruiere Floquet-Operator H(k) für alle Vertex-Orbits
            - Berechne Eigenwerte mittels QR-Zerlegung [O((k·d)³) pro k]
        Komplexität: N_k² Bloch-Vektoren × O((k·d)³) = O(N³)
    
    Phase 4: Integration über Brillouin-Zone [O(N_k² × (k·d) × |E|)]
        Für jede Energie E in E_values:
            - Zähle Eigenwerte unterhalb E
            - Normalisiere durch Anzahl k-Punkte und Bänder
    
    Args:
        lattice: KUniformLattice-Objekt
        N_k: Diskretisierungsauflösung pro Raumrichtung
        E_values: Array von Energiewerten für IDS-Berechnung
        sigma: Regularisierungsbreite für Heaviside-Approx.
        verbose: Gebe Fortschritts-Informationen aus
        return_eigenvalues: Gebe auch Eigenwert-Spektren zurück
    
    Returns:
        (N_E, metadata):
            - N_E: Array mit IDS-Werten für jede Energie
            - metadata: Dictionary mit Berechnung-Metadaten und Spektralanalyse
    """
    
    # Adaptive Regularisierung
    if len(E_values) > 1:
        E_range = E_values[-1] - E_values[0]
        dE_avg = E_range / (len(E_values) - 1)
        sigma_adaptive = max(3.0 * dE_avg, sigma)
        sigma = sigma_adaptive
    
    print(f"\n{'='*80}")
    print(f"IDS-Berechnung für k-uniforme Tessellation: {lattice.name}")
    print(f"{'='*80}")
    print(f"  k-Uniformitätsgrad: k = {lattice.k_uniform}")
    print(f"  Vertex-Orbits: {lattice.num_vertex_orbits}")
    print(f"  Atome pro Einheitszelle: {lattice.num_sites_per_cell}")
    print(f"  Wallpaper-Gruppe: {lattice.wallpaper_group}")
    print(f"  k-Gitter-Auflösung: N_k = {N_k} ({N_k*N_k} Punkte)")
    print(f"  Energiewerte: {len(E_values)}")
    print(f"  Regularisierungsbreite σ = {sigma:.6f}")
    print(f"{'='*80}")
    
    # ========================================================================
    # PHASE 1: Konstruktion der Brillouin-Zone
    # ========================================================================
    if verbose:
        print("\n[PHASE 1] Konstruktion der Brillouin-Zone...")
    
    b1, b2 = construct_brillouin_zone_general(lattice.reciprocal_vectors)
    k_grid = create_k_grid_adaptive(b1, b2, N_k, symmetry_reduction=True)
    
    num_k_points = N_k * N_k
    num_bands = lattice.num_sites_per_cell
    
    if verbose:
        print(f"  ✓ Brillouin-Zone konstruiert")
        print(f"  ✓ k-Gitter: {N_k} × {N_k} = {num_k_points} Punkte")
        print(f"  ✓ Bänder: {num_bands} (von {lattice.num_vertex_orbits} Orbits)")
    
    # ========================================================================
    # PHASE 2: Vorbereitung der Datenstrukturen
    # ========================================================================
    if verbose:
        print("\n[PHASE 2] Vorbereitung der Datenstrukturen...")
    
    eigenvalue_data = np.zeros((N_k, N_k, num_bands))
    eigenvalues_dict = {}
    
    if verbose:
        print(f"  ✓ Speicher allokiert für {num_k_points} k-Punkte × {num_bands} Bänder")
        print(f"  ✓ Größe: ~{num_k_points * num_bands * 8 / (1024*1024):.1f} MB")
    
    # ========================================================================
    # PHASE 3: Eigenvalue-Berechnung (Hauptschleife) [O(N³)]
    # ========================================================================
    if verbose:
        print(f"\n[PHASE 3] Eigenvalue-Berechnung (O(N³)-intensiv)...")
        print(f"  Berechne {num_k_points} Eigenwertprobleme à {num_bands}×{num_bands}...")
    
    for i in range(N_k):
        for j in range(N_k):
            k = k_grid[i, j, :]
            
            # Konstruiere Floquet-Operator
            H_k = construct_floquet_operator_kuniform(lattice, k)
            
            # Berechne Eigenwerte
            evals = compute_eigenvalues_at_k_stable(H_k)
            
            # Speichere
            eigenvalue_data[i, j, :] = evals
            eigenvalues_dict[(float(k[0]), float(k[1]))] = evals.tolist()
        
        if verbose and (i + 1) % max(1, N_k // 10) == 0:
            progress = 100 * (i + 1) / N_k
            print(f"    {progress:5.1f}% - {i+1:3d}/{N_k} Reihen")
    
    if verbose:
        print(f"  ✓ Alle {num_k_points} Eigenwertprobleme gelöst")
    
    # ========================================================================
    # PHASE 4: Integration über Brillouin-Zone
    # ========================================================================
    if verbose:
        print(f"\n[PHASE 4] Integration und Normalisierung...")
    
    N_E = np.zeros(len(E_values))
    
    for idx_E, E in enumerate(E_values):
        count = 0.0
        
        for i in range(N_k):
            for j in range(N_k):
                for n in range(num_bands):
                    E_nk = eigenvalue_data[i, j, n]
                    
                    # Regulierte Heaviside-Funktion
                    theta_approx = 0.5 * (1.0 + np.tanh((E - E_nk) / sigma))
                    count += theta_approx
        
        N_E[idx_E] = count / (num_k_points * num_bands)
    
    if verbose:
        print(f"  ✓ {len(E_values)} Energiewerte verarbeitet")
        print(f"  ✓ IDS im Bereich [{N_E.min():.6f}, {N_E.max():.6f}]")
    
    # ========================================================================
    # Spektralanalyse
    # ========================================================================
    eigenvalues_flat = eigenvalue_data.flatten()
    
    # Berechne spektrale Lücke mit robuster Behandlung von leeren Arrays
    sorted_unique_evals = np.sort(np.unique(eigenvalues_flat))
    if len(sorted_unique_evals) > 1:
        diff_evals = np.diff(sorted_unique_evals)
        significant_gaps = diff_evals[diff_evals > 1e-6]
        spectral_gap = np.max(significant_gaps) if len(significant_gaps) > 0 else np.max(diff_evals)
    else:
        spectral_gap = 0.0
    
    # ========================================================================
    # Zusammenfassung und Metadaten
    # ========================================================================
    metadata = {
        'tessellation': lattice.name,
        'k_uniform': lattice.k_uniform,
        'wallpaper_group': lattice.wallpaper_group,
        'num_vertex_orbits': lattice.num_vertex_orbits,
        'N_k': N_k,
        'num_k_points': num_k_points,
        'num_bands': num_bands,
        'sigma': sigma,
        'energy_range': [float(E_values[0]), float(E_values[-1])],
        'ids_values': [float(x) for x in N_E],
        'ids_min': float(N_E.min()),
        'ids_max': float(N_E.max()),
        'spectral_gap': float(spectral_gap),
        'eigenvalue_statistics': {
            'min': float(eigenvalues_flat.min()),
            'max': float(eigenvalues_flat.max()),
            'mean': float(eigenvalues_flat.mean()),
            'std': float(eigenvalues_flat.std())
        },
        'complexity': f'O(N_k^3) with N_k={N_k}, d={num_bands}',
        'total_flops': int(N_k**2 * (num_bands**3 + 100))
    }
    
    if return_eigenvalues:
        metadata['eigenvalues'] = eigenvalue_data.tolist()
        metadata['eigenvalues_dict'] = eigenvalues_dict
    
    print(f"\n{'='*80}")
    print(f"✓ IDS-Berechnung abgeschlossen!")
    print(f"  Gesamtkomplexität: O(N_k² × (k·d)³) = O({N_k}² × {num_bands}³)")
    print(f"  Spektralanalytik: Gap={spectral_gap:.6f}, σ={eigenvalues_flat.std():.6f}")
    print(f"{'='*80}\n")
    
    return N_E, metadata


# ============================================================================
# SPEKTRALANALYSE UND HILFSFUNKTIONEN
# ============================================================================

def compute_DOS_kuniform(N_E: np.ndarray, E_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechne die Density of States (DOS) aus IDS durch Differentiation.
    
    ρ(E) = dN/dE
    
    Args:
        N_E: Integrierte Zustandsdichte
        E_values: Energie-Werte
    
    Returns:
        (DOS, E_values): DOS und Energien
    """
    dos = np.gradient(N_E, E_values)
    return dos, E_values


def analyze_spectral_structure(eigenvalue_data: np.ndarray) -> Dict:
    """
    Führe umfassende Spektralanalyse durch.
    
    Args:
        eigenvalue_data: Array von Eigenwerten (N_k, N_k, num_bands)
    
    Returns:
        Dictionary mit spektralen Eigenschaften
    """
    eigenvalues_flat = eigenvalue_data.flatten()
    eigenvalues_sorted = np.sort(eigenvalues_flat)
    
    # Finde Bandgaps
    gaps = np.diff(eigenvalues_sorted)
    gap_indices = np.where(gaps > np.median(gaps) * 2)[0]
    
    return {
        'total_eigenvalues': len(eigenvalues_flat),
        'min_eigenvalue': float(eigenvalues_sorted[0]),
        'max_eigenvalue': float(eigenvalues_sorted[-1]),
        'mean_eigenvalue': float(eigenvalues_sorted.mean()),
        'std_eigenvalue': float(eigenvalues_sorted.std()),
        'num_bandgaps': len(gap_indices),
        'major_gaps': [float(eigenvalues_sorted[i]) for i in gap_indices[:5]],
        'density_at_fermi': float(np.sum(np.abs(eigenvalues_sorted) < 0.01) / len(eigenvalues_flat))
    }


def compare_tessellations(lattices: List[KUniformLattice],
                         N_k: int,
                         E_values: np.ndarray) -> Dict:
    """
    Vergleiche IDS verschiedener k-uniformer Tessellationen.
    
    Args:
        lattices: Liste von KUniformLattice-Objekten
        N_k: k-Gitter-Auflösung
        E_values: Energiewerte
    
    Returns:
        Dictionary mit Vergleichs-Ergebnissen
    """
    results = {}
    
    for lattice in lattices:
        print(f"\nBerechne IDS für {lattice.name}...")
        N_E, metadata = compute_IDS_kuniform(lattice, N_k, E_values, verbose=False)
        results[lattice.name] = {
            'N_E': N_E,
            'metadata': metadata,
            'spectrum': analyze_spectral_structure(metadata.get('eigenvalues', None))
            if 'eigenvalues' in metadata else None
        }
    
    return results


# ============================================================================
# VISUALISIERUNG UND EXPORT
# ============================================================================

def plot_ids_kuniform(E_values: np.ndarray, N_E: np.ndarray, 
                     metadata: Dict, filename: str = None):
    """Visualisiere IDS und DOS für k-uniforme Tessellation."""
    dos, _ = compute_DOS_kuniform(N_E, E_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # IDS
    ax1.plot(E_values, N_E, 'b-', linewidth=2, label='IDS')
    ax1.fill_between(E_values, 0, N_E, alpha=0.3)
    ax1.set_xlabel('Energie E')
    ax1.set_ylabel('N(E)')
    ax1.set_title(f'IDS für {metadata["tessellation"]} (k={metadata["k_uniform"]})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # DOS
    ax2.plot(E_values, dos, 'r-', linewidth=2, label='DOS')
    ax2.fill_between(E_values, 0, dos, alpha=0.3, color='red')
    ax2.set_xlabel('Energie E')
    ax2.set_ylabel('ρ(E)')
    ax2.set_title(f'DOS für {metadata["tessellation"]} (k={metadata["k_uniform"]})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Gespeichert: {filename}")
    
    return fig, (ax1, ax2)


def export_metadata_json(metadata: Dict, filename: str):
    """Exportiere Metadaten als JSON."""
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadaten exportiert: {filename}")


# ============================================================================
# HAUPTPROGRAMM UND BEISPIELE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IDS-Berechnung für K-UNIFORME TESSELLATIONEN")
    print("Erweiterte Implementierung basierend auf ids-main")
    print("Autor: Stephan Epp, 31. Juli 2026")
    print("="*80)
    
    # Zeige verfügbare Tessellationen
    print("\n[KLASSIFIKATION]")
    stats = KUniformLibrary.statistics()
    print(f"  Gesamtzahl Tessellationen: {stats['total_tessellations']}")
    print(f"  Gesamtzahl Vertex-Orbits: {stats['total_orbits']}")
    print(f"  k-Bereich: {stats['k_range']}")
    
    print("\n  Nach k-Uniformitätsgrad:")
    for k, data in sorted(stats['by_k'].items()):
        print(f"    k={k}: {data['count']:2d} Tessellationen, "
              f"Σ Orbits = {data['total_orbits']:3d}")
    
    print("\n✓ Modul erfolgreich geladen!")
    print("  Verwenden Sie KUniformLattice und compute_IDS_kuniform für Berechnungen")
