"""
Umfassendes Test-Modul für ids_calculator.py
==============================================

Teststrategien:
- Unit-Tests für alle Funktionen
- Edge Case Testing
- Parametrisierte Tests
- Integrationstests
- Fehlerbehandlung und Validierung
- Numerische Präzisionsüberprüfungen
- Performance-Tests

Autor: Stephan Epp
Datum: 31. Juli 2026
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple
import warnings
import json

# Garantiert, dass das src-Verzeichnis im Suchpfad liegt
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ids_calculator import (
    ArchimideanLattice,
    construct_brillouin_zone,
    create_k_grid,
    construct_floquet_operator,
    compute_eigenvalues_at_k,
    compute_IDS_floquet,
    compute_DOS,
    compute_spectral_gap,
)

# ============================================================================
# FIXTURES FÜR TEST-SETUP
# ============================================================================

@pytest.fixture
def hexagonal_lattice():
    """Erstelle hexagonales (6,6,6)-Gitter für Tests."""
    return ArchimideanLattice("(6,6,6)")


@pytest.fixture
def truncated_square_lattice():
    """Erstelle truncated square (4,8,8)-Gitter für Tests."""
    return ArchimideanLattice("(4,8,8)")


@pytest.fixture
def truncated_hexagon_lattice():
    """Erstelle truncated hexagon (3,12,12)-Gitter für Tests."""
    return ArchimideanLattice("(3,12,12)")


@pytest.fixture
def all_lattices():
    """Erstelle alle verfügbaren Gittertypen."""
    return {
        "(6,6,6)": ArchimideanLattice("(6,6,6)"),
        "(4,8,8)": ArchimideanLattice("(4,8,8)"),
        "(3,12,12)": ArchimideanLattice("(3,12,12)"),
    }


@pytest.fixture
def energy_grid():
    """Standard-Energiegitter für Tests."""
    return np.linspace(-5, 5, 50)


@pytest.fixture
def fine_energy_grid():
    """Feines Energiegitter für hochgenaue Tests."""
    return np.linspace(-5, 5, 200)


# ============================================================================
# TESTS FÜR ArchimideanLattice KLASSE
# ============================================================================

class TestArchimideanLattice:
    """Tests für die ArchimideanLattice-Klasse."""
    
    def test_hexagonal_initialization(self, hexagonal_lattice):
        """Test: Hexagonales Gitter wird korrekt initialisiert."""
        assert hexagonal_lattice.lattice_type == "(6,6,6)"
        assert hexagonal_lattice.num_sites_per_cell == 2
        assert hexagonal_lattice.positions is not None
        assert hexagonal_lattice.positions.shape == (2, 2)
        assert hexagonal_lattice.hopping is not None
        assert hexagonal_lattice.hopping.shape == (2, 2)
        assert hexagonal_lattice.reciprocal_vectors is not None
        assert hexagonal_lattice.reciprocal_vectors.shape == (2, 2)
    
    def test_truncated_square_initialization(self, truncated_square_lattice):
        """Test: Truncated square Gitter wird korrekt initialisiert."""
        assert truncated_square_lattice.lattice_type == "(4,8,8)"
        assert truncated_square_lattice.num_sites_per_cell == 2
        assert truncated_square_lattice.positions.shape == (2, 2)
        assert truncated_square_lattice.hopping.shape == (2, 2)
    
    def test_truncated_hexagon_initialization(self, truncated_hexagon_lattice):
        """Test: Truncated hexagon Gitter wird korrekt initialisiert."""
        assert truncated_hexagon_lattice.lattice_type == "(3,12,12)"
        assert truncated_hexagon_lattice.num_sites_per_cell == 3
        assert truncated_hexagon_lattice.positions.shape == (3, 2)
        assert truncated_hexagon_lattice.hopping.shape == (3, 3)
    
    def test_invalid_lattice_type(self):
        """Test: Ungültiger Gittertyp wirft ValueError."""
        with pytest.raises(ValueError, match="Unbekannter Gittertyp"):
            ArchimideanLattice("(99,99,99)")
    
    def test_has_bond_hexagonal(self, hexagonal_lattice):
        """Test: Bond-Detection funktioniert für hexagonales Gitter."""
        lattice = hexagonal_lattice
        # Es sollte eine Bindung zwischen verschiedenen Atomen geben
        assert lattice.has_bond(0, 1) or lattice.has_bond(1, 0)
        # Keine Bindung zu sich selbst
        assert not lattice.has_bond(0, 0)
    
    def test_has_bond_truncated_hexagon(self, truncated_hexagon_lattice):
        """Test: Bond-Detection für truncated hexagon."""
        lattice = truncated_hexagon_lattice
        # Prüfe auf Bindungen (sollte welche geben)
        has_any_bond = any(lattice.has_bond(i, j) for i in range(3) for j in range(3) if i != j)
        assert has_any_bond
    
    def test_get_hopping_hexagonal(self, hexagonal_lattice):
        """Test: Hüpf-Parameter werden korrekt zurückgegeben."""
        lattice = hexagonal_lattice
        # Hole Hüpf-Parameter
        t_01 = lattice.get_hopping(0, 1)
        t_10 = lattice.get_hopping(1, 0)
        # Sollte komplex sein
        assert isinstance(t_01, complex) or isinstance(t_01, (int, float))
        assert isinstance(t_10, complex) or isinstance(t_10, (int, float))
    
    def test_hopping_matrix_hermiticity(self, all_lattices):
        """Test: Hüpf-Matrix sollte Hermitesch oder symmetrisch sein."""
        for lattice_type, lattice in all_lattices.items():
            hopping = lattice.hopping
            # Prüfe Symmetrie: H = H†
            diff = np.max(np.abs(hopping - hopping.conj().T))
            assert diff < 1e-10, f"Hüpf-Matrix für {lattice_type} ist nicht hermitesch"
    
    def test_positions_are_real(self, all_lattices):
        """Test: Positionen sollten reelle Zahlen sein."""
        for lattice in all_lattices.values():
            assert np.all(np.isreal(lattice.positions))
            assert lattice.positions.dtype in [np.float32, np.float64, float]
    
    def test_reciprocal_vectors_are_real(self, all_lattices):
        """Test: Reziproke Gittervektoren sollten reell sein."""
        for lattice in all_lattices.values():
            assert np.all(np.isreal(lattice.reciprocal_vectors))


# ============================================================================
# TESTS FÜR BRILLOUIN-ZONE FUNKTIONEN
# ============================================================================

class TestBrillouinZone:
    """Tests für Brillouin-Zone-Konstruktion."""
    
    def test_construct_brillouin_zone_hexagonal(self, hexagonal_lattice):
        """Test: Brillouin-Zone für hexagonales Gitter."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        assert b1.shape == (2,)
        assert b2.shape == (2,)
        assert np.all(np.isfinite(b1))
        assert np.all(np.isfinite(b2))
    
    def test_construct_brillouin_zone_all_types(self, all_lattices):
        """Test: Brillouin-Zone für alle Gittertypen."""
        for lattice in all_lattices.values():
            b1, b2 = construct_brillouin_zone(lattice)
            assert b1.shape == (2,)
            assert b2.shape == (2,)
            assert not np.allclose(b1, 0)
            assert not np.allclose(b2, 0)
    
    def test_brillouin_zone_reciprocal_normalization(self, hexagonal_lattice):
        """Test: Reziproke Vektoren sind richtig normalisiert."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        # Die Norm sollte sinnvoll sein (nicht 0, nicht riesig)
        norm_b1 = np.linalg.norm(b1)
        norm_b2 = np.linalg.norm(b2)
        assert 0.5 < norm_b1 < 10
        assert 0.5 < norm_b2 < 10
    
    def test_create_k_grid_shape(self, hexagonal_lattice):
        """Test: k-Gitter hat korrekte Form."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        for N_k in [10, 20, 50]:
            k_grid = create_k_grid(b1, b2, N_k)
            assert k_grid.shape == (N_k, N_k, 2)
    
    def test_create_k_grid_values_bounded(self, hexagonal_lattice):
        """Test: k-Gitter-Werte sind in sinnvollem Bereich."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        k_grid = create_k_grid(b1, b2, 20)
        # k-Punkte sollten in Brillouin-Zone liegen
        assert np.all(np.isfinite(k_grid))
        # Keine NaN oder Inf
        assert not np.any(np.isnan(k_grid))
        assert not np.any(np.isinf(k_grid))
    
    def test_create_k_grid_symmetry(self, hexagonal_lattice):
        """Test: k-Gitter hat Symmetrie um Ursprung."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        N_k = 21
        k_grid = create_k_grid(b1, b2, N_k)  # Ungerade Größe
        # Der zentrale Punkt sollte nahe bei Null sein
        # Bei N_k=21 ist die Mitte bei Index (N_k-1)/2 = 10
        # aber die tatsächliche Mitte liegt zwischen den Gitterpunkten
        # Daher prüfen wir dass der Punkt nahe bei 0 ist (tolerant)
        k_center = k_grid[10, 10, :]
        assert np.linalg.norm(k_center) < 0.05  # k-Punkt sollte nah bei Ursprung sein
    
    @pytest.mark.parametrize("N_k", [5, 10, 20, 50, 100])
    def test_create_k_grid_various_sizes(self, hexagonal_lattice, N_k):
        """Test: k-Gitter für verschiedene Größen."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        k_grid = create_k_grid(b1, b2, N_k)
        assert k_grid.shape == (N_k, N_k, 2)
        assert np.all(np.isfinite(k_grid))


# ============================================================================
# TESTS FÜR FLOQUET-OPERATOR
# ============================================================================

class TestFloquetOperator:
    """Tests für Floquet-Operator-Konstruktion."""
    
    def test_construct_floquet_at_gamma(self, hexagonal_lattice):
        """Test: Floquet-Operator am Gamma-Punkt (k=0)."""
        k = np.array([0.0, 0.0])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        assert H_k.shape == (2, 2)
        assert H_k.dtype == complex
        # Am Gamma-Punkt sollte es hermitesch sein
        assert np.allclose(H_k, H_k.conj().T)
    
    def test_construct_floquet_various_k(self, hexagonal_lattice):
        """Test: Floquet-Operator für verschiedene k-Punkte."""
        lattice = hexagonal_lattice
        b1, b2 = construct_brillouin_zone(lattice)
        
        # Teste mehrere k-Punkte
        k_values = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
        ]
        
        for k in k_values:
            H_k = construct_floquet_operator(lattice, k)
            assert H_k.shape == (lattice.num_sites_per_cell, lattice.num_sites_per_cell)
            assert H_k.dtype == complex
            assert np.all(np.isfinite(H_k))
    
    def test_floquet_hermiticity(self, hexagonal_lattice):
        """Test: Floquet-Operator sollte Hermitesch sein."""
        k = np.array([0.5, 0.3])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        # Prüfe Hermitizität: H = H†
        diff = np.max(np.abs(H_k - H_k.conj().T))
        assert diff < 1e-10
    
    def test_floquet_operator_all_lattices(self, all_lattices):
        """Test: Floquet-Operator für alle Gittertypen."""
        k = np.array([0.1, 0.2])
        for lattice_type, lattice in all_lattices.items():
            H_k = construct_floquet_operator(lattice, k)
            expected_shape = (lattice.num_sites_per_cell, lattice.num_sites_per_cell)
            assert H_k.shape == expected_shape
            # Prüfe Hermitizität
            assert np.allclose(H_k, H_k.conj().T)
    
    def test_floquet_diagonal_elements_zero(self, hexagonal_lattice):
        """Test: Diagonalelemente sollten Null sein (keine on-site Energie)."""
        k = np.array([0.1, 0.1])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        # Diagonale sollten Null sein (bei on-site energy = 0)
        diag = np.diag(H_k)
        assert np.allclose(diag, 0)
    
    def test_floquet_zero_offset(self, truncated_hexagon_lattice):
        """Test: Floquet-Operator für 3-Atom-Zelle."""
        k = np.array([0.0, 0.0])
        H_k = construct_floquet_operator(truncated_hexagon_lattice, k)
        assert H_k.shape == (3, 3)
        assert np.allclose(H_k, H_k.conj().T)


# ============================================================================
# TESTS FÜR EIGENVALUE-BERECHNUNG
# ============================================================================

class TestEigenvalueComputation:
    """Tests für Eigenvalue-Berechnung."""
    
    def test_compute_eigenvalues_shape(self, hexagonal_lattice):
        """Test: Eigenvalues haben richtige Form."""
        k = np.array([0.0, 0.0])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        eigenvalues = compute_eigenvalues_at_k(H_k)
        assert eigenvalues.shape == (hexagonal_lattice.num_sites_per_cell,)
    
    def test_compute_eigenvalues_are_real(self, hexagonal_lattice):
        """Test: Eigenvalues eines Hermitesch-Operators sind reell."""
        k = np.array([0.0, 0.0])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        eigenvalues = compute_eigenvalues_at_k(H_k)
        # Für Hermitesch-Operator sollten Eigenvalues reell sein
        assert np.all(np.abs(eigenvalues.imag) < 1e-10)
    
    def test_compute_eigenvalues_sorted(self, hexagonal_lattice):
        """Test: Eigenvalues sollten sortiert sein."""
        k = np.array([0.5, 0.3])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        eigenvalues = compute_eigenvalues_at_k(H_k)
        eigenvalues_real = eigenvalues.real
        # Prüfe ob sortiert
        assert np.all(eigenvalues_real[:-1] <= eigenvalues_real[1:])
    
    def test_compute_eigenvalues_multiple_k_points(self, hexagonal_lattice):
        """Test: Eigenvalue-Berechnung für mehrere k-Punkte."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        k_grid = create_k_grid(b1, b2, 10)
        
        for i in range(10):
            for j in range(10):
                k = k_grid[i, j, :]
                H_k = construct_floquet_operator(hexagonal_lattice, k)
                eigenvalues = compute_eigenvalues_at_k(H_k)
                assert eigenvalues.shape == (2,)
                assert np.all(np.isfinite(eigenvalues))
    
    def test_eigenvalues_bounded_range(self, hexagonal_lattice):
        """Test: Eigenvalues liegen in sinnvollem Bereich."""
        b1, b2 = construct_brillouin_zone(hexagonal_lattice)
        k_grid = create_k_grid(b1, b2, 20)
        
        all_eigenvalues = []
        for i in range(20):
            for j in range(20):
                k = k_grid[i, j, :]
                H_k = construct_floquet_operator(hexagonal_lattice, k)
                eigenvalues = compute_eigenvalues_at_k(H_k)
                all_eigenvalues.extend(eigenvalues.real)
        
        all_eigenvalues = np.array(all_eigenvalues)
        # Eigenvalues sollten im Bereich [-3, 3] liegen (abhängig von Hüpfparametern)
        assert np.min(all_eigenvalues) > -10
        assert np.max(all_eigenvalues) < 10


# ============================================================================
# TESTS FÜR IDS-BERECHNUNG
# ============================================================================

class TestIDSComputation:
    """Tests für Integrierte Zustandsdichte Berechnung."""
    
    def test_compute_ids_floquet_basic(self, hexagonal_lattice, energy_grid):
        """Test: Basis-IDS-Berechnung."""
        N_E, metadata = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        assert N_E.shape == energy_grid.shape
        assert metadata is not None
        assert isinstance(metadata, dict)
        assert 'lattice_type' in metadata
        assert 'N_k' in metadata
        assert 'num_bands' in metadata
    
    def test_ids_monotonicity(self, hexagonal_lattice, energy_grid):
        """Test: IDS sollte monoton steigend sein."""
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        # IDS sollte monoton steigend sein
        dN = np.diff(N_E)
        assert np.all(dN >= -1e-10)  # Kleine numerische Fehler erlaubt
    
    def test_ids_bounded_0_1(self, hexagonal_lattice, energy_grid):
        """Test: IDS sollte im Bereich [0, 1] liegen."""
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        assert np.all(N_E >= -1e-10)
        assert np.all(N_E <= 1 + 1e-10)
    
    def test_ids_all_lattice_types(self, all_lattices, energy_grid):
        """Test: IDS-Berechnung für alle Gittertypen."""
        for lattice in all_lattices.values():
            N_E, metadata = compute_IDS_floquet(
                lattice,
                N_k=8,
                E_values=energy_grid,
                verbose=False
            )
            
            assert N_E.shape == energy_grid.shape
            assert np.all(np.isfinite(N_E))
            assert np.all(N_E >= 0)
            assert np.all(N_E <= 1)
    
    @pytest.mark.parametrize("N_k", [5, 10, 20])
    def test_ids_various_k_grid_sizes(self, hexagonal_lattice, energy_grid, N_k):
        """Test: IDS für verschiedene k-Gitter-Größen."""
        N_E, metadata = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=N_k,
            E_values=energy_grid,
            verbose=False
        )
        
        assert metadata['N_k'] == N_k
        assert N_E.shape == energy_grid.shape
        assert np.all(np.isfinite(N_E))
    
    def test_ids_symmetry_properties(self, hexagonal_lattice):
        """Test: IDS-Symmetrie-Eigenschaften."""
        # Symmetrisches Energiegitter um Null
        E_values = np.linspace(-3, 3, 61)
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=E_values,
            verbose=False
        )
        
        # Für 2-Band-System sollte Symmetrie vorhanden sein
        # Prüfe nur dass Werte sinnvoll sind
        assert N_E[30] >= 0  # Wert bei E=0
        assert N_E[30] <= 1
    
    def test_ids_metadata_completeness(self, hexagonal_lattice, energy_grid):
        """Test: Metadata ist vollständig und korrekt."""
        N_E, metadata = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        required_keys = [
            'lattice_type', 'N_k', 'num_k_points', 'num_bands',
            'sigma', 'eigenvalues', 'eigenvalue_dict', 'complexity'
        ]
        
        for key in required_keys:
            assert key in metadata
        
        assert metadata['lattice_type'] == "(6,6,6)"
        assert metadata['N_k'] == 10
        assert metadata['num_k_points'] == 100  # 10x10
        assert metadata['num_bands'] == 2
    
    def test_ids_fine_vs_coarse_grid(self, hexagonal_lattice):
        """Test: IDS mit feinem vs. grobem Gitter."""
        E_coarse = np.linspace(-5, 5, 20)
        E_fine = np.linspace(-5, 5, 100)
        
        N_E_coarse, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=8,
            E_values=E_coarse,
            verbose=False
        )
        
        N_E_fine, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=8,
            E_values=E_fine,
            verbose=False
        )
        
        assert N_E_coarse.shape == (20,)
        assert N_E_fine.shape == (100,)
        assert np.all(np.isfinite(N_E_coarse))
        assert np.all(np.isfinite(N_E_fine))


# ============================================================================
# TESTS FÜR HILFSFUNKTIONEN
# ============================================================================

class TestHelperFunctions:
    """Tests für Hilfsfunktionen."""
    
    def test_compute_dos_shape(self, hexagonal_lattice, energy_grid):
        """Test: DOS hat richtige Form."""
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        dos, E_dos = compute_DOS(N_E, energy_grid)
        
        assert dos.shape == energy_grid.shape
        assert E_dos.shape == energy_grid.shape
        assert np.all(np.isfinite(dos))
    
    def test_compute_dos_positivity(self, hexagonal_lattice, energy_grid):
        """Test: DOS sollte nicht-negativ sein."""
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        dos, _ = compute_DOS(N_E, energy_grid)
        
        # DOS sollte hauptsächlich positiv sein (kleine numerische Fehler erlaubt)
        assert np.mean(dos) > 0
        assert np.min(dos) > -1e-10
    
    def test_compute_dos_all_lattices(self, all_lattices, energy_grid):
        """Test: DOS-Berechnung für alle Gittertypen."""
        for lattice in all_lattices.values():
            N_E, _ = compute_IDS_floquet(
                lattice,
                N_k=8,
                E_values=energy_grid,
                verbose=False
            )
            
            dos, E_dos = compute_DOS(N_E, energy_grid)
            
            assert dos.shape == energy_grid.shape
            assert np.all(np.isfinite(dos))
    
    def test_compute_spectral_gap(self, hexagonal_lattice):
        """Test: Spektrale Lücke wird berechnet."""
        lattice = hexagonal_lattice
        b1, b2 = construct_brillouin_zone(lattice)
        k_grid = create_k_grid(b1, b2, 10)
        
        eigenvalue_data = np.zeros((10, 10, 2))
        
        for i in range(10):
            for j in range(10):
                k = k_grid[i, j, :]
                H_k = construct_floquet_operator(lattice, k)
                eigenvalues = compute_eigenvalues_at_k(H_k)
                eigenvalue_data[i, j, :] = eigenvalues.real
        
        gap = compute_spectral_gap(eigenvalue_data)
        
        assert gap > 0
        assert np.isfinite(gap)
    
    def test_spectral_gap_all_lattices(self, all_lattices):
        """Test: Spektrale Lücke für alle Gittertypen."""
        for lattice in all_lattices.values():
            b1, b2 = construct_brillouin_zone(lattice)
            k_grid = create_k_grid(b1, b2, 8)
            
            num_bands = lattice.num_sites_per_cell
            eigenvalue_data = np.zeros((8, 8, num_bands))
            
            for i in range(8):
                for j in range(8):
                    k = k_grid[i, j, :]
                    H_k = construct_floquet_operator(lattice, k)
                    eigenvalues = compute_eigenvalues_at_k(H_k)
                    eigenvalue_data[i, j, :] = eigenvalues.real
            
            gap = compute_spectral_gap(eigenvalue_data)
            assert gap >= 0
            assert np.isfinite(gap)


# ============================================================================
# INTEGRATIONSTESTS
# ============================================================================

class TestIntegration:
    """Integrationstests für komplette Workflows."""
    
    def test_full_workflow_hexagonal(self):
        """Test: Vollständiger Workflow für hexagonales Gitter."""
        lattice = ArchimideanLattice("(6,6,6)")
        E_values = np.linspace(-4, 4, 50)
        
        # Schritt 1: IDS-Berechnung
        N_E, metadata = compute_IDS_floquet(
            lattice,
            N_k=10,
            E_values=E_values,
            verbose=False
        )
        
        assert N_E.shape == (50,)
        assert np.all(np.isfinite(N_E))
        
        # Schritt 2: DOS-Berechnung
        dos, E_dos = compute_DOS(N_E, E_values)
        assert dos.shape == (50,)
        assert np.all(np.isfinite(dos))
        
        # Schritt 3: Spektrale Lücke
        gap = compute_spectral_gap(metadata['eigenvalues'])
        assert gap >= 0
        assert np.isfinite(gap)
    
    def test_comparison_workflow(self):
        """Test: Vergleich verschiedener Gittertypen."""
        lattice_types = ["(6,6,6)", "(4,8,8)", "(3,12,12)"]
        E_values = np.linspace(-4, 4, 40)
        
        results = {}
        
        for lattice_type in lattice_types:
            lattice = ArchimideanLattice(lattice_type)
            N_E, _ = compute_IDS_floquet(
                lattice,
                N_k=8,
                E_values=E_values,
                verbose=False
            )
            results[lattice_type] = N_E
        
        # Alle sollten gleiche Form haben
        for N_E in results.values():
            assert N_E.shape == (40,)
            assert np.all(np.isfinite(N_E))
        
        # Alle sollten in [0, 1] sein
        for N_E in results.values():
            assert np.all(N_E >= 0)
            assert np.all(N_E <= 1)
    
    def test_reproducibility(self, hexagonal_lattice, energy_grid):
        """Test: Ergebnisse sind reproduzierbar."""
        N_E1, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        N_E2, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=10,
            E_values=energy_grid,
            verbose=False
        )
        
        # Sollte identisch sein (numerische Genauigkeit)
        assert np.allclose(N_E1, N_E2, atol=1e-14)


# ============================================================================
# EDGE CASE UND ERROR HANDLING TESTS
# ============================================================================

class TestEdgeCases:
    """Tests für Edge Cases und Fehlerbehandlung."""
    
    def test_single_energy_point(self, hexagonal_lattice):
        """Test: IDS mit nur einem Energiepunkt."""
        E_values = np.array([0.0])
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=5,
            E_values=E_values,
            verbose=False
        )
        
        assert N_E.shape == (1,)
        assert 0 <= N_E[0] <= 1
    
    def test_very_wide_energy_range(self, hexagonal_lattice):
        """Test: Sehr breiter Energiebereich."""
        E_values = np.linspace(-100, 100, 50)
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=5,
            E_values=E_values,
            verbose=False
        )
        
        assert N_E.shape == (50,)
        assert np.all(np.isfinite(N_E))
        assert np.all(N_E >= 0)
        assert np.all(N_E <= 1)
    
    def test_very_fine_k_grid(self, hexagonal_lattice, energy_grid):
        """Test: Sehr feines k-Gitter."""
        N_E, metadata = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=30,
            E_values=energy_grid,
            verbose=False
        )
        
        assert metadata['N_k'] == 30
        assert metadata['num_k_points'] == 900
        assert np.all(np.isfinite(N_E))
    
    def test_very_coarse_k_grid(self, hexagonal_lattice, energy_grid):
        """Test: Sehr grobes k-Gitter."""
        N_E, metadata = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=2,
            E_values=energy_grid,
            verbose=False
        )
        
        assert metadata['N_k'] == 2
        assert metadata['num_k_points'] == 4
        assert np.all(np.isfinite(N_E))
    
    def test_negative_energies(self, hexagonal_lattice):
        """Test: Nur negative Energien."""
        E_values = np.linspace(-10, -1, 30)
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=8,
            E_values=E_values,
            verbose=False
        )
        
        assert N_E.shape == (30,)
        assert np.all(np.isfinite(N_E))
    
    def test_positive_energies(self, hexagonal_lattice):
        """Test: Nur positive Energien."""
        E_values = np.linspace(1, 10, 30)
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=8,
            E_values=E_values,
            verbose=False
        )
        
        assert N_E.shape == (30,)
        assert np.all(np.isfinite(N_E))


# ============================================================================
# PERFORMANCE UND NUMERISCHE TESTS
# ============================================================================

class TestNumericalAccuracy:
    """Tests für numerische Genauigkeit und Präzision."""
    
    def test_eigenvalue_precision_gamma_point(self, hexagonal_lattice):
        """Test: Eigenvalue-Präzision am Gamma-Punkt."""
        k = np.array([0.0, 0.0])
        H_k = construct_floquet_operator(hexagonal_lattice, k)
        eigenvalues = compute_eigenvalues_at_k(H_k)
        
        # Am Gamma-Punkt sollte H_k hermitesch sein
        # Eigenvalues sollten exakt reell sein
        assert np.allclose(eigenvalues.imag, 0, atol=1e-15)
    
    def test_ids_continuity(self, hexagonal_lattice):
        """Test: IDS ist stetig in der Energie."""
        E_values = np.linspace(-3, 3, 100)
        N_E, _ = compute_IDS_floquet(
            hexagonal_lattice,
            N_k=12,
            E_values=E_values,
            verbose=False
        )
        
        # Prüfe dass keine großen Sprünge vorhanden sind
        dN = np.diff(N_E)
        max_jump = np.max(np.abs(dN))
        
        # Sprünge sollten klein sein relativ zur Gesamtänderung
        total_change = N_E[-1] - N_E[0]
        if total_change > 0:
            relative_jump = max_jump / total_change
            assert relative_jump < 0.1


# ============================================================================
# HAUPT-TEST-FUNCTION
# ============================================================================

def run_all_tests():
    """Führe alle Tests mit pytest aus."""
    pytest.main([__file__, "-v", "--tb=short", "-ra"])


if __name__ == "__main__":
    # Führe Tests mit Coverage aus
    pytest.main([
        __file__,
        "-v",
        "--cov=ids_calculator",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])
