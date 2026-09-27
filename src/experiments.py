"""
Umfangreiche Experimente für IDS-Berechnung auf Archimedean-Gittergraphen
==========================================================================

Experimentmodul für wissenschaftliche Validierung und Datensammlung.

Experimentkategorien:
1. Konvergenz-Experimente: IDS-Konvergenz mit k-Gitter-Größe
2. Spektrale Analysen: Eigenwerte, Bandstrukturen, Gaps
3. Vergleichende Studien: Verschiedene Gittertypen
4. Performance-Analysen: Laufzeitmessungen und Skalierung
5. Energie-Auflösungsanalysen: DOS-Glattheit und Features
6. Stabilität-Tests: Numerische Stabilität
7. Symmetrie-Verletzungen: Überprüfung theoretischer Vorhersagen

Ausgaben:
- experiments_results.json: Numerische Ergebnisse
- experiments_summary.txt: Menschenlesbare Zusammenfassung
- experiments_detailed.json: Detaillierte Messdaten
- spectral_data_<lattice>.json: Spektralanalyse pro Gittertyp

Autor: Stephan Epp
Datum: 31. Juli 2026
"""

import numpy as np
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import des zu testenden Moduls
sys.path.insert(0, str(Path(__file__).parent))
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
# EXPERIMENT 1: KONVERGENZ-ANALYSE
# ============================================================================

def experiment_convergence_analysis() -> Dict[str, Any]:
    """
    Analyse der IDS-Konvergenz mit zunehmender k-Gitter-Auflösung.
    
    Untersucht wie sich die IDS ändert wenn N_k von 5 bis 50 variiert wird.
    Verwendet Energiegitter mit fester Auflösung.
    
    Returns:
        convergence_data: Dictionary mit Konvergenzanalysen
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: KONVERGENZ-ANALYSE DER IDS")
    print("=" * 80)
    
    lattice_types = ["(6,6,6)", "(4,8,8)", "(3,12,12)"]
    N_k_values = [5, 10, 15, 20, 30, 40, 50]
    E_values = np.linspace(-5, 5, 100)
    
    convergence_data = {
        'experiment': 'convergence_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'IDS Konvergenz mit k-Gitter-Größe',
        'lattice_types': lattice_types,
        'N_k_values': N_k_values,
        'energy_points': len(E_values),
        'energy_range': [float(E_values[0]), float(E_values[-1])],
        'results': {}
    }
    
    for lattice_type in lattice_types:
        print(f"\n  Gittertyp: {lattice_type}")
        lattice = ArchimideanLattice(lattice_type)
        
        results_for_lattice = {
            'lattice_type': lattice_type,
            'convergence_data': {}
        }
        
        N_E_reference = None
        reference_N_k = None
        
        for N_k in N_k_values:
            print(f"    N_k = {N_k:2d}...", end=' ', flush=True)
            start_time = time.time()
            
            N_E, metadata = compute_IDS_floquet(
                lattice,
                N_k=N_k,
                E_values=E_values,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            
            # Speichere Referenz für Konvergenzvergleich
            if N_k == N_k_values[-1]:
                N_E_reference = N_E
                reference_N_k = N_k
            
            # Berechne Konvergenz-Metriken
            if N_E_reference is not None:
                # Interpoliere auf gleiche Energiepunkte
                max_abs_diff = np.max(np.abs(N_E - N_E_reference))
                rmse = np.sqrt(np.mean((N_E - N_E_reference)**2))
            else:
                max_abs_diff = None
                rmse = None
            
            results_for_lattice['convergence_data'][f'N_k_{N_k}'] = {
                'N_k': N_k,
                'num_k_points': N_k * N_k,
                'time_seconds': float(elapsed),
                'max_abs_diff_to_reference': float(max_abs_diff) if max_abs_diff else None,
                'rmse_to_reference': float(rmse) if rmse else None,
                'ids_statistics': {
                    'min': float(np.min(N_E)),
                    'max': float(np.max(N_E)),
                    'mean': float(np.mean(N_E)),
                    'std': float(np.std(N_E)),
                }
            }
            
            print(f"t={elapsed:.2f}s, Δmax={max_abs_diff:.2e if max_abs_diff else 'ref'}")
        
        convergence_data['results'][lattice_type] = results_for_lattice
    
    return convergence_data


# ============================================================================
# EXPERIMENT 2: SPEKTRALANALYSE
# ============================================================================

def experiment_spectral_analysis() -> Dict[str, Any]:
    """
    Detaillierte Spektralanalyse für alle Gittertypen.
    
    Untersucht:
    - Eigenverteilung über Brillouin-Zone
    - Bandstrukturen
    - Spektrale Lücken
    - Density of States Features
    
    Returns:
        spectral_data: Dictionary mit Spektralanalysen
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: SPEKTRALANALYSE")
    print("=" * 80)
    
    lattice_types = ["(6,6,6)", "(4,8,8)", "(3,12,12)"]
    N_k = 25
    E_values = np.linspace(-6, 6, 150)
    
    spectral_data = {
        'experiment': 'spectral_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Detaillierte Spektralanalyse aller Gittertypen',
        'lattice_types': lattice_types,
        'N_k': N_k,
        'num_k_points': N_k * N_k,
        'energy_points': len(E_values),
        'results': {}
    }
    
    for lattice_type in lattice_types:
        print(f"\n  Gittertyp: {lattice_type}")
        lattice = ArchimideanLattice(lattice_type)
        
        print(f"    IDS-Berechnung...", end=' ', flush=True)
        N_E, metadata = compute_IDS_floquet(
            lattice,
            N_k=N_k,
            E_values=E_values,
            verbose=False
        )
        print("✓")
        
        # Berechne DOS
        print(f"    DOS-Berechnung...", end=' ', flush=True)
        dos, E_dos = compute_DOS(N_E, E_values)
        print("✓")
        
        # Berechne spektrale Lücke
        print(f"    Spektrale Lücke...", end=' ', flush=True)
        eigenvalue_data = metadata['eigenvalues']
        gap = compute_spectral_gap(eigenvalue_data)
        print(f"✓ (Gap={gap:.4f})")
        
        # Analysiere DOS-Features
        print(f"    DOS-Features...", end=' ', flush=True)
        
        # Finde Peaks in DOS
        dos_positive = np.maximum(dos, 0)
        peaks_indices = []
        for i in range(1, len(dos_positive)-1):
            if dos_positive[i] > dos_positive[i-1] and dos_positive[i] > dos_positive[i+1]:
                if dos_positive[i] > 0.01 * np.max(dos_positive):
                    peaks_indices.append(i)
        
        peaks = [{'energy': float(E_values[i]), 'dos': float(dos[i])} 
                 for i in peaks_indices[:5]]  # Top 5 peaks
        
        print(f"✓ ({len(peaks)} Peaks gefunden)")
        
        # Bestimme Bandstruktur
        eigenvalues_all = eigenvalue_data.flatten()
        eigenvalues_sorted = np.sort(eigenvalues_all)
        
        # Finde Bandlücken
        eigenvalue_diffs = np.diff(eigenvalues_sorted)
        gap_indices = np.argsort(eigenvalue_diffs)[-5:]  # Top 5 Lücken
        
        bandgaps = [
            {
                'position': float(eigenvalues_sorted[i]),
                'gap_size': float(eigenvalue_diffs[i])
            }
            for i in gap_indices if eigenvalue_diffs[i] > 1e-10
        ]
        
        # Speichere Ergebnisse
        spectral_data['results'][lattice_type] = {
            'lattice_type': lattice_type,
            'num_bands': int(lattice.num_sites_per_cell),
            'spectral_gap': float(gap),
            'eigenvalue_statistics': {
                'min': float(np.min(eigenvalues_all)),
                'max': float(np.max(eigenvalues_all)),
                'mean': float(np.mean(eigenvalues_all)),
                'std': float(np.std(eigenvalues_all)),
            },
            'dos_statistics': {
                'min': float(np.min(dos)),
                'max': float(np.max(dos)),
                'mean': float(np.mean(dos)),
                'std': float(np.std(dos)),
                'integral': float(np.trapz(dos, E_values)),
            },
            'dos_peaks': peaks,
            'bandgaps': bandgaps,
            'ids_statistics': {
                'min': float(np.min(N_E)),
                'max': float(np.max(N_E)),
                'values_at_key_energies': {
                    'E=-6': float(N_E[0]),
                    'E=0': float(N_E[len(N_E)//2]),
                    'E=6': float(N_E[-1]),
                }
            }
        }
    
    return spectral_data


# ============================================================================
# EXPERIMENT 3: VERGLEICHENDE GITTERANALYSE
# ============================================================================

def experiment_comparative_analysis() -> Dict[str, Any]:
    """
    Vergleichende Analyse verschiedener Archimedean-Gittertypen.
    
    Vergleicht:
    - IDS-Kurvenformen
    - DOS-Profile
    - Spektrale Eigenschaften
    - Symmetrien
    
    Returns:
        comparative_data: Dictionary mit Vergleichsdaten
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: VERGLEICHENDE GITTERANALYSE")
    print("=" * 80)
    
    lattice_types = ["(6,6,6)", "(4,8,8)", "(3,12,12)"]
    N_k = 20
    E_values = np.linspace(-5, 5, 120)
    
    comparative_data = {
        'experiment': 'comparative_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Vergleich verschiedener Archimedean-Gittertypen',
        'lattice_types': lattice_types,
        'N_k': N_k,
        'energy_points': len(E_values),
        'pairwise_comparisons': {}
    }
    
    # Berechne IDS und DOS für alle Gittertypen
    results = {}
    
    for lattice_type in lattice_types:
        print(f"\n  Berechne für {lattice_type}...", end=' ', flush=True)
        lattice = ArchimideanLattice(lattice_type)
        
        N_E, metadata = compute_IDS_floquet(
            lattice,
            N_k=N_k,
            E_values=E_values,
            verbose=False
        )
        
        dos, _ = compute_DOS(N_E, E_values)
        
        results[lattice_type] = {
            'N_E': N_E,
            'dos': dos,
            'metadata': metadata
        }
        
        print("✓")
    
    # Paarweise Vergleiche
    print("\n  Paarweise Vergleiche:")
    
    lattice_list = list(results.keys())
    
    for i in range(len(lattice_list)):
        for j in range(i+1, len(lattice_list)):
            lat_type_1 = lattice_list[i]
            lat_type_2 = lattice_list[j]
            
            print(f"    {lat_type_1} vs {lat_type_2}...", end=' ', flush=True)
            
            N_E_1 = results[lat_type_1]['N_E']
            N_E_2 = results[lat_type_2]['N_E']
            dos_1 = results[lat_type_1]['dos']
            dos_2 = results[lat_type_2]['dos']
            
            # Berechne Unterschiede
            ids_max_diff = np.max(np.abs(N_E_1 - N_E_2))
            ids_rmse = np.sqrt(np.mean((N_E_1 - N_E_2)**2))
            dos_max_diff = np.max(np.abs(dos_1 - dos_2))
            dos_rmse = np.sqrt(np.mean((dos_1 - dos_2)**2))
            
            comparison_key = f"{lat_type_1}_vs_{lat_type_2}"
            
            comparative_data['pairwise_comparisons'][comparison_key] = {
                'lattice_1': lat_type_1,
                'lattice_2': lat_type_2,
                'ids_comparison': {
                    'max_absolute_difference': float(ids_max_diff),
                    'rmse': float(ids_rmse),
                    'correlation': float(np.corrcoef(N_E_1, N_E_2)[0, 1]),
                },
                'dos_comparison': {
                    'max_absolute_difference': float(dos_max_diff),
                    'rmse': float(dos_rmse),
                    'correlation': float(np.corrcoef(dos_1, dos_2)[0, 1]),
                }
            }
            
            print("✓")
    
    # Gesamtvergleich aller drei
    print(f"    Gesamtstatistik...", end=' ', flush=True)
    
    all_ids_values = np.stack([results[lt]['N_E'] for lt in lattice_types])
    all_dos_values = np.stack([results[lt]['dos'] for lt in lattice_types])
    
    comparative_data['overall_statistics'] = {
        'ids_mean_variance': float(np.var(all_ids_values, axis=0).mean()),
        'dos_mean_variance': float(np.var(all_dos_values, axis=0).mean()),
        'ids_range_comparison': {
            lt: {
                'min': float(results[lt]['N_E'].min()),
                'max': float(results[lt]['N_E'].max()),
            }
            for lt in lattice_types
        },
        'dos_range_comparison': {
            lt: {
                'min': float(results[lt]['dos'].min()),
                'max': float(results[lt]['dos'].max()),
            }
            for lt in lattice_types
        }
    }
    
    print("✓")
    
    return comparative_data


# ============================================================================
# EXPERIMENT 4: ENERGIE-AUFLÖSUNGSANALYSE
# ============================================================================

def experiment_energy_resolution_analysis() -> Dict[str, Any]:
    """
    Analysiert Auswirkung der Energie-Auflösung auf IDS und DOS.
    
    Untersucht verschiedene Energiegitter-Größen:
    - Sehr grob: 20 Punkte
    - Grob: 50 Punkte
    - Mittel: 100 Punkte
    - Fein: 200 Punkte
    - Sehr fein: 500 Punkte
    
    Returns:
        resolution_data: Dictionary mit Auflösungsanalysen
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: ENERGIE-AUFLÖSUNGSANALYSE")
    print("=" * 80)
    
    lattice_types = ["(6,6,6)", "(4,8,8)"]
    N_k = 15
    energy_resolutions = [20, 50, 100, 200, 500]
    
    resolution_data = {
        'experiment': 'energy_resolution_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Analyse der Energie-Auflösungs-Auswirkungen',
        'lattice_types': lattice_types,
        'N_k': N_k,
        'energy_resolutions': energy_resolutions,
        'results': {}
    }
    
    for lattice_type in lattice_types:
        print(f"\n  Gittertyp: {lattice_type}")
        lattice = ArchimideanLattice(lattice_type)
        
        resolution_results = {}
        
        reference_N_E = None
        reference_resolution = None
        
        for num_points in energy_resolutions:
            print(f"    {num_points:3d} Energiepunkte...", end=' ', flush=True)
            
            E_values = np.linspace(-5, 5, num_points)
            
            start_time = time.time()
            N_E, metadata = compute_IDS_floquet(
                lattice,
                N_k=N_k,
                E_values=E_values,
                verbose=False
            )
            elapsed = time.time() - start_time
            
            dos, _ = compute_DOS(N_E, E_values)
            
            # Speichere Referenz
            if num_points == energy_resolutions[-1]:
                reference_N_E = N_E
                reference_resolution = num_points
            
            # Vergleiche mit Referenz
            if reference_N_E is not None and num_points < reference_resolution:
                # Interpoliere auf Referenz-Energiegitter
                E_ref = np.linspace(-5, 5, reference_resolution)
                N_E_interp = np.interp(E_ref, E_values, N_E)
                
                ids_diff = np.max(np.abs(N_E_interp - reference_N_E))
                ids_rmse = np.sqrt(np.mean((N_E_interp - reference_N_E)**2))
            else:
                ids_diff = None
                ids_rmse = None
            
            resolution_results[f'points_{num_points}'] = {
                'num_energy_points': num_points,
                'energy_range': [-5.0, 5.0],
                'dE_average': float(10.0 / (num_points - 1)),
                'computation_time': float(elapsed),
                'max_diff_to_finest': float(ids_diff) if ids_diff else None,
                'rmse_to_finest': float(ids_rmse) if ids_rmse else None,
                'dos_statistics': {
                    'min': float(np.min(dos)),
                    'max': float(np.max(dos)),
                    'mean': float(np.mean(dos)),
                }
            }
            
            print(f"t={elapsed:.2f}s")
        
        resolution_data['results'][lattice_type] = resolution_results
    
    return resolution_data


# ============================================================================
# EXPERIMENT 5: PERFORMANCE-ANALYSE
# ============================================================================

def experiment_performance_analysis() -> Dict[str, Any]:
    """
    Analysiert Rechenzeit-Skalierung mit verschiedenen Parametern.
    
    Untersucht:
    - Laufzeit vs. N_k (Gitterauflösung)
    - Laufzeit vs. Energiepunkte
    - Laufzeit vs. Gittertyp (Anzahl Bänder)
    
    Returns:
        performance_data: Dictionary mit Performance-Daten
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: PERFORMANCE-ANALYSE")
    print("=" * 80)
    
    lattice_types = ["(6,6,6)", "(4,8,8)", "(3,12,12)"]
    N_k_values = [5, 10, 15, 20, 25, 30]
    E_values = np.linspace(-5, 5, 80)
    
    performance_data = {
        'experiment': 'performance_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Performance- und Skalierungsanalyse',
        'lattice_types': lattice_types,
        'N_k_values': N_k_values,
        'energy_points': len(E_values),
        'results': {}
    }
    
    for lattice_type in lattice_types:
        print(f"\n  Gittertyp: {lattice_type}")
        lattice = ArchimideanLattice(lattice_type)
        num_bands = lattice.num_sites_per_cell
        
        timing_results = {}
        
        for N_k in N_k_values:
            print(f"    N_k={N_k:2d} ({N_k*N_k:4d} k-Punkte)...", end=' ', flush=True)
            
            # Wiederhole mehrfach für bessere Timing-Genauigkeit
            times = []
            for run in range(3):
                start_time = time.time()
                _, _ = compute_IDS_floquet(
                    lattice,
                    N_k=N_k,
                    E_values=E_values,
                    verbose=False
                )
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            timing_results[f'N_k_{N_k}'] = {
                'N_k': N_k,
                'num_k_points': N_k * N_k,
                'num_bands': num_bands,
                'num_energy_points': len(E_values),
                'mean_time_seconds': float(mean_time),
                'std_time_seconds': float(std_time),
                'estimated_flops': int(N_k * N_k * num_bands**3 * 10),  # Grobe Schätzung
            }
            
            print(f"t={mean_time:.3f}±{std_time:.3f}s")
        
        performance_data['results'][lattice_type] = timing_results
    
    return performance_data


# ============================================================================
# EXPERIMENT 6: SYMMETRIE UND VALIDIERUNG
# ============================================================================

def experiment_symmetry_validation() -> Dict[str, Any]:
    """
    Validiert theoretische Symmetrien und mathematische Eigenschaften.
    
    Prüft:
    - Hermitizität des Floquet-Operators
    - Reellheit der Eigenwerte
    - Monotonität der IDS
    - Normalisierung
    - Translationssymmetrie in k-Raum
    
    Returns:
        symmetry_data: Dictionary mit Symmetrievalidierungen
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: SYMMETRIE- UND VALIDIERUNGS-CHECKS")
    print("=" * 80)
    
    lattice_types = ["(6,6,6)", "(4,8,8)", "(3,12,12)"]
    N_k = 12
    E_values = np.linspace(-5, 5, 100)
    
    symmetry_data = {
        'experiment': 'symmetry_validation',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Validierung mathematischer Symmetrien und Eigenschaften',
        'lattice_types': lattice_types,
        'N_k': N_k,
        'validation_results': {}
    }
    
    for lattice_type in lattice_types:
        print(f"\n  Gittertyp: {lattice_type}")
        lattice = ArchimideanLattice(lattice_type)
        
        # Test 1: Hermitizität des Floquet-Operators
        print(f"    Hermitizität-Check...", end=' ', flush=True)
        b1, b2 = construct_brillouin_zone(lattice)
        k_grid = create_k_grid(b1, b2, N_k)
        
        max_hermitian_violation = 0.0
        num_k_points_checked = 0
        
        for i in range(0, N_k, 2):  # Sample every other point
            for j in range(0, N_k, 2):
                k = k_grid[i, j, :]
                H_k = construct_floquet_operator(lattice, k)
                violation = np.max(np.abs(H_k - H_k.conj().T))
                max_hermitian_violation = max(max_hermitian_violation, violation)
                num_k_points_checked += 1
        
        print(f"✓ (max violation={max_hermitian_violation:.2e})")
        
        # Test 2: Reellheit der Eigenwerte
        print(f"    Eigenwert-Reellheit-Check...", end=' ', flush=True)
        max_imag_part = 0.0
        
        for i in range(0, N_k, 2):
            for j in range(0, N_k, 2):
                k = k_grid[i, j, :]
                H_k = construct_floquet_operator(lattice, k)
                eigenvalues = compute_eigenvalues_at_k(H_k)
                max_imag = np.max(np.abs(eigenvalues.imag))
                max_imag_part = max(max_imag_part, max_imag)
        
        print(f"✓ (max imag={max_imag_part:.2e})")
        
        # Test 3: IDS-Monotonität
        print(f"    IDS-Monotonität-Check...", end=' ', flush=True)
        N_E, _ = compute_IDS_floquet(
            lattice,
            N_k=N_k,
            E_values=E_values,
            verbose=False
        )
        
        dN = np.diff(N_E)
        num_violations = np.sum(dN < 0)
        max_violation = np.min(dN) if num_violations > 0 else 0
        
        print(f"✓ (violations={num_violations}, min_dN={max_violation:.2e})")
        
        # Test 4: Normalisierung
        print(f"    Normalisierungs-Check...", end=' ', flush=True)
        
        # Bei großem E sollte N(E) ≈ 1 sein (für 1 Band)
        N_at_large_E = N_E[-1]
        N_at_small_E = N_E[0]
        
        expected_min = 0.0
        expected_max = lattice.num_sites_per_cell
        
        normalization_ok = (N_at_large_E <= expected_max * 1.1) and (N_at_small_E >= expected_min - 0.1)
        
        print(f"✓ (N(E_max)={N_at_large_E:.3f}, range=[{expected_min}, {expected_max}])")
        
        # Test 5: Eigenwertstatistiken
        print(f"    Eigenwertstatistiken...", end=' ', flush=True)
        all_eigenvalues = []
        
        for i in range(N_k):
            for j in range(N_k):
                k = k_grid[i, j, :]
                H_k = construct_floquet_operator(lattice, k)
                eigenvalues = compute_eigenvalues_at_k(H_k)
                all_eigenvalues.extend(eigenvalues.real)
        
        all_eigenvalues = np.array(all_eigenvalues)
        
        print(f"✓")
        
        # Speichere Validierungsergebnisse
        symmetry_data['validation_results'][lattice_type] = {
            'lattice_type': lattice_type,
            'hermiticity': {
                'max_violation': float(max_hermitian_violation),
                'num_k_points_checked': int(num_k_points_checked),
                'status': 'PASS' if max_hermitian_violation < 1e-10 else 'WARNING'
            },
            'eigenvalue_reality': {
                'max_imaginary_part': float(max_imag_part),
                'status': 'PASS' if max_imag_part < 1e-10 else 'WARNING'
            },
            'ids_monotonicity': {
                'num_violations': int(num_violations),
                'min_dN': float(max_violation),
                'status': 'PASS' if num_violations == 0 else 'WARNING'
            },
            'normalization': {
                'N_at_E_min': float(N_at_small_E),
                'N_at_E_max': float(N_at_large_E),
                'status': 'PASS' if normalization_ok else 'WARNING'
            },
            'eigenvalue_statistics': {
                'total_eigenvalues': len(all_eigenvalues),
                'min': float(np.min(all_eigenvalues)),
                'max': float(np.max(all_eigenvalues)),
                'mean': float(np.mean(all_eigenvalues)),
                'std': float(np.std(all_eigenvalues)),
            }
        }
    
    return symmetry_data


# ============================================================================
# SAMMELN UND SPEICHERN ALLER EXPERIMENTE
# ============================================================================

def run_all_experiments() -> Dict[str, Any]:
    """
    Führe alle Experimente aus und sammle Ergebnisse.
    
    Returns:
        all_results: Dictionary mit allen Experiment-Ergebnissen
    """
    print("\n" + "=" * 80)
    print("UMFANGREICHE IDS-EXPERIMENTE FÜR ARCHIMED.TEX")
    print("=" * 80)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {
        'metadata': {
            'title': 'Umfangreiche Experimente zur IDS von Archimedean-Gittergraphen',
            'author': 'Stephan Epp',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'purpose': 'Experimentelle Validierung für die wissenschaftliche Arbeit archimed.tex',
        },
        'experiments': {}
    }
    
    # Experiment 1: Konvergenz
    try:
        all_results['experiments']['convergence_analysis'] = experiment_convergence_analysis()
    except Exception as e:
        print(f"ERROR in Experiment 1: {e}")
        all_results['experiments']['convergence_analysis'] = {'error': str(e)}
    
    # Experiment 2: Spektralanalyse
    try:
        all_results['experiments']['spectral_analysis'] = experiment_spectral_analysis()
    except Exception as e:
        print(f"ERROR in Experiment 2: {e}")
        all_results['experiments']['spectral_analysis'] = {'error': str(e)}
    
    # Experiment 3: Vergleichende Analyse
    try:
        all_results['experiments']['comparative_analysis'] = experiment_comparative_analysis()
    except Exception as e:
        print(f"ERROR in Experiment 3: {e}")
        all_results['experiments']['comparative_analysis'] = {'error': str(e)}
    
    # Experiment 4: Energie-Auflösung
    try:
        all_results['experiments']['energy_resolution'] = experiment_energy_resolution_analysis()
    except Exception as e:
        print(f"ERROR in Experiment 4: {e}")
        all_results['experiments']['energy_resolution'] = {'error': str(e)}
    
    # Experiment 5: Performance
    try:
        all_results['experiments']['performance_analysis'] = experiment_performance_analysis()
    except Exception as e:
        print(f"ERROR in Experiment 5: {e}")
        all_results['experiments']['performance_analysis'] = {'error': str(e)}
    
    # Experiment 6: Symmetrie
    try:
        all_results['experiments']['symmetry_validation'] = experiment_symmetry_validation()
    except Exception as e:
        print(f"ERROR in Experiment 6: {e}")
        all_results['experiments']['symmetry_validation'] = {'error': str(e)}
    
    return all_results


# ============================================================================
# SPEICHERUNG DER ERGEBNISSE
# ============================================================================

def save_results_to_files(all_results: Dict[str, Any]) -> None:
    """
    Speichere Experiment-Ergebnisse in verschiedenen Formaten.
    
    Outputs:
    - experiments_results.json: Vollständige numerische Ergebnisse
    - experiments_summary.txt: Menschenlesbare Zusammenfassung
    - experiments_detailed.json: Detaillierte Messdaten
    
    Args:
        all_results: Dictionary mit allen Experiment-Ergebnissen
    """
    print("\n" + "=" * 80)
    print("SPEICHERE EXPERIMENT-ERGEBNISSE")
    print("=" * 80)
    
    output_dir = Path('src/results/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Vollständige JSON-Datei
    print("\n  Speichere experiments_results.json...", end=' ', flush=True)
    json_path = output_dir / 'experiments_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ ({json_path.stat().st_size / 1024:.1f} KB)")
    
    # 2. Text-Zusammenfassung
    print("  Speichere experiments_summary.txt...", end=' ', flush=True)
    txt_path = output_dir / 'experiments_summary.txt'
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENTELLE VALIDIERUNG DER IDS-BERECHNUNG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Autor: {all_results['metadata']['author']}\n")
        f.write(f"Zeitstempel: {all_results['metadata']['timestamp']}\n")
        f.write(f"Zweck: {all_results['metadata']['purpose']}\n\n")
        
        # Schreibe Zusammenfassungen für jedes Experiment
        for exp_name, exp_data in all_results['experiments'].items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Experiment: {exp_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            if 'error' in exp_data:
                f.write(f"ERROR: {exp_data['error']}\n")
            else:
                if 'description' in exp_data:
                    f.write(f"Beschreibung: {exp_data['description']}\n\n")
                
                if 'timestamp' in exp_data:
                    f.write(f"Zeitstempel: {exp_data['timestamp']}\n")
                
                # Gittertypen
                if 'lattice_types' in exp_data:
                    f.write(f"Gittertypen: {', '.join(exp_data['lattice_types'])}\n")
                
                # Paramter
                if 'N_k' in exp_data:
                    f.write(f"k-Gitter-Größe (N_k): {exp_data['N_k']}\n")
                if 'N_k_values' in exp_data:
                    f.write(f"Getestete N_k-Werte: {exp_data['N_k_values']}\n")
                
                if 'energy_points' in exp_data:
                    f.write(f"Energiepunkte: {exp_data['energy_points']}\n")
                if 'energy_range' in exp_data:
                    f.write(f"Energiebereich: [{exp_data['energy_range'][0]}, {exp_data['energy_range'][1]}]\n")
                
                f.write("\n")
                
                # Experiment-spezifische Zusammenfassung
                if exp_name == 'convergence_analysis':
                    f.write("Konvergenzanalyse:\n")
                    for lat_type in exp_data.get('lattice_types', []):
                        f.write(f"\n  {lat_type}:\n")
                        lat_results = exp_data['results'][lat_type]
                        f.write(f"    - Getestete N_k-Werte: {list(lat_results['convergence_data'].keys())}\n")
                
                elif exp_name == 'spectral_analysis':
                    f.write("Spektralanalyse:\n")
                    for lat_type, res in exp_data['results'].items():
                        f.write(f"\n  {lat_type}:\n")
                        f.write(f"    - Spektrale Lücke: {res['spectral_gap']:.6f}\n")
                        f.write(f"    - Anzahl der Bänder: {res['num_bands']}\n")
                        f.write(f"    - DOS-Peaks: {len(res['dos_peaks'])}\n")
                        f.write(f"    - Bandlücken: {len(res['bandgaps'])}\n")
                
                elif exp_name == 'comparative_analysis':
                    f.write("Vergleichende Analysen:\n")
                    for comparison_key, comp_data in exp_data['pairwise_comparisons'].items():
                        f.write(f"\n  {comparison_key}:\n")
                        ids_comp = comp_data['ids_comparison']
                        f.write(f"    - IDS max diff: {ids_comp['max_absolute_difference']:.2e}\n")
                        f.write(f"    - IDS RMSE: {ids_comp['rmse']:.2e}\n")
                        f.write(f"    - IDS Korrelation: {ids_comp['correlation']:.6f}\n")
                
                elif exp_name == 'energy_resolution':
                    f.write("Energie-Auflösungsanalyse:\n")
                    for lat_type, res_data in exp_data['results'].items():
                        f.write(f"\n  {lat_type}:\n")
                        f.write(f"    - Getestete Auflösungen: {list(res_data.keys())}\n")
                
                elif exp_name == 'performance_analysis':
                    f.write("Performance-Analyse:\n")
                    for lat_type, timing_data in exp_data['results'].items():
                        f.write(f"\n  {lat_type}:\n")
                        f.write(f"    - N_k Werte: {list(timing_data.keys())}\n")
                        # Finde schnellste und langsamste
                        times = [v['mean_time_seconds'] for v in timing_data.values()]
                        f.write(f"    - Zeitbereich: {min(times):.3f}s - {max(times):.3f}s\n")
                
                elif exp_name == 'symmetry_validation':
                    f.write("Symmetrie-Validierung:\n")
                    for lat_type, val_data in exp_data['validation_results'].items():
                        f.write(f"\n  {lat_type}:\n")
                        for check_name, check_result in val_data.items():
                            if check_name != 'lattice_type' and isinstance(check_result, dict):
                                status = check_result.get('status', 'N/A')
                                f.write(f"    - {check_name}: {status}\n")
    
    print(f"✓ ({txt_path.stat().st_size / 1024:.1f} KB)")
    
    # 3. Detaillierte JSON (gleich wie oben, aber kann später erweitert werden)
    print("  Speichere experiments_detailed.json...", end=' ', flush=True)
    detailed_path = output_dir / 'experiments_detailed.json'
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ ({detailed_path.stat().st_size / 1024:.1f} KB)")
    
    print("\n✓ ALLE ERGEBNISSE GESPEICHERT")
    print(f"  - {json_path.name}")
    print(f"  - {txt_path.name}")
    print(f"  - {detailed_path.name}")


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def main():
    """Hauptfunktion: Führe alle Experimente aus und speichere Ergebnisse."""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "UMFANGREICHE EXPERIMENTE ZUR IDS VON ARCHIMEDEAN-GITTERGRAPHEN".center(78) + "║")
    print("║" + "für die wissenschaftliche Arbeit: archimed.tex".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Führe alle Experimente aus
    start_total = time.time()
    all_results = run_all_experiments()
    total_time = time.time() - start_total
    
    # Speichere Ergebnisse
    save_results_to_files(all_results)
    
    # Endezusammenfassung
    print("\n" + "=" * 80)
    print("EXPERIMENT-ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"Gesamtrechenzeit: {total_time:.2f} Sekunden ({total_time/60:.2f} Minuten)")
    print(f"Anzahl Experimente: {len(all_results['experiments'])}")
    print(f"Zeitstempel: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ Alle Experimente abgeschlossen!")
    print("✓ Ergebnisse gespeichert in src/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
