# IDS-Berechnung für Archimedean-Gittergraphen

**Integrierte Zustandsdichte (IDS) von k-uniformen Tessellationen der euklidischen Ebene**

## Übersicht

Dieses Repository enthält eine umfassende Python-Implementierung zur numerischen Berechnung der **Integrierten Zustandsdichte (IDS)** von k-uniformen Tessellationen basierend auf der wissenschaftlichen Arbeit aus https://github.com/hjstephan86/science/tree/main/science/archimed.

### Hauptmerkmale

 **k-uniforme Tessellationen Support**
- Archimedean-Tessellationen (k=1): 11 Varianten
- 2-uniforme bis 8-uniforme Tessellationen (k=2 bis k=8)
- Vollständige Klassifikation von 183 k-uniformen Tessellationen

 **Floquet-Bloch-Theorie Implementation**
- Brillouin-Zone Konstruktion für beliebige 2D-Gitter
- Adaptive k-Gitter-Generierung mit Symmetrie-Reduktion
- Floquet-Operator für quasi-periodische Systeme
- Eigenvalue-Berechnung mit numerischer Stabilität

 **Spektralanalyse**
- Integrierte Zustandsdichte (IDS) mittels Heaviside-Regularisierung
- Density of States (DOS) durch Differentiation
- Spektrale Lücken und Bandstrukturen
- Eigenvalue-Statistiken und Komplexitätsanalyse

## Installation

### Voraussetzungen

- Python 3.10 oder höher
- pip oder conda

### Setup

```bash
# 1. Repository klonen (oder als ZIP herunterladen)
git clone https://github.com/hjstephan86/ids.git
cd ids

# 2. Virtual Environment erstellen
python -m venv venv

# 3. Virtual Environment aktivieren
source venv/bin/activate              # Linux/macOS
# oder
venv\Scripts\activate                 # Windows

# 4. Dependencies installieren
 pip install -e ".[dev]"
```

## Tests ausführen

### Schnell-Test

```bash
# Alle Tests ausführen
pytest
```

### Nur spezifische Test-Klasse

```bash
# Nur k-uniforme Tests
pytest tests/test_kuniform_ids.py -v

# Nur Original-Tests
pytest tests/test_ids_calculator.py -v

# Nur IDS-Berechnung Tests
pytest tests/test_kuniform_ids.py::TestIDSComputation -v
```

## Hauptmodule

### 1. `ids_kuniform_calculator.py`

Implementierung für **k-uniforme Tessellationen** mit Floquet-Bloch-Theorie.

#### Haupt-Klassen

**`VertexOrbit`**
```python
orbit = VertexOrbit(
    orbit_id=0,
    vertex_configuration=(6, 6, 6),
    positions=np.array([[0.0, 0.0]]),
    coordination_number=3,
    symmetry_group='p6mm',
    multiplicity=1
)
```

**`KUniformTessellation`**
```python
tessellation = KUniformTessellation(
    name="(6.6.6)",
    k_uniform=1,
    vertex_orbits=[orbit],
    hopping_matrix=np.array([[0.0]]),
    reciprocal_vectors=np.array([[2*np.pi, 0], [np.pi, np.pi*np.sqrt(3)]]),
    wallpaper_group='p6mm',
    total_vertices_per_cell=1
)
```

**`KUniformLattice`**
```python
from ids_kuniform_calculator import KUniformLattice

lattice = KUniformLattice(tessellation)
print(f"k-Uniformitätsgrad: {lattice.k_uniform}")
print(f"Vertex-Orbits: {lattice.num_vertex_orbits}")
```

#### Haupt-Funktionen

**IDS-Berechnung**
```python
from ids_kuniform_calculator import compute_IDS_kuniform

# Berechne IDS über Brillouin-Zone
E_values = np.linspace(-3, 3, 50)
N_E, metadata = compute_IDS_kuniform(
    lattice, 
    N_k=10,              # k-Gitter Auflösung
    E_values=E_values,
    sigma=0.01,          # Regularisierungs-Breite
    verbose=True
)

# Resultat: N_E ist die IDS, metadata enthält Statistiken
print(f"IDS-Bereich: [{N_E.min():.4f}, {N_E.max():.4f}]")
```

**DOS-Berechnung**
```python
from ids_kuniform_calculator import compute_DOS_kuniform

dos, dos_E = compute_DOS_kuniform(N_E, E_values)
```

**Spektralanalyse**
```python
from ids_kuniform_calculator import analyze_spectral_structure

spectrum = analyze_spectral_structure(np.array(metadata['eigenvalues']))
```

#### Unterstützte Tessellationen

```python
from ids_kuniform_calculator import KUniformLibrary

# Liste alle verfügbaren Tessellationen
all_tess = KUniformLibrary.list_all()

# Hole Info über spezifische Tessellation
info = KUniformLibrary.get_tessellation_info("(6.6.6)")
print(info)
# Output: {'k': 1, 'name': 'Hexagonal', 'orbits': 1, 'vertices_per_cell': 1}
```

### 2. `ids_calculator.py` (Original)

Implementierung für **1-uniforme (Archimedean)** Tessellationen.

```python
from ids_calculator import compute_IDS_archimedean

# Für einfache Archimedean-Gitter
IDS = compute_IDS_archimedean(lattice_type="hexagonal", N_k=20)
```

## Experiment-Module

### `experiments_kuniform.py`

Wissenschaftliche Experimente für k-uniforme Tessellationen:

```bash
python -m src.experiments_kuniform
```

**Enthält Experimente zu:**
- Konvergenzverhalten mit k-Gitter-Größe
- Spektralanalyse verschiedener Tessellationen
- Vergleichende Gitteranalyse
- Performance-Charakteristiken

### `experiments.py` (Original)

Experimente für Archimedean-Gitter:

```bash
python -m src.experiments
```

## Beispiel-Workflow

### Archimedean Hexagonal Lattice (k=1)

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from ids_kuniform_calculator import (
    KUniformTessellation, VertexOrbit, KUniformLattice,
    compute_IDS_kuniform, compute_DOS_kuniform
)

# 1. Erstelle Tessellation
orbit = VertexOrbit(
    orbit_id=0,
    vertex_configuration=(6, 6, 6),
    positions=np.array([[0.0, 0.0]]),
    coordination_number=3,
    symmetry_group='p6mm',
    multiplicity=1
)

tess = KUniformTessellation(
    name="(6.6.6)",
    k_uniform=1,
    vertex_orbits=[orbit],
    hopping_matrix=np.array([[0.0]]),
    reciprocal_vectors=np.array([
        [2*np.pi, 0.0],
        [np.pi, np.pi*np.sqrt(3)]
    ]),
    wallpaper_group='p6mm',
    total_vertices_per_cell=1
)

# 2. Erstelle Gitter
lattice = KUniformLattice(tess)

# 3. Berechne IDS
E_values = np.linspace(-3, 3, 100)
N_E, metadata = compute_IDS_kuniform(
    lattice, N_k=20, E_values=E_values, verbose=True
)

# 4. Berechne DOS
dos, dos_E = compute_DOS_kuniform(N_E, E_values)

# 5. Visualisiere
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(E_values, N_E, 'b-', linewidth=2)
plt.xlabel('Energie E')
plt.ylabel('Integrierte Zustandsdichte N(E)')
plt.title('IDS des Hexagonal Lattice')
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.plot(dos_E, dos, 'r-', linewidth=2)
plt.xlabel('Energie E')
plt.ylabel('Density of States ρ(E)')
plt.title('DOS des Hexagonal Lattice')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ids_dos_hexagonal.png', dpi=150)
plt.show()
```

*Die Spektraltheorie periodischer Gitter ist fundamental für unser Verständnis von Festkörpern, Photonischen Kristallen und vielen anderen physikalischen Systemen.*
