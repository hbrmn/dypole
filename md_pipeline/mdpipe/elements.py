"""Element data needed by the pipeline: masses (g/mol) and coherent
neutron scattering lengths b (fm, natural isotopic abundance, NIST tables)."""

MASS = {
    'H': 1.008, 'Li': 6.94, 'B': 10.81, 'O': 15.999, 'F': 18.998,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'K': 39.098, 'Ca': 40.078, 'Ti': 47.867, 'Zn': 65.38, 'Sr': 87.62,
    'Ba': 137.327,
}

NEUTRON_B = {
    'H': -3.739, 'Li': -1.90, 'B': 5.30, 'O': 5.803, 'F': 5.654,
    'Na': 3.63, 'Mg': 5.375, 'Al': 3.449, 'Si': 4.1491, 'P': 5.13,
    'K': 3.67, 'Ca': 4.70, 'Ti': -3.438, 'Zn': 5.68, 'Sr': 7.02,
    'Ba': 5.07,
}
