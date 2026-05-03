"""
Shared figure style for publication-quality plots: fonts, sizes, DPI, color palette.
Use apply_style() at the top of plotting scripts for consistency across the paper.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# Single-column and double-column figure sizes (inches) for typical journals
FIG_SIZE_SINGLE = (8, 6)
FIG_SIZE_DOUBLE = (12, 8)
DPI = 300


def apply_style(style='default'):
    """
    Apply shared style to matplotlib.
    Call once at the start of a script (e.g. after importing pyplot).
    """
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.dpi'] = 100  # screen; savefig(..., dpi=DPI) for output
    mpl.rcParams['savefig.dpi'] = DPI
    mpl.rcParams['savefig.bbox'] = 'tight'
    # Prefer colorblind-friendly palette when available
    for name in ('seaborn-v0_8-colorblind', 'seaborn-colorblind'):
        if name in plt.style.available:
            plt.style.use(name)
            break
