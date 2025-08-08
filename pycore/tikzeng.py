"""Minimal subset of PlotNeuralNet's tikzeng utilities.
This implementation intentionally omits numeric dimension labels.
"""

from typing import List

# ------------------------------------------------------------
# Basic document structure helpers
# ------------------------------------------------------------

def to_head(title: str = '', scale: float = 1.0) -> str:
    """Return LaTeX header for a tikzfigure with optional scale."""
    return f"""\\documentclass{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning}}
\\tikzstyle{{caption}}=[font=\\footnotesize]
\\begin{{document}}
\\begin{{tikzpicture}}[scale={scale}]
"""


def to_cor() -> str:
    """Compatibility placeholder for original library."""
    return ""


def to_begin() -> str:
    return ""


def to_end() -> str:
    return "\\end{tikzpicture}\n\\end{document}\n"

# ------------------------------------------------------------
# Layer drawing helpers without numeric dimension labels
# ------------------------------------------------------------

def _box(name: str, caption: str, to: str, offset: str,
         width: float, height: float, color: str) -> str:
    return f"""
% {name}
\\coordinate ({name}_pos) at {to};
\\node[draw, fill={color}, minimum width={width}cm, minimum height={height}cm] ({name}) at ($( {name}_pos ) + {offset}$) {{}};
\\node[caption] at ({name}.south) {{{caption}}};
"""


def to_Conv(name: str, s_filer: int, n_filer: int, offset: str = "(0,0,0)",
            to: str = "(0,0,0)", width: float = 2, height: float = 2,
            depth: float = 2, caption: str = "") -> str:
    return _box(name, caption, to, offset, width, height, color="blue!20")


def to_Pool(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
            width: float = 1, height: float = 1, depth: float = 1,
            caption: str = "") -> str:
    return _box(name, caption, to, offset, width, height, color="green!20")


def to_FullyConnected(name: str, n_neuron: int, offset: str = "(0,0,0)",
                       to: str = "(0,0,0)", width: float = 1.5,
                       height: float = 1, depth: float = 1,
                       caption: str = "") -> str:
    return _box(name, caption, to, offset, width, height, color="orange!20")


def to_SoftMax(name: str, n_classes: int, offset: str = "(0,0,0)",
               to: str = "(0,0,0)", width: float = 1.5,
               height: float = 1, depth: float = 1,
               caption: str = "") -> str:
    return _box(name, caption, to, offset, width, height, color="red!20")

# ------------------------------------------------------------
# Connections and generation utilities
# ------------------------------------------------------------

def to_connection(of: str, to: str) -> str:
    return f"\\draw[->] ({of}.east) -- ({to}.west);\n"


def to_generate(arch: List[str], filename: str) -> None:
    with open(filename, 'w') as f:
        f.writelines(arch)
