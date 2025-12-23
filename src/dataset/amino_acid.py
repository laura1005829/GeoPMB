import numpy as np
from enum import Enum
from numpy.typing import NDArray
from typing_extensions import Self


class Polarity(Enum):
    """One-hot encoding of the amino acid polarity."""
    NONPOLAR = 0
    POLAR = 1
    NEGATIVE = 2
    POSITIVE = 3

    @property
    def onehot(self) -> NDArray:
        t = np.zeros(4)
        t[self.value] = 1.0
        return t


class AminoAcid:
    """An amino acid represents the type of `Residue` in a `PDBStructure`.

    Args:
        name (str): Full name of the amino acid.
        three_letter_code (str): Three-letter code of the amino acid (as in PDB).
        one_letter_code (str): One-letter of the amino acid (as in fasta).
        charge (int): Charge of the amino acid.
        polarity (:class:`Polarity`): The polarity of the amino acid.
        size (int): The number of non-hydrogen atoms in the side chain.
        mass (float): Average residue mass (i.e. mass of amino acid - H20) in Daltons.
        pI (float): Isolectric point; pH at which the molecule has no net electric charge.
        hydrogen_bond_donors (int): Number of hydrogen bond donors.
        hydrogen_bond_acceptors (int): Number of hydrogen bond acceptors.
        index (int): The rank of the amino acid, used for computing one-hot encoding.
    """

    def __init__(
        self,
        name: str,
        three_letter_code: str,
        one_letter_code: str,
        charge: int,
        polarity: Polarity,
        size: int,
        mass: float,
        pI: float,
        hydrogen_bond_donors: int,
        hydrogen_bond_acceptors: int,
        index: int,
    ):
        # amino acid nomenclature
        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

        # side chain properties
        self._charge = charge
        self._polarity = polarity
        self._size = size
        self._mass = mass
        self._pI = pI
        self._hydrogen_bond_donors = hydrogen_bond_donors
        self._hydrogen_bond_acceptors = hydrogen_bond_acceptors

        # one hot encoding
        self._index = index

    @property
    def name(self) -> str:
        return self._name

    @property
    def three_letter_code(self) -> str:
        return self._three_letter_code

    @property
    def one_letter_code(self) -> str:
        return self._one_letter_code

    @property
    def charge(self) -> int:
        return self._charge

    @property
    def polarity(self) -> Polarity:
        return self._polarity

    @property
    def size(self) -> int:
        return self._size

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def pI(self) -> float:
        return self._pI

    @property
    def hydrogen_bond_donors(self) -> int:
        return self._hydrogen_bond_donors

    @property
    def hydrogen_bond_acceptors(self) -> int:
        return self._hydrogen_bond_acceptors

    @property
    def onehot(self) -> NDArray:
        if self._index is None:
            msg = f"Amino acid {self._name} index is not set, thus no onehot can be computed."
            raise ValueError(msg)

        a = np.zeros(20)
        a[self._index] = 1.0
        return a

    @property
    def index(self) -> int:
        return self._index

    def __hash__(self) -> hash:
        return hash(self.name)

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, AminoAcid):
            return other.name == self.name
        return NotImplemented

    def __repr__(self) -> str:
        return self._three_letter_code


alanine = AminoAcid(
    "Alanine",
    "ALA",
    "A",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=1,
    mass=71.1,
    pI=6.00,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=0,
)

cysteine = AminoAcid(
    "Cysteine",
    "CYS",
    "C",
    charge=0,
    polarity=Polarity.POLAR,  
    size=2,
    mass=103.2,
    pI=5.07,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=1,
)

aspartate = AminoAcid(
    "Aspartate",
    "ASP",
    "D",
    charge=-1,
    polarity=Polarity.NEGATIVE,
    size=4,
    mass=115.1,
    pI=2.77,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=4,
    index=2,
)

glutamate = AminoAcid(
    "Glutamate",
    "GLU",
    "E",
    charge=-1,
    polarity=Polarity.NEGATIVE,
    size=5,
    mass=129.1,
    pI=3.22,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=4,
    index=3,
)

phenylalanine = AminoAcid(
    "Phenylalanine",
    "PHE",
    "F",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=7,
    mass=147.2,
    pI=5.48,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=4,
)

glycine = AminoAcid(
    "Glycine",
    "GLY",
    "G",
    charge=0,
    polarity=Polarity.NONPOLAR,  
    size=0,
    mass=57.1,
    pI=5.97,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=5,
)

histidine = AminoAcid(
    "Histidine",
    "HIS",
    "H",
    charge=1,
    polarity=Polarity.POSITIVE,
    size=6,
    mass=137.1,
    pI=7.59,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=1,
    # both position 7 and 10 can serve as either donor or acceptor (depending on tautomer), but any single His will have exactly one donor and one acceptor
    # (see https://foldit.fandom.com/wiki/Histidine)
    index=6,
)

isoleucine = AminoAcid(
    "Isoleucine",
    "ILE",
    "I",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=4,
    mass=113.2,
    pI=6.02,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=7,
)

lysine = AminoAcid(
    "Lysine",
    "LYS",
    "K",
    charge=1,
    polarity=Polarity.POSITIVE,
    size=5,
    mass=128.2,
    pI=9.74,  # 9.60 in source 3
    hydrogen_bond_donors=3,
    hydrogen_bond_acceptors=0,
    index=8,
)

leucine = AminoAcid(
    "Leucine",
    "LEU",
    "L",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=4,
    mass=113.2,
    pI=5.98,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=9,
)

methionine = AminoAcid(
    "Methionine",
    "MET",
    "M",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=4,
    mass=131.2,
    pI=5.74,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=10,
)

asparagine = AminoAcid(
    "Asparagine",
    "ASN",
    "N",
    charge=0,
    polarity=Polarity.POLAR,
    size=4,
    mass=114.1,
    pI=5.41,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=11,
)

proline = AminoAcid(
    "Proline",
    "PRO",
    "P",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=3,
    mass=97.1,
    pI=6.30,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=12,
)

glutamine = AminoAcid(
    "Glutamine",
    "GLN",
    "Q",
    charge=0,
    polarity=Polarity.POLAR,
    size=5,
    mass=128.1,
    pI=5.65,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=13,
)

arginine = AminoAcid(
    "Arginine",
    "ARG",
    "R",
    charge=1,
    polarity=Polarity.POSITIVE,
    size=7,
    mass=156.2,
    pI=10.76,
    hydrogen_bond_donors=5,
    hydrogen_bond_acceptors=0,
    index=14,
)

serine = AminoAcid(
    "Serine",
    "SER",
    "S",
    charge=0,
    polarity=Polarity.POLAR,
    size=2,
    mass=87.1,
    pI=5.68,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=15,
)

threonine = AminoAcid(
    "Threonine",
    "THR",
    "T",
    charge=0,
    polarity=Polarity.POLAR,
    size=3,
    mass=101.1,
    pI=5.60,  # 6.16 in source 2
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=16,
)

valine = AminoAcid(
    "Valine",
    "VAL",
    "V",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=3,
    mass=99.1,
    pI=5.96,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=17,
)

tryptophan = AminoAcid(
    "Tryptophan",
    "TRP",
    "W",
    charge=0,
    polarity=Polarity.NONPOLAR,  # source 4: polar
    size=10,
    mass=186.2,
    pI=5.89,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=0,
    index=18,
)

tyrosine = AminoAcid(
    "Tyrosine",
    "TYR",
    "Y",
    charge=-0.0,
    polarity=Polarity.POLAR,  # source 3: nonpolar
    size=8,
    mass=163.2,
    pI=5.66,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=1,
    index=19,
)

amino_acids = [
    alanine,
    arginine,
    asparagine,
    aspartate,
    cysteine,
    glutamate,
    glutamine,
    glycine,
    histidine,
    isoleucine,
    leucine,
    lysine,
    methionine,
    phenylalanine,
    proline,
    serine,
    threonine,
    tryptophan,
    tyrosine,
    valine,
]
amino_acids_by_code = {amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids}
amino_acids_by_letter = {amino_acid.one_letter_code: amino_acid for amino_acid in amino_acids}
amino_acids_by_name = {amino_acid.name: amino_acid for amino_acid in amino_acids}