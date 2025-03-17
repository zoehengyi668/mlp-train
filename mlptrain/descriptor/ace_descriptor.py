import numpy as np
import mlptrain
import julia
from julia import Main
from typing import Union, Optional, Sequence
from mlptrain.descriptor._base import Descriptor
from julia.api import Julia

jl = Julia(compiled_modules=False)


class ACEDescriptor(Descriptor):
    """ACE Descriptor Representation."""

    def __init__(
        self,
        elements: Optional[Sequence[str]] = None,
        N: int = 3,
        max_deg: int = 12,
        r0: float = 2.3,
        rin: float = 0.1,
        rcut: float = 5.0,
        pin: int = 2,
    ):
        """
        Initializes an ACE descriptor for computing the Atomic Cluster Expansion (ACE) representation.

        Arguments:
            elements (Optional[Sequence[str]]): Atomic species to be used for the ACE basis.
            N (int): (N+1)is the body correlation number, i.e., N=3 means up to 4-body correlations.
            max_deg (int): Maximum polynomial degree for expansion.
            r0 (float): Reference bond length.
            rin (float): Inner cutoff for radial basis.
            rcut (float): Cutoff radius for interactions.
            pin (int): Power exponent for the radial basis.
        """
        super().__init__(name='ACEDescriptor')
        self.elements = elements
        self.N = N
        self.max_deg = max_deg
        self.r0 = r0
        self.rin = rin
        self.rcut = rcut
        self.pin = pin

        # Initialize Julia and ACE1pack
        self.jl = julia.api.Julia(compiled_modules=False)
        Main.eval(
            'using ACE1pack, LazyArtifacts, MultivariateStats, JuLIP, Glob'
        )

        # Dynamically set basis if elements are provided
        if self.elements:
            self._initialize_basis()
        else:
            self.basis = None

    def _initialize_basis(self):
        """Initializes the ACE basis with the given elements."""
        species_julia = '[:{}]'.format(', :'.join(self.elements))
        Main.eval(
            f"""
        basis = ace_basis(
            species = {species_julia},
            N = {self.N},
            maxdeg = {self.max_deg},
            r0 = {self.r0},
            rin = {self.rin},
            rcut = {self.rcut},
            pin = {self.pin}
        )
        """
        )
        self.basis = Main.eval('basis')

    def compute_representation(
        self,
        configurations: Union[
            mlptrain.Configuration, mlptrain.ConfigurationSet
        ],
    ) -> np.ndarray:
        """Create a SOAP vector using dscribe (https://github.com/SINGROUP/dscribe)
        for a set of configurations

        ace_vector(config)           -> [[v0, v1, ..]]

        ace_vector(config1, config2) -> [[v0, v1, ..],
                                      [u0, u1, ..]]

        ace_vector(configset)        -> [[v0, v1, ..], ..]

        ---------------------------------------------------------------------------
        Arguments:
        args: Configurations to use
        """
        if isinstance(configurations, mlptrain.Configuration):
            configurations = [
                configurations
            ]  # Convert to list if it's a single Configuration
        elif not isinstance(configurations, mlptrain.ConfigurationSet):
            raise ValueError(
                f'Unsupported configuration type: {type(configurations)}'
            )
        # Dynamically initialize basis if needed
        if self.basis is None:
            if not self.elements:
                self.elements = list(
                    set(atom.label for c in configurations for atom in c.atoms)
                )
            self._initialize_basis()

        ace_vecs = []
        for conf in configurations:
            conf = conf.ase_atoms
            Main.eval(f'dataset = JuLIP.read_extxyz(IOBuffer("{conf}"))')
            descriptor = Main.eval(
                """
            descriptors = []
            for atoms in dataset
                per_atom_descriptors = []
                for i in 1:length(atoms)
                    push!(per_atom_descriptors, site_energy(basis, atoms, i))
                end
                push!(descriptors, per_atom_descriptors)
            end
            return descriptors
            """
            )
            ace_vecs.append(np.array(descriptor))

        ace_vecs = np.array(ace_vecs)
        return ace_vecs if ace_vecs.ndim > 1 else ace_vecs.reshape(1, -1)

    def kernel_vector(
        self,
        configuration: mlptrain.Configuration,
        configurations: mlptrain.ConfigurationSet,
        zeta: int = 4,
    ) -> np.ndarray:
        """
        Compute similarity kernel between configurations using ACE descriptors.

        Arguments:
            configuration: Single molecular structure.
            configurations: Set of molecular structures.
            zeta (int): Exponent in the kernel function.

        Returns:
            np.ndarray: Kernel similarity vector.
        """
        v1 = self.compute_representation(configuration)[0]
        m1 = self.compute_representation(configurations)
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        m1 /= np.linalg.norm(m1, axis=2, keepdims=True)

        per_atom_similarities = np.einsum(
            'ad,cad->ca', v1, m1
        )  # Compute per-atom kernel similarities
        structure_similarity = np.mean(
            per_atom_similarities, axis=1
        )  # Average per-atom similarities
        structure_similarity = np.power(structure_similarity, zeta)
        return structure_similarity
