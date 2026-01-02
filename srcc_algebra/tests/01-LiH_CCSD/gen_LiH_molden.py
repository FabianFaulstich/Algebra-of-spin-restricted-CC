from pyscf import gto, scf, ao2mo, lib, cc

from pyscf.tools import molden

from functools import reduce
import numpy as np

if __name__ == '__main__':

    np.set_printoptions(linewidth=300, suppress = True)

    basis= 'sto6g'
    bds = np.linspace(1, 3, num = 21) 
    bds = [1.4] 
    
    for bd in bds:
        Mol = [['Li',[0,0,0]],['H',[bd,0,0]]]

        mol = gto.M()
        mol.basis = basis
        mol.atom  = Mol
        mol.unit  = 'bohr'
        mol.verbose = 3
        mol.build()
      
        mf= mol.RHF()
        mf.kernel()

        breakpoint()
        exit()
        molden.from_mo(mol, f"molden/scf_{np.round(bd, 3)}.molden", mf.mo_coeff, occ=mf.mo_occ, ene=mf.mo_energy)

