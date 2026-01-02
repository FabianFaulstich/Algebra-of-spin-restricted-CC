from pyscf import gto, lib

from functools import reduce

import numpy as np

def get_mo_integrals(mf, act_orbs):

    C = mf.mo_coeff[:, act_orbs]
    eri_ao = mf.mol.intor('int2e')
    eri = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, C, C, C, C)
    h1e = lib.einsum('pi,pq,qj->ij', C, mf.get_hcore(), C)
    
    return h1e, eri

if __name__ == '__main__':

    np.set_printoptions(linewidth=300, suppress = True)

    basis= 'sto6g'

    Mol = [['O', [0.0000, 0.0000, 0.1173]],
           ['H', [0.0000, 0.7572,-0.4692]],
           ['H', [0.0000,-0.7572,-0.4692]]]

    mol = gto.M()
    mol.basis = basis
    mol.atom  = Mol
    mol.unit  = 'angstrom'
    mol.verbose = 3
    mol.build()
  
    mf= mol.RHF()
    mf.kernel()

    S = mf.get_ovlp()
    nocc = np.count_nonzero(mf.mo_occ > 0)

    core_aos = mol.search_ao_label('O 1s')   # -> [0] in your case

    weights = []
    for i in range(nocc):
        c = mf.mo_coeff[:, i]
        Sc = S @ c
        w = float(c[core_aos] @ Sc[core_aos])
        weights.append(w)

    core_mo = int(np.argmax(weights))
    print("core_mo index:", core_mo)
    print("core weight:", weights[core_mo])
    print("core energy:", mf.mo_energy[core_mo])

    act_orbs = [1,2,3,4,5,6]
   
    h1e, eri = get_mo_integrals(mf, act_orbs)

    #writing to file:
    np.save(f'h1e.npy', h1e)  
    np.save(f'eri.npy', eri) 
    np.save(f'HF_energy.npy', mf.e_tot)

    enuc = mol.get_enuc()
    
    np.save(f'E_nuc.npy', enuc)
