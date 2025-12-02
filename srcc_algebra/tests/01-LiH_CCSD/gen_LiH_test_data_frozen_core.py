from pyscf import gto, scf, ao2mo, lib, cc

from functools import reduce

import numpy as np

def get_mo_integrals(mf, act_orbs):

    C = mf.mo_coeff[:, act_orbs]
    eri_ao = mf.mol.intor('int2e')
    eri = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, C, C, C, C)
    h1e = lib.einsum('pi,pq,qj->ij', C, mf.get_hcore(), C)
    
    return h1e, eri

def energy(mf, h1e, eri, t1, t2):

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    fockao = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    fock = reduce(np.dot, (mf.mo_coeff.T, fockao, mf.mo_coeff))

    e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    tau = np.einsum('ia,jb->ijab',t1,t1)
    tau += t2

    o = slice(0, nocc)
    v = slice(nocc, nmo)

    eris_ovov = eri[o, v, o, v]
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    return e.real

if __name__ == '__main__':

    np.set_printoptions(linewidth=300, suppress = True)

    basis= 'sto6g'

    #opt = 1 is 2 electron pairs in 4 spatial orbitals 
    #opt = 2 is 1 electron pair in 5 spatial orbitals
    opt = 1

    n_steps = 10
    bds = np.linspace(1, 3, num = n_steps)

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

        if opt == 1:
            #2 electron pairs in 4 spatial orbitals
            act_orbs = [0,1,2,5]
        elif opt == 2:
            #1 electron pair in 5 spatial orbitals
            act_orbs = [1,2,3,4,5]

        h1e, eri = get_mo_integrals(mf, act_orbs)

        #writing to file:
        np.save(f'h1e_{opt}_{np.round(bd, 3)}.npy', h1e)  
        np.save(f'eri_{opt}_{np.round(bd, 3)}.npy', eri) 
        np.save(f'HF_energy_{opt}_{np.round(bd, 3)}.npy', mf.e_tot)
        np.save(f'nuc_energy.npy_{opt}_{np.round(bd, 3)}', mol.get_enuc())
