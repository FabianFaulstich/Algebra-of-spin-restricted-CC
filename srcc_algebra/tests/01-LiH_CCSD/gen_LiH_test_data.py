from pyscf import gto, scf, ao2mo, lib, cc

from functools import reduce

import numpy as np

def get_mo_integrals(mf):

    eri = ao2mo.restore(1, ao2mo.full(mf.mol, mf.mo_coeff), mf.mol.nao)
    h1e = lib.einsum('pi,pq,qj->ij', mf.mo_coeff, mf.get_hcore(), mf.mo_coeff)
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

    Mol = [['Li',[0,0,0]],['H',[1.4,0,0]]]

    mol = gto.M()
    mol.basis = basis
    mol.atom  = Mol
    mol.unit  = 'bohr'
    mol.verbose = 3
    mol.build()
  
    mf= mol.RHF()
    mf.kernel()

    h1e, eri = get_mo_integrals(mf)

    mycc = cc.CCSD(mf)
    mycc.kernel()

    t1, t2 = mycc.t1, mycc.t2

    ene = energy(mf, h1e, eri, t1, t2)

    #writing to file:
    np.save('h1e.npy', h1e)  
    np.save('eri.npy', eri) 
    np.save('t1.npy', h1e)  
    np.save('t2.npy', h1e) 
    np.save('HF_energy.npy', mf.e_tot)
    np.save('nuc_energy.npy', mol.get_enuc())
    np.save('CC_total_energy.npy', mycc.e_tot)
