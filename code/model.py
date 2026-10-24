"""Four-state oxygen-regulated FUCCI model.

    dR/dt = 2 p k2 G - kappa(N) R - alpha(N) R + beta(N) Q - muR(N) R
    dG/dt = kappa(N) R - k2 G - muG(N) G
    dQ/dt = alpha(N) R + 2 (1-p) k2 G - beta(N) Q - muQ(N) Q
    dN/dt = S0 / (1 + delta P) - gamma N - (lamR R + lamG G + lamQ Q) N

Here P = R + G + Q, kappa(N) = (1-u1) k1bar N/(KN+N), alpha(N) = albar Kal^m/(Kal^m+N^m) (m = nal),
beta(N) = bebar N^n/(Kbe^n + N^n), muX(N) = muX0 + muXmax KmuX/(KmuX+N) (+ u2 etaG for G).

The state order is y = (R, G, Q, N), and time is measured in hours.
"""
from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Params:
    # Oxygen
    S0: float = 1.0
    gamma: float = 1.0
    delta: float = 0.02
    lamR: float = 0.1445
    lamG: float = 0.1456
    lamQ: float = 0.0096
    # Cell-cycle progression
    k1bar: float = 0.5045
    KN: float = 0.2526
    k2: float = 0.1356
    p: float = 0.8816
    # Quiescence
    albar: float = 0.2078
    Kal: float = 0.1178
    nal: float = 1.0      # Entry Hill exponent
    bebar: float = 0.1744
    Kbe: float = 0.4086
    n: float = 2.0
    # Loss rates
    muR0: float = 0.0044
    muRmax: float = 0.0615
    KmuR: float = 0.2154
    muG0: float = 0.0078
    muGmax: float = 0.0741
    KmuG: float = 0.0709
    muQ0: float = 0.0
    muQmax: float = 0.0842
    KmuQ: float = 0.1354
    etaG: float = 0.20
    # Controls
    u1: float = 0.0
    u2: float = 0.0

    def copy(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)

    @property
    def Nhat(self):
        return self.S0 / self.gamma


# Response functions
def kappa(N, pr):
    return (1.0 - pr.u1) * pr.k1bar * N / (pr.KN + N)


def alpha(N, pr):
    Km = pr.Kal ** pr.nal
    return pr.albar * Km / (Km + np.power(N, pr.nal))


def beta(N, pr):
    Nn = np.power(N, pr.n)
    return pr.bebar * Nn / (pr.Kbe ** pr.n + Nn)


def muR(N, pr):
    return pr.muR0 + pr.muRmax * pr.KmuR / (pr.KmuR + N)


def muG(N, pr):
    return pr.muG0 + pr.muGmax * pr.KmuG / (pr.KmuG + N) + pr.u2 * pr.etaG


def muQ(N, pr):
    return pr.muQ0 + pr.muQmax * pr.KmuQ / (pr.KmuQ + N)


# Vector field
def rhs(t, y, pr):
    R, G, Q, N = y
    P = R + G + Q
    ka, al, be = kappa(N, pr), alpha(N, pr), beta(N, pr)
    dR = 2 * pr.p * pr.k2 * G - ka * R - al * R + be * Q - muR(N, pr) * R
    dG = ka * R - pr.k2 * G - muG(N, pr) * G
    dQ = al * R + 2 * (1 - pr.p) * pr.k2 * G - be * Q - muQ(N, pr) * Q
    dN = pr.S0 / (1 + pr.delta * P) - pr.gamma * N - (pr.lamR * R + pr.lamG * G + pr.lamQ * Q) * N
    return np.array([dR, dG, dQ, dN])


def jacobian(y, pr, eps=1e-7):
    y = np.asarray(y, float)
    J = np.zeros((4, 4))
    for j in range(4):
        h = eps * max(1.0, abs(y[j]))
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        J[:, j] = (rhs(0, yp, pr) - rhs(0, ym, pr)) / (2 * h)
    return J


# Cellular transition matrix
def A(N, pr):
    """Cellular Metzler matrix at fixed oxygen, ordered as (R, G, Q)."""
    ka, al, be = kappa(N, pr), alpha(N, pr), beta(N, pr)
    return np.array([[-(ka + al + muR(N, pr)), 2 * pr.p * pr.k2, be],
                     [ka, -(pr.k2 + muG(N, pr)), 0.0],
                     [al, 2 * (1 - pr.p) * pr.k2, -(be + muQ(N, pr))]])


def growth_exponent(pr, N=None):
    """Return the spectral abscissa of A(N), with N = Nhat by default."""
    N = pr.Nhat if N is None else N
    return float(np.max(np.linalg.eigvals(A(N, pr)).real))


def R0_factors(pr, N=None):
    """Return (pi_S, pi_M, pi_ret, R0) at oxygen N (Nhat by default)."""
    N = pr.Nhat if N is None else N
    ka, al, be = kappa(N, pr), alpha(N, pr), beta(N, pr)
    c = be + muQ(N, pr)
    piS = ka / (ka + muR(N, pr) + al * muQ(N, pr) / c)
    piM = pr.k2 / (pr.k2 + muG(N, pr))
    piret = pr.p + (1 - pr.p) * be / c
    return piS, piM, piret, 2 * piS * piM * piret


def R0(pr, N=None):
    return R0_factors(pr, N)[3]


def equilibrium(pr):
    """Return the positive equilibrium, or None when R0(Nhat) <= 1."""
    Nh = pr.Nhat
    if R0(pr) <= 1.0:
        return None
    Ns = brentq(lambda N: R0(pr, N) - 1.0, 1e-12, Nh * (1 - 1e-14), xtol=1e-15, rtol=1e-14)
    phiG = kappa(Ns, pr) / (pr.k2 + muG(Ns, pr))
    phiQ = (alpha(Ns, pr) + 2 * (1 - pr.p) * pr.k2 * phiG) / (beta(Ns, pr) + muQ(Ns, pr))
    v = np.array([1.0, phiG, phiQ])
    v /= v.sum()
    ell = pr.lamR * v[0] + pr.lamG * v[1] + pr.lamQ * v[2]
    a2 = pr.delta * ell * Ns
    a1 = ell * Ns + pr.delta * pr.gamma * Ns
    a0 = -(pr.S0 - pr.gamma * Ns)
    P = (-a1 + np.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2) if a2 > 0 else -a0 / a1
    return np.array([*(P * v), Ns])


def u1_star(pr):
    """Return the u1 extinction threshold at fixed u2, if it exists."""
    if R0(pr.copy(u1=0.0)) <= 1:
        return 0.0
    if R0(pr.copy(u1=0.999999)) > 1:
        return None
    return brentq(lambda u: R0(pr.copy(u1=u)) - 1.0, 0.0, 0.999999, xtol=1e-13)
