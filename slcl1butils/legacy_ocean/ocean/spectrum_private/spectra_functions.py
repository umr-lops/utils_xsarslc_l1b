import numpy as np
import sys

def tony_omni(k, u10, us, omega):
    import numpy as np
    G = 9.81
    KM = 363.2
    c = np.sqrt(G/k*(1.+(k/KM)**2))
    cp = u10/omega
    cm = np.sqrt(2*G/KM)
    k0 = G/u10**2
    kp = k0*omega**2

    #~ Peak enhancement according to Jonswap
    sigma = 0.08*(1+4./omega**3)
    gamma = 1.7
    if omega>1.:
        gamma=1.7+6.*np.log10(omega)
    LPM = np.exp(-(5./4.)*(kp/k)**2)
    Jp=gamma**(np.exp(-(np.sqrt(k/kp)-1)**2/2./sigma/sigma))

    #~ Curvature spectrum of Long waves
    alpha_p=6.0e-3*np.sqrt(omega)
    Fp=LPM*Jp*np.exp(-omega/np.sqrt(10.)*(np.sqrt(k/kp)-1.))
    Bl=0.5*alpha_p*cp/c*Fp

    #~ Curvature spectrum of Short waves

    if us < cm:
        alpha_m=1e-2*(1+np.log(us/cm))
    else:
        alpha_m=1e-2*(1.+3.*np.log(us/cm))
    alpha_m=max(0.,alpha_m)
    Fm=np.exp(-(k/KM-1)**2/4.)
    Bh=0.5*alpha_m*cm/c*Fm

    return Bl+LPM*Bh

def tony_omni_shortwaves(k, u10, us, omega):
    import numpy as np
    G = 9.81
    KM = 363.2
    c = np.sqrt(G/k*(1.+(k/KM)**2))
    cm = np.sqrt(2*G/KM)
    kp = k0*omega**2
    LPM = np.exp(-(5./4.)*(kp/k)**2)

    #~ Curvature spectrum of Short waves
    if us < cm:
        alpha_m=1e-2*(1+np.log(us/cm))
    else:
        alpha_m=1e-2*(1.+3.*np.log(us/cm))
    Fm=np.exp(-(k/KM-1)**2/4.)
    Bh=0.5*alpha_m*cm/c*Fm
    return LPM*Bh


def tony_spread(k, u10, omega):
    import numpy as np
    G = 9.81
    KM = 363.2
    CD10=1e-3*(0.8+0.065*u10)
    us=np.sqrt(CD10)*u10
    cp=u10/omega
    c=np.sqrt(G/k*(1.+(k/KM)**2));
    cm=np.sqrt(2.*G/KM)
    a0=np.log(2.)/4.
    ap=4
    am=0.13*us/cm
    return np.tanh(a0+ap*(c/cp)**2.5+am*(cm/c)**2.5)

def findUstar(wind, height):
    import numpy as np
    ustar = 0.25
    epsilon = 1e-7
    h = height
    uh = wind
    xn = ustar
    dx = 2*epsilon
    while abs(dx)>epsilon:
        yn = xn*np.log(h/(1.5e-3*xn**2))/0.4
        dn = (np.log(h/(1.5e-3*xn**2))-2.0)/0.4
        dx = (uh-yn)/dn
        xn = xn+dx
    return xn

def kudry_spectrum(kx, ky, U10, fetch):
    #NG from shared.my_functions import cart2pol

    kudry_spec, kudry_omni = kudry_to_gridded(U10, fetch)
    TH,K=cart2pol(kx,ky)
    spec, up, cross = np.zeros(TH.shape), np.zeros(TH.shape), np.zeros(TH.shape)
    for i in np.arange(0, TH.shape[0]):
        for j in np.arange(0, TH.shape[1]):
            spec[i,j] = kudry_spec(TH[i,j], K[i,j])
            up[i,j] = kudry_spec(0, K[i,j])
            cross[i,j] = kudry_spec(np.pi/2, K[i,j])


    spreading=(up-cross)/(up+cross)
    curvature = kudry_omni(K) if (K.shape[0]==1) or (K.shape[1]==1) else np.nan

    spec=spec/(K**4)

    return curvature, spreading, spec


def tun_fs(Nk,Nphi,kgam,k,ng,nc,a):
    from scipy.special import erf
    # FILTER AND TUNING FUNCTIONS
    p = k/kgam
    pl = 2. #1.5;# it is recomended pl=2 after Moiche et al comparison with radar data
    ph = 10.
    sl = 2.
    sh = 2. #0.5
    U1 = 1./2.*(erf(sl*(p-pl))+1)

    U1 = 0.5*(erf(sl*(p-pl))+1.);
    U2 = 0.5*(erf(sh*(p-ph))+1.);
    F = np.maximum(0,U1-U2)
    FF = np.outer(np.ones(Nphi), F)

    p = kgam/k
    U1 = 0.5*(erf(sl*(p-pl))+1.)
    U2 = 0.5*(erf(sh*(p-ph))+1.)
    Fg = U1-U2

    inte = k*Fg
    f = np.zeros(inte.shape)

    for j in np.arange(f.shape[0]-1):
        f[j+1]=f[j]+inte[j]

    f = f/f[-1]
    n0 = np.outer(np.ones(Nphi), 1./((1.-1./ng)*f+1./ng))
    avCbeta = 0.03

    lna = np.log(a)-np.log(avCbeta)*1./n0
    alpha = np.exp(lna)

    return FF, alpha, n0

def kudry_to_gridded(u10, fetch):
    from scipy.interpolate import interp2d, interp1d
    rho = 1e3/1.3
    ck = 0.41
    g = 9.81
    T = 74e-6
    Vw = 1.0e-6
    Va = 14e-6
    H = 10. #reference level for wind
    Cst=1.5
    Cst = Cst/rho
    #~ spectrum model constants
    nc =1.
    ng = 5.
    a = 2.5e-3

    #~ GRID
    Nk = 400
    Nph = 48
    kmin = 2.*np.pi/600. # 0.01
    kmax =1e4# 2*np.pi/0.001
    dlnk = (np.log(kmax) - np.log(kmin))/(Nk - 1)
    k = kmin*np.exp(np.arange(Nk)*dlnk)
    dthet = 2*np.pi/Nph
    theta= np.arange(-np.pi, np.pi, dthet)
    Nphi=theta.shape[0]


    w = np.sqrt(g*k + T*k**3)
    c = w/k
    kgam = np.sqrt(g/T)
    ikgam = np.round(np.log(kgam/kmin)/dlnk)
    kres = g/T/k #wavenumbers of waves generating parasitic capillaries

    ikres = np.round(np.log(kres/kmin)/dlnk) #indexes of these waves
    ikres = np.int_(np.minimum(ikres, Nk-1))

    FF, alpha, n0 = tun_fs(Nk, Nphi, kgam, k, ng, nc, a) #sub-routine calculating filter-function, alpha- and n-functions

    #-------------------------------------------------------------------------------

    const = 0.018 #Charnok constant

    ustar= np.sqrt(1e-3)*u10 #first guess for ustar
    for it in np.arange(5):
        z0 = const*ustar**2/g + 0.1 *Va/ustar
        ustar = ck*u10/np.log(H/z0)
        #UST(it)=ustar;

    CD10 = (ustar/u10)**2
    Z0 = z0

    # SPECTRUM
    ftch = 9.8*fetch/u10**2
    age = 2*np.pi*3.5*ftch**(-0.33)
    age = max(0.83,age)
    AGE = age
    Cp  = u10/age
    kp = g/Cp**2 # spectral peak
    ikp = round(np.log(kp/kmin)/dlnk)

    #LOW FREQUENCY  Donelan et al (1985) SPECTRUM
    hlow = np.zeros(Nk)
    for i in np.arange(Nk):
        if (k[i]/kp < .31):
            hlow[i] = 1.24
        elif (k[i]/kp < .90):
            hlow[i] = 2.61*(k[i]/kp)**(.65);
        else:
            hlow[i] = 2.28*(k[i]/kp)**(-.65)

    fphi = 0.5*np.outer(np.ones(Nphi),hlow)/(np.cosh(np.outer(theta, hlow))**2)

    al = 6e-3 * age**0.55
    si = 0.08 * (1 + 4 * age**(-3))
    ga = 1.7 + 6*np.log10(max(1,age))

    p = age/np.sqrt(10.)
    Cut = np.exp(-p*np.maximum(0,(np.sqrt(k/kp)-1))) #Cut-off function according to Elfouhaily etal (1997)
    G = np.exp(-(np.sqrt(k/kp) - 1)**2 /(2*si**2))
    Blow1 =0.5*al*np.sqrt(k/kp)*np.exp(-(kp/k)**2)*ga**G*Cut
    Blow = np.outer(np.ones(Nphi), Blow1)*fphi

    #SHORT WIND WAVE SPECTRUM

    diss1 = 4.*Vw*k**2/w
    #diss  = ones(Nphi,)*diss1;
    zgen = np.pi/k
    angbeta = (np.exp(-theta**2)).T

    Cbeta=np.maximum(0,1/ck*np.log(1/z0*zgen))
    Cbeta=Cst*np.maximum(0.,Cbeta-c/ustar)
    beta1 = Cbeta*(ustar/c)**2
    #betaphi = angbeta* beta1
    grow=np.outer(angbeta, beta1-diss1)

    Brip0 = alpha*(np.maximum(0,grow))**(1 /n0) #short wave spectrum without parasitic capillaries
    Bgen = Blow+Brip0*np.outer(np.ones(Nphi), 1-Cut) #spectrum of waves generating parasitic capillaries
    Ipc = Bgen*np.maximum(0,grow)

    Ipc = Ipc[:,ikres]*FF
    Brip = grow**2+4./alpha*Ipc


    Brip = ((grow + Brip**0.5 )/2.)**(1./n0)
    Brip = alpha*Brip

    #Full Spectrum
    B2 = Blow+Brip*np.outer(np.ones(Nphi), 1.-Cut)
    B2 = np.maximum(0,B2)

    B1 = np.trapz(B2, axis=0)*dthet

    #mssup = dthet*dlnk*trapz(cos(theta)**2*trapz(B2'));
    #msscr = dthet*dlnk*trapz(sin(theta)**2*trapz(B2'));
    mssup = dthet*np.vdot(dlnk, np.trapz(np.cos(theta)**2*np.trapz(Brip)))
    msscr = dthet*np.vdot(dlnk, np.trapz(np.sin(theta)**2*np.trapz(Brip)))

    MSSUP=mssup
    MSSCR=msscr
    MSS=msscr+mssup
    BUP=B2[(Nphi+1)/2-1,:] # UP-WIND saturation spectrum
    BOM=B1 # OMNI-DIRECTIONAL saturation spectrum

    # Calculation of fraction of wavebreaking zone
    kwb = 2*np.pi/0.15 #wavenumber of shortest breaking waves. As discussed in JGR2003, for radar applications the upper limit of integration is knb=min(0.1*kradar,kwb)
    cq = 10.5; #tuning constant for breaking zones from JGR2003
    dQ = 0.5/alpha*grow*B2

    #~ Q = cq*dthet*np.vdot(dlnk, np.outer(np.trapz((k<kwb), np.trapz(dQ))))

    kudry_omni = interp1d(k, B1)
    kudry_spec = interp2d(theta, k, B2)

    return kudry_spec, kudry_omni





def kudryavtsev(k, theta, dthet, u10, fetch):

    rho = 1e3/1.3
    ck = 0.41
    g = 9.81
    T = 74e-6
    Vw = 1.0e-6
    Va = 14e-6
    H = 10. #reference level for wind
    Cst=1.5
    Cst = Cst/rho
    #~ spectrum model constants
    nc =1.
    ng = 5.
    a = 2.5e-3

    #~ GRID
    #~ Nk = 400
    #~ Nph = 48
    #~ kmin = 2.*np.pi/600. # 0.01
    #~ kmax =1e4# 2*np.pi/0.001

    #~ k = kmin*np.exp(np.arange(Nk)*dlnk)
    #~ dthet = 2*np.pi/Nph
    #~ theta= np.arange(-np.pi, np.pi, dthet)
    #~ Nphi=theta.shape[0]

    Nk = len(k)
    Nphi = len(theta)
    kmin = k.min()
    kmax = k.max()
    dlnk = (np.log(kmax) - np.log(kmin))/(Nk - 1)

    w = np.sqrt(g*k + T*k**3)
    c = w/k
    kgam = np.sqrt(g/T)
    ikgam = np.round(np.log(kgam/kmin)/dlnk)
    kres = g/T/k #wavenumbers of waves generating parasitic capillaries

    ikres = np.round(np.log(kres/kmin)/dlnk) #indexes of these waves
    ikres = np.int_(np.minimum(ikres, Nk-1))

    FF, alpha, n0 = tun_fs(Nk, Nphi, kgam, k, ng, nc, a) #sub-routine calculating filter-function, alpha- and n-functions

    #-------------------------------------------------------------------------------

    const = 0.018 #Charnok constant

    ustar= np.sqrt(1e-3)*u10 #first guess for ustar
    for it in np.arange(5):
        z0 = const*ustar**2/g + 0.1 *Va/ustar
        ustar = ck*u10/np.log(H/z0)
        #UST(it)=ustar;

    CD10 = (ustar/u10)**2
    Z0 = z0

    # SPECTRUM
    ftch = 9.8*fetch/u10**2
    age = 2*np.pi*3.5*ftch**(-0.33)
    age = max(0.83,age)
    AGE = age
    Cp  = u10/age
    kp = g/Cp**2 # spectral peak

    ikp = round(np.log(kp/kmin)/dlnk)

    #LOW FREQUENCY  Donelan et al (1985) SPECTRUM
    hlow = np.zeros(Nk)
    for i in np.arange(Nk):
        if (k[i]/kp < .31):
            hlow[i] = 1.24
        elif (k[i]/kp < .90):
            hlow[i] = 2.61*(k[i]/kp)**(.65);
        else:
            hlow[i] = 2.28*(k[i]/kp)**(-.65)

    fphi = 0.5*np.outer(np.ones(Nphi),hlow)/(np.cosh(np.outer(theta, hlow))**2)

    al = 6e-3 * age**0.55
    si = 0.08 * (1 + 4 * age**(-3))
    ga = 1.7 + 6*np.log10(max(1,age))

    p = age/np.sqrt(10.)
    Cut = np.exp(-p*np.maximum(0,(np.sqrt(k/kp)-1))) #Cut-off function according to Elfouhaily etal (1997)
    G = np.exp(-(np.sqrt(k/kp) - 1)**2 /(2*si**2))
    Blow1 =0.5*al*np.sqrt(k/kp)*np.exp(-(kp/k)**2)*ga**G*Cut
    Blow = np.outer(np.ones(Nphi), Blow1)*fphi

    #SHORT WIND WAVE SPECTRUM

    diss1 = 4.*Vw*k**2/w
    #diss  = ones(Nphi,)*diss1;
    zgen = np.pi/k
    angbeta = (np.exp(-theta**2)).T

    Cbeta=np.maximum(0,1/ck*np.log(1/z0*zgen))
    Cbeta=Cst*np.maximum(0.,Cbeta-c/ustar)
    beta1 = Cbeta*(ustar/c)**2
    #betaphi = angbeta* beta1
    grow=np.outer(angbeta, beta1-diss1)

    Brip0 = alpha*(np.maximum(0,grow))**(1 /n0) #short wave spectrum without parasitic capillaries
    Bgen = Blow+Brip0*np.outer(np.ones(Nphi), 1-Cut) #spectrum of waves generating parasitic capillaries
    Ipc = Bgen*np.maximum(0,grow)

    Ipc = Ipc[:,ikres]*FF
    Brip = grow**2+4./alpha*Ipc


    Brip = ((grow + Brip**0.5 )/2.)**(1./n0)
    Brip = alpha*Brip

    #Full Spectrum
    B2 = Blow+Brip*np.outer(np.ones(Nphi), 1.-Cut)
    B2 = np.maximum(0,B2)

    B1 = np.trapz(B2, axis=0)*dthet

    #mssup = dthet*dlnk*trapz(cos(theta)**2*trapz(B2'));
    #msscr = dthet*dlnk*trapz(sin(theta)**2*trapz(B2'));
    mssup = dthet*np.vdot(dlnk, np.trapz(np.cos(theta)**2*np.trapz(Brip)))
    msscr = dthet*np.vdot(dlnk, np.trapz(np.sin(theta)**2*np.trapz(Brip)))

    MSSUP=mssup
    MSSCR=msscr
    MSS=msscr+mssup
    BUP=B2[(Nphi+1)/2-1,:] # UP-WIND saturation spectrum
    BOM=B1 # OMNI-DIRECTIONAL saturation spectrum

    # Calculation of fraction of wavebreaking zone
    kwb = 2*np.pi/0.15 #wavenumber of shortest breaking waves. As discussed in JGR2003, for radar applications the upper limit of integration is knb=min(0.1*kradar,kwb)
    cq = 10.5; #tuning constant for breaking zones from JGR2003
    dQ = 0.5/alpha*grow*B2

    #~ Q = cq*dthet*np.vdot(dlnk, np.outer(np.trapz((k<kwb), np.trapz(dQ))))

    kudry_omni = B1
    kudry_spec = B2

    return kudry_spec
