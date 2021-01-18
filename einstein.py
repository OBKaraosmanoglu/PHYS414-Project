import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from decimal import Decimal
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

pi = np.pi
G = 6.67408e-11 # Gravitational constant in m^3 kg^-1 s^-2
m_solar = 1.989e30 # Solar mass in kg
c = 3e8 # m s^-1

geom_to_km = 1.477

def einsteinian():
    
    # The main function exists to prevent using an improper loop. It contains all of the graphs.
    def main(KNS):
        pc_orig = 1e-9
        
        
        # Solves the TOV equations for a given p_c value.
        def TOV_pc(pci):
            
            pc = pci
            
            def TOV(r, mvpmp):
                m, v, p, mp = mvpmp
                rho = np.sqrt(p/KNS)
                if r==0:
                    return [0, 0 ,0, 0]
                return [4 * pi * r**2 * rho,
                        2 * (m  +  4 * pi * r**3 * p)  /  (r * (r  -  2 * m)),
                        - (m  +  4 * pi * r**3 * p)  /  (r * (r  -  2 * m))  *  (rho + p),
                        4 * pi * r**2 * p  /  (np.sqrt(1  -  2 * m / r))]
            
            def terminate(r, mvpmp):
                return mvpmp[2]
            
            terminate.terminal = True
            terminate.direction = -1
            
            return solve_ivp(TOV, [0, 1e5], [0, 0, pc, 0], events=terminate)
        
        # number of different p_c samples to be tested.
        num_trial = int(1e2)
        
        pc = pc_orig
        
        masses = np.zeros((num_trial, 4))
        
        pc_upper_bound = 1
        
        # p_c's had to be geometrically spaced to avoid them accumulating near the low-radius zones.
        pcs = np.geomspace(pc_orig, pc_upper_bound, num=num_trial, endpoint=True)
        
        for i in range(0, num_trial):
            
            pc = pcs[i]
            
            rhoc = np.sqrt(pc/KNS)
            rhoc = rhoc * c**6  /  (m_solar**2 * G**3)
            
            
            sol = TOV_pc(pc)
            
            mass = sol.y[0, -1]
            radius = sol.t[-1] * geom_to_km
            
            baryonic_mass = sol.y[3, -1]
            
            delta = (baryonic_mass - mass) / mass
            
            masses[i] = [radius, mass, delta, rhoc]
        
        fig, axs = plt.subplots()
        
        TOV_mdata = masses[:, 1]
        TOV_rdata = masses[:, 0]
        
        axs.plot(TOV_rdata, TOV_mdata)
        axs.set_title("Mass v Radius for Neutron Stars (KNS = %d)" % KNS)
        axs.set_xlabel("Radius (km)")
        axs.set_ylabel("Mass (Solar Mass)")
        
        axs.grid()
        if KNS == 50:
            plt.savefig("NS_DatvPrd_50.jpg", dpi=150)
        elif KNS == 116:
            plt.savefig("NS_DatvPrd_116.jpg", dpi=150)
        elif KNS == 125:
            plt.savefig("NS_DatvPrd_125.jpg", dpi=150)
            
        fig, axs = plt.subplots()
        
        TOV_deltadata = masses[:, 2]
        
        axs.plot(TOV_rdata, TOV_deltadata)
        axs.set_title("Fractional Binding Energy vs Radius")
        axs.set_xlabel("Radius (km)")
        axs.set_ylabel("Fractional Binding Energy")
        
        axs.grid()
        if KNS == 50:
            plt.savefig("NS_FBE_50.jpg", dpi=150)
        elif KNS == 116:
            plt.savefig("NS_FBE_116.jpg", dpi=150)
        elif KNS == 125:
            plt.savefig("NS_FBE_125.jpg", dpi=150)
            
        fig, axs = plt.subplots()
        
        TOV_rhocdata = masses[:, 3]
        
        # This part separates the stable and unstable regions for varying plotting shapes.
        d = {'mass' : TOV_mdata, 'rhoc' : TOV_rhocdata}
        df = pandas.DataFrame(data = d)
        df.sort_values(by='rhoc', inplace=True)
        
        mrho = df.to_numpy()
        
        ms = mrho[:, 0]
        rhos = mrho[:, 1]
            
        stable_ms = np.array([ms[0]])
        
        for i in range(1, len(df.index) - 1):
            if (ms[i+1] - ms[i-1]) >= 0:
                stable_ms = np.append(stable_ms, ms[i])
        
        axs.plot(df['rhoc'], df['mass'], 'm--', label = "unstable region")
        axs.plot(rhos[:len(stable_ms)], stable_ms, 'm', label = 'stable region')
        axs.set_title(r"Mass vs $\rho_c$ (max. mass : %.2f $M_o$)" % np.max(stable_ms))
        axs.set_xlabel(r"$\rho_c$(kg $m^{-3}$)")
        axs.set_ylabel("Mass (solar mass)")
        
        axs.legend()
        axs.grid() 
        if KNS == 50:
            plt.savefig("NS_rho_50.jpg", dpi=150)
        elif KNS == 116:
            plt.savefig("NS_rho_116.jpg", dpi=150)
        elif KNS == 125:
            plt.savefig("NS_rho_125.jpg", dpi=150)
     

    main(50)
    
    for KNS in range(116, 126):
        main(KNS)
    
    
    