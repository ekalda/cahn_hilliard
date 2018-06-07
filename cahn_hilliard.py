import numpy as np
import matplotlib.pyplot as plt
import simplejson as json
from collections import OrderedDict
import ast
from pylab import cm

class CahnHilliard(object):
    def __init__(self, a=0.1, b=0.1, x_dim=50, y_dim=50, dt=0.1, dx=1, phi0=0, M=0.1, k=0.1, anim=True, free_energy=False, out_file = None):
        self.a = a
        self.b = b
        self.x_dim = x_dim # x dim of a system
        self.y_dim = y_dim # y  dim of a system
        self.dt = dt # time step size
        self.dx = dx # distance between spacial points
        self.phi0 = phi0 # initial state
        self.M = M
        self.k = k
        self.sweep = self.x_dim * self.y_dim
        self.system = self.create_system()
        self.temp = np.zeros((self.y_dim, self.x_dim))
        self.anim = anim
        self.free_energy = free_energy
        self.out_file = out_file
        self.f_vals = []

    def create_system(self):
        system = np.ndarray((self.y_dim, self.x_dim))
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                # add some noise to the initial phi value
                noise = np.random.uniform(-0.001,0.001)
                system[j, i] = self.phi0 + noise
        return system

    # finding mu using periodic boundary conditions
    def find_mu(self, y, x):
        x_len = self.x_dim
        y_len = self.y_dim
        phi = self.system[y, x]
        phi_up = self.system[(y - 1 + y_len) % y_len, x]
        phi_down = self.system[(y + 1) % y_len, x]
        phi_right = self.system[y, (x + 1) % x_len]
        phi_left = self.system[y, (x - 1 + x_len) % x_len]
        mu = -self.a*phi + self.b*(phi**3) - (self.k/(self.dx**2))*(phi_up + phi_down + phi_left + phi_right - 4*phi)
        return mu

    # using mu to perform a time update on one lattice site (using periodic boundary conditions)
    def update_phi(self, y, x):
        x_len = self.x_dim
        y_len = self.y_dim
        mu = self.find_mu(y, x)
        mu_up = self.find_mu((y - 1 + y_len) % y_len, x)
        mu_down = self.find_mu((y + 1) % y_len, x)
        mu_right = self.find_mu(y, (x + 1) % x_len)
        mu_left = self.find_mu(y, (x - 1 + x_len) % x_len)
        self.temp[y, x] = self.system[y, x] + (self.M*self.dt/(self.dx**2))*(mu_down + mu_left + mu_right + mu_up - 4*mu)

    # calculate free energy per site
    def find_f_per_site(self, y, x):
        x_len = self.x_dim
        y_len = self.y_dim
        phi = self.system[y, x]
        phi_up = self.system[(y - 1 + y_len) % y_len, x]
        phi_down = self.system[(y + 1) % y_len, x]
        phi_right = self.system[y, (x + 1) % x_len]
        phi_left = self.system[y, (x - 1 + x_len) % x_len]
        return -(self.a/2.)*(phi**2) + (self.a/4)*(phi**4) + (self.k/(4*self.dx))*(phi_up - phi_down + phi_right - phi_left)

    # free energy of a whole system
    def find_total_f(self):
        f = 0
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                f += self.find_f_per_site(y, x)
        return f/float(self.sweep)

    # advance time by dt for the whole system
    def update_sys(self):
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                self.update_phi(j, i)

    def animate_ch(self):
        im = plt.imshow(self.system, cmap=cm.summer)
        plt.ion()
        plt.show()
        plt.pause(0.00001)
        plt.cla()

    def run(self):
        for n in range(100*self.sweep):
            self.update_sys()
            if (n % (self.sweep*1)) == 0 and self.anim:
                self.animate_ch()
            self.system = self.temp
            if self.free_energy and (n % self.sweep) == 0:
                with open(self.out_file, 'a+') as f:
                    f.write(str(n) + ' ' + str(self.find_total_f()) + '\n')


def main():
    # reading data from the input file
    print('reading in the input...')
    with open('input_ch.dat', 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
        x_dim = config["x dim"]
        y_dim = config["y dim"]
        a = config["a"]
        b = config["b"]
        k = config["k"]
        M = config["M"]
        phi0 = config["phi0"]
        dx = config["dx"]
        dt = config["dt"]
        anim = ast.literal_eval(config["animate"])
        measure_f = ast.literal_eval(config["measure free energy"])
        out_file = config["data file"]

        print('setting up the simulation...')
        sim = CahnHilliard(a=a, b=b, x_dim=x_dim, y_dim=y_dim, dt=dt, dx=dx, phi0=phi0, M=M, k=k, anim=anim, free_energy=measure_f, out_file=out_file)
        print("running the simulation...")
        sim.run()


if __name__ == "__main__":
    main()

