import numpy as np
from tqdm.auto import tqdm
import copy
from src.utils.leadlag import leadlag
from esig import tosig

class Oganism:
    def __init__(self, n_points, price_pip, spot_vol_pip, price_n_pips, spot_vol_n_pips):
        self.n_points = n_points
        self.price_pip = price_pip
        self.spot_vol_pip = spot_vol_pip
        self.price_n_pips = price_n_pips
        self.spot_vol_n_pips = spot_vol_n_pips

        # Initialise
        self.randomise()

    def __add__(self, other):
        """Breed."""
        
        derivative_list = zip(self.price_derivatives, other.price_derivatives, self.spot_vol_derivatives, other.spot_vol_derivatives)
        price_derivatives = []
        spot_vol_derivatives = []
        for p_deri_1, p_deri_2, sv_deri_1, sv_deri_2 in derivative_list:
            if np.random.random() < 0.5:
                price_derivatives.append(p_deri_1)
                spot_vol_derivatives.append(sv_deri_1)
            else:
                price_derivatives.append(p_deri_2)
                spot_vol_derivatives.append(sv_deri_2)

        

        prices = np.r_[0., np.cumsum(price_derivatives)]
        spot_vols = np.r_[0., np.cumsum(spot_vol_derivatives)]
        price_path = leadlag(prices)
        spot_voL_path = leadlag(spot_vols)

        o = Oganism(self.n_points, self.price_pip, self.spot_vol_pip, self.price_n_pips, self.spot_vol_n_pips)
        
        o.price_derivatives = price_derivatives
        o.spot_vol_derivatives = spot_vol_derivatives

        o.set_path(price_path, spot_voL_path)

        return o

    def randomise(self):
        self.price_derivatives = np.array([self.random_price_derivative() for _ in range(self.n_points - 1)])
        prices = np.r_[0., self.price_derivatives.cumsum()]

        self.spot_vol_derivatives = np.array([self.random_spot_vol_derivative() for _ in range(self.n_points - 1)])
        spot_vols = np.r_[0., self.spot_vol_derivatives.cumsum()]

        price_path = leadlag(prices)
        spot_vol_path = leadlag(spot_vols)

        self.set_path(price_path, spot_vol_path)

    def random_price_derivative(self):
        r = self.price_pip * np.random.randint(-self.price_n_pips, self.price_n_pips)

        return r

    def random_spot_vol_derivative(self):
        r = self.spot_vol_pip * np.random.randint(-self.spot_vol_n_pips, self.spot_vol_n_pips)

        return r

    def set_path(self, price_path, spot_vol_path):
        self.price_path = price_path
        self.spot_vol_path = spot_vol_path

    def mutate(self, prob=0.1):
        for i in range(len(self.price_derivatives)):
            if np.random.random() < prob:
                self.price_derivatives[i] = self.random_price_derivative()

        for i in range(len(self.spot_vol_derivatives)):
            if np.random.random() < prob:
                self.spot_vol_derivatives[i] = self.random_spot_vol_derivative()

        prices = np.r_[0., np.cumsum(self.price_derivatives)]
        spot_vols = np.r_[0., np.cumsum(self.spot_vol_derivatives)]
        
        price_path = leadlag(prices)
        spot_vol_path = leadlag(spot_vols)

        self.set_path(price_path, spot_vol_path)


    def logsignature(self, order):
        self.concat_path = np.c_[price_path, spot_vol_path]
        return tosig.stream2logsig(self.concat_path, order)

    def loss(self, sig, order):
        diff = np.abs((sig - self.logsignature(order)) / sig)
        diff /= 1 + np.arange(len(sig))
        return np.mean(diff)
    
class Population:
    def __init__(self, n_organisms, n_points, price_pip, spot_vol_pip, price_n_pips, spot_vol_n_pips):
        self.n_points = n_points
        self.price_pip = price_pip
        self.spot_vol_pip = spot_vol_pip
        self.price_n_pips = price_n_pips
        self.spot_vol_n_pips = spot_vol_n_pips
        self.n_organisms = n_organisms

        self.organisms = [Oganism(n_points, price_pip, spot_vol_pip, price_n_pips, spot_vol_n_pips) for _ in range(n_organisms)]

    def fittest(self, sig, p, order):
        n = int(len(self.organisms) * p)
        return sorted(self.organisms, key=lambda o: o.loss(sig, order))[:n]
        
    def evolve(self, sig, p, order, mutation_prob=0.1):
        parents = self.fittest(sig, p, order)
        new_generation = copy.deepcopy(parents)

        while len(new_generation) != self.n_organisms:
            i = j = 0
            while i == j:
                i, j = np.random.choice(range(len(parents)), size=2)
                parent1, parent2 = parents[i], parents[j]

            child = parent1 + parent2
            child.mutate(prob=mutation_prob)
            
            new_generation.append(copy.deepcopy(child))

        self.organisms = new_generation

        # Return loss
        return new_generation[0].loss(sig, order)

def train(sig, order, n_iterations, n_organisms, n_points, price_pip, spot_vol_pip, price_n_pips, spot_vol_n_pips, top_p=0.1, mutation_prob=0.1):
    population = Population(n_organisms, n_points, price_pip, spot_vol_pip, price_n_pips, spot_vol_n_pips)
    pbar = tqdm(range(n_iterations))

    for _ in pbar:
        loss = population.evolve(sig, p=top_p, order=order, mutation_prob=mutation_prob)
        pbar.set_description(f"Loss: {loss}")
        pbar.refresh()

        if loss == 0.:
            break

    return population.fittest(sig, p=top_p, order=order)[0].path[::2, 1], loss

    
