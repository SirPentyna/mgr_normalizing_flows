import torch
from torch import nn
import numpy as np
import normflows as nf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Distribution 


class MyDistribution(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, num_samples):
        """Sample from distribution and calculate log prob"""
        raise NotImplementedError
    
    def log_prob(self, z):
        """Calculate log prob for batch"""
        raise NotImplementedError


class NormUnif(MyDistribution):
    """Mixture with standard normal distribution and uniform distribution on a rectangle
    Args:
          x_dim: Dimension of each data point
          prob_delta: probability of normal distribution
          K_intervals: intervals that construct a rectangle - first row is a0, a1, ... and second row b0, b1, ...
                        and rectangle is constructed based on (a0, b0), (a1, b1), ...
    """
    def __init__(self, x_dim, prob_delta, K_intervals):
        super().__init__()

        self.x_dim = x_dim
        self.prob_delta = prob_delta
        self.K_intervals = K_intervals

        self.K_area = torch.prod(torch.diff(self.K_intervals, dim = 0))

        self.m = torch.distributions.MultivariateNormal(torch.zeros(self.x_dim), torch.eye(self.x_dim))

    def calculate_pdf(self, sample_point):

        if len(sample_point.shape) == 1:
            is_in_area = torch.sum(
                torch.logical_and(
                    torch.gt(sample_point, self.K_intervals[0]),
                    torch.lt(sample_point, self.K_intervals[1])
                    )
                )
            return self.prob_delta * np.exp(m.log_prob(sample_point)) + (1 - self.prob_delta) * (1 / self.K_area) * is_in_area

        elif len(sample_point.shape) == 2:
            is_in_area = \
            torch.eq( 
            torch.sum(torch.logical_and(
                torch.gt(sample_point, self.K_intervals[0]),
                torch.lt(sample_point, self.K_intervals[1])), dim = 1), 
            sample_point.shape[1] * torch.ones((sample_point.shape[0],)))

            return self.prob_delta * np.exp(self.m.log_prob(sample_point)) + (1 - self.prob_delta) * (1 / self.K_area) * is_in_area

    def log_prob(self, z):
        return torch.log(self.calculate_pdf(z))

    def prob_greater_t(self, t):
        # Probability of Normal part
        standard_norm =  torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        prob_norm = (1 - standard_norm.cdf(t))**self.x_dim

        # Probability of rectangle part
        a = self.K_intervals[0]
        b = self.K_intervals[1]
        max_at = np.maximum(a, torch.ones(len(a))*t)

        int_len = torch.maximum(b- max_at, torch.zeros(len(b)))
        area_above_t = torch.prod(int_len)
        prob_rectangle = area_above_t / self.K_area

        return self.prob_delta * prob_norm + (1 - self.prob_delta) * prob_rectangle


    def forward(self, num_samples=1):
        # Sample from X
        # 1) Sample delta
        delta = torch.bernoulli(torch.ones((num_samples, 1))*self.prob_delta)

        # 2) Sample from Z
        
        Z = self.m.sample((num_samples,))

        # 3) Sample from K
        K = torch.rand(num_samples, self.x_dim)


        #1st row
        K_start = self.K_intervals[0][None, :] # dimension expanded

        # start minus end
        K_range = torch.diff(self.K_intervals, dim = 0)

        #K times range plus starting points
        K = K *  K_range + K_start

        # 4) Take X

        X = delta * Z   + (1 - delta) * K
        return  X, self.log_prob(X)


def estim_prob_greater_t(model, R, t_float):
    samples = model.sample(R)[0]
    samples_np = samples.detach().numpy()
    return np.sum(np.all(samples_np>t_float, axis = 1)) / R


### Model functions

# class SimpleDense(nn.Module):
#     def __init__(self, input_dim) -> None:
#         super().__init__()
#         net = nn.ModuleList([nn.Linear(input_dim, int(4*input_dim))])
#         net.append(nn.LeakyReLU(negative_slope=0.01))
#         net.append(nn.Linear(int(4*input_dim), int(16*input_dim)))
#         net.append(nn.LeakyReLU(negative_slope=0.01))
#         net.append(nn.Linear(int(16*input_dim), int(4*input_dim)))
#         net.append(nn.LeakyReLU(negative_slope=0.01))
#         net.append(nn.Linear(int(4*input_dim), input_dim))
#         self.nets = nn.Sequential(*net)

#     def forward(self, x):
#         return self.nets(x)


class SimpleDense(nn.Module):
    def __init__(self, input_dim, max_power_of_two) -> None:
        super().__init__()
        powers_of_two = list(range(5)) + list(range(5, -1, -1))
        nets = []
        for i in range(len(powers_of_two) - 1):
            p = powers_of_two[i]
            p_next = powers_of_two[i+1]
            nets.append(nn.Linear(input_dim* 2 ** p, input_dim*2**p_next))
            nets.append(nn.LeakyReLU(0.2))
        self.seq_nets =  nn.Sequential(*nets)

    def forward(self, x):
        return self.seq_nets(x)

class Swap(nn.Module):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels=2):
        
        super().__init__()
        
        self.num_channels = num_channels

    def forward(self, z):
        z1 = z[:, : self.num_channels // 2, ...]
        z2 = z[:, self.num_channels // 2 :, ...]
        z = torch.cat([z2, z1], dim=1)
        log_det = 0
        return z, log_det

    def inverse(self, z):
        z1 = z[:, : (self.num_channels + 1) // 2, ...]
        z2 = z[:, (self.num_channels + 1) // 2 :, ...]
        z = torch.cat([z2, z1], dim=1)
        log_det = 0
        return z, log_det



def Split(z):
    return z[:,0][:, None], z[:,1][:, None]
def Con(y1, y2):
    return torch.cat((y1, y2), 1)

def zero_log_det_like_z(z):
    return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

class AffineSingleBlock(nn.Module):
    def __init__(self, param_net = None):
        super().__init__()

        self.param_net = param_net
        if param_net:
            pass
        else:
            self.a = torch.nn.Parameter(torch.tensor(0.0))
    
    def forward(self, z):
        x1, x2 = Split(z)
        if self.param_net:
            x1_modified = self.param_net(x1)
        else:
            x1_modified = self.a * x1
        y1 = x1
        y2 = x2 + x1_modified
        log_det = zero_log_det_like_z(y1)
        return Con(y1, y2), log_det

    def inverse(self, z):
        y1, y2 = Split(z)
        if self.param_net:
            y1_modified = self.param_net(y1)
        else:
            y1_modified = self.a * y1
        x1 = y1
        x2 = y2 - y1_modified
        log_det = zero_log_det_like_z(y1)
        return Con(x1, x2), log_det


class AffineMultipleBlocks(nn.Module):
    def __init__(self, num_affine_blocks=3):
        super().__init__()

        self.flows = nn.ModuleList([])
        for i in range(num_affine_blocks):
            self.flows += [AffineSingleBlock()]

    def forward(self, z):
        log_det_total = zero_log_det_like_z(z)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        return z, log_det_total
    
    def inverse(self, z):
        log_det_total = zero_log_det_like_z(z)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_total += log_det
        return z, log_det_total



class MyNormFlow(nn.Module):
    def __init__(self, q0, flows):
        super().__init__()
        self.q0 =  q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)
        Args:
          x: Batch sampled from target distribution
        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    
    def sample(self, num_samples):
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def forward(self, z):
        log_det_total = zero_log_det_like_z(z)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x):
        log_q = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q








