import torch
from torch import nn
import numpy as np
import normflows as nf
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.stats import chi2
import seaborn as sns

# Distribution 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def calc_prob_greater_t(samples_np, t_float):
    return np.sum(np.all(samples_np>t_float, axis = 1)) / samples_np.shape[0]

def estim_prob_greater_t(model, R, t_float):
    samples = model.sample(R)[0]
    samples_np = samples.detach().numpy()
    return calc_prob_greater_t(samples_np=samples_np, t_float=t_float)

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

class SimpleDenseCustDim(nn.Module):
    def __init__(self, dims,  init_zeros=False) -> None:
        super().__init__()
        nets = []
        for i in range(len(dims) - 2):
            d = dims[i]
            d_next = dims[i+1]
            nets.append(nn.Linear(d, d_next))
            nets.append(nn.LeakyReLU(0.2))
        nets.append(nn.Linear(dims[-2], dims[-1]))
        if init_zeros:
            #nets[-1].weight.data.zero_()
            #nets[-1].bias.data.zero_()
            torch.nn.init.zeros_(nets[-1].weight)
            torch.nn.init.zeros_(nets[-1].bias)
            #nn.init.zeros_(nets[-1].weight)
            #nn.init.zeros_(nets[-1].bias)
            #nn.init.uniform_(nets[-1].weight, 0.0, 0.001)
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


class RealNVPBlock(nn.Module):
    def __init__(self, param_net_t = None, param_net_s = None):
        super().__init__()

        self.param_net_t = param_net_t
        self.param_net_s = param_net_s
        
    
    def forward(self, z):
        x1, x2 = Split(z)
        s = self.param_net_s(x1)
        x1_modified = self.param_net_t(x1)
        y1 = x1
        y2 = x2 * torch.exp(s)+ x1_modified
        log_det = zero_log_det_like_z(y1) + torch.sum(s)
        return Con(y1, y2), log_det

    def inverse(self, z):
        y1, y2 = Split(z)
        s = self.param_net_s(y1)
        y1_modified = self.param_net_t(y1)
        x1 = y1
        x2 = (y2 - y1_modified) * torch.exp(-s)
        log_det = zero_log_det_like_z(y1) - torch.sum(s)
        return Con(x1, x2), log_det

class ScalingBlock(nn.Module):
    def __init__(self, dim_x = 2):
        super().__init__()

        self.s = torch.nn.Parameter(torch.zeros([1 ,dim_x]))
       
    def forward(self, z):
        log_det = zero_log_det_like_z(z) + torch.sum(self.s)
        return z * torch.exp(self.s), log_det

    def inverse(self, z):
        log_det = zero_log_det_like_z(z) - torch.sum(self.s)
        return z / torch.exp(self.s), log_det


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
        ### if inf then
        return torch.abs(-torch.mean(log_q))

    
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



#### Stratification



def generate_from_normal_stratum(n, Rm, m, stratum_i):
    # Sampling from N(0,1) with dimension n:
    mNorm = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), torch.eye(n))
    zeta = mNorm.sample((Rm,))

    # Normalisation
    length = torch.sqrt(torch.sum(torch.mul(zeta,zeta), axis=1, keepdims=True))
    X = zeta/length

    # Sampling from uniform distribution on [0,1]
    unif = torch.distributions.Uniform(0, 1)
    u = unif.sample((Rm,1))

    # Making sample from uniform distribution on [i/m, (i+1)/m]
    v = stratum_i/m + u*1/m

    # Inverse cdf of chi2 
    d2 = chi2.ppf(v, df = n)

    # Rescaling samples from uniform normal
    sample = torch.tensor(np.sqrt(d2)) * X
    return sample.detach().numpy()

def generate_samples_from_model_stratified(model, number_of_samples_from_model, m, n, verbose = False, palette_type = 'viridis'):
    Rm = int(number_of_samples_from_model/m)
    samples_full = np.empty((0,n))
    stratum_number = []

    if verbose:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), subplot_kw=dict(box_aspect=1))
        if palette_type == 'viridis':
            palette = list(sns.color_palette('viridis', m).as_hex())
        elif palette_type == 'python_blue':
            palette = list(sns.dark_palette(python_blue, m).as_hex())

    for i in range(m):
        s_np = generate_from_normal_stratum(n=n, Rm=Rm, m=m, stratum_i=i)
        stratum_number +=[i]*Rm

        if verbose:
            axes[0].scatter(s_np[:,0],s_np[:,1], label=f'Stratum {i+1}',c=palette[i], alpha=0.5, marker=".")

        #put through model
        sample2 = torch.tensor(s_np, dtype=torch.float)
        for flow in model.flows:
            sample2, log_det = flow(sample2)
        s_np2 = sample2.detach().numpy()
        if verbose:
            axes[1].scatter(s_np2[:,0],s_np2[:,1], label=f'Stratum {i+1}',c=palette[i], alpha=0.5, marker=".")
        samples_full = np.concatenate((samples_full,s_np2))
        

    if verbose:
        axes[0].set_title("Standard Normal")
        axes[1].set_title("After model")
        
        axes[0].legend()
        axes[1].legend()

        plt.show()
    
    if not verbose:
        return {'samples':samples_full, 
                'stratum_number':stratum_number}

            
def generate_samples_from_model_stratified_old(model, number_of_samples_from_model, m, n, x_m, x_sd, verbose = False, palette_type = 'viridis'):
    Rm = int(number_of_samples_from_model/m)
    samples_unnorm_full = np.empty((0,n))
    stratum_number = []

    if verbose:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
        if palette_type == 'viridis':
            palette = list(sns.color_palette('viridis', m).as_hex())
        elif palette_type == 'python_blue':
            palette = list(sns.dark_palette(python_blue, m).as_hex())

    for i in range(m):
        s_np = generate_from_normal_stratum(n=n, Rm=Rm, m=m, stratum_i=i)
        stratum_number +=[i]*Rm

        if verbose:
            axes[0].scatter(s_np[:,0],s_np[:,1], label=f'Strata {i+1}',c=palette[i], alpha=0.5, marker=".")
            axes[0].gca().set_aspect('equal', 'box')

        #put through model
        sample2 = torch.tensor(s_np, dtype=torch.float)
        for flow in model.flows:
            sample2, log_det = flow(sample2)
        s_np2 = sample2.detach().numpy()
        if verbose:
            axes[1].scatter(s_np2[:,0],s_np2[:,1], label=f'Stratum {i+1}',c=palette[i], alpha=0.5, marker=".")
            axes[1].gca().set_aspect('equal', 'box')
        
        # unnormalized
        samples_unnorm = s_np2 * x_sd + x_m
        samples_unnorm_full = np.concatenate((samples_unnorm_full,samples_unnorm))

        if verbose:
            axes[2].scatter(samples_unnorm[:,0],samples_unnorm[:,1], label=f'Stratum {i+1}',c=palette[i], alpha=0.5, marker=".")

    if verbose:
        axes[0].set_title("Standard Normal")
        axes[1].set_title("After model")
        axes[2].set_title("After model, unnormalized")
        
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()

        plt.show()
    return {'samples':samples_unnorm_full, 
            'stratum_number':stratum_number}




## FFJORD Functions

def get_batch(num_samples):
    
    prob_delta = 0.5
    K_intervals = torch.Tensor([[0.5, 1],
                            [3,3]])
    nu = NormUnif(x_dim = 2, prob_delta=prob_delta, K_intervals=K_intervals)
    points= nu.forward(num_samples)[0].numpy()
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)




class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

#### Here cam be added my data type
def get_batch_their(num_samples):
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)



