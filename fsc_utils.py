import torch
from aspire.utils import grid_2d, grid_3d
from projection_funcs import centered_fft3

class FourierShell():
    def __init__(self,L,dim,dtype=torch.float32,device=torch.device('cpu')):
        self.L = L
        self.dim = dim
        self.dtype = dtype
        self.device = device
        if(dim == 2):
            grid_func = grid_2d
        elif(dim == 3):
            grid_func = grid_3d
        self.grid_radius = torch.tensor(grid_func(L,shifted=True,normalized=False)['r'],dtype=dtype,device=device)


    @staticmethod
    def from_tensor(tensor):
        L = tensor.shape[0]
        dim = tensor.ndim
        return FourierShell(L,dim,tensor.dtype,tensor.device)

    def avergage_fourier_shell(self,*spectrum_signals):
        n = len(spectrum_signals)
        #TODO : what should be done with zero freqeuncy component in odd image length?
        shell_avg = torch.zeros(n, self.L//2, dtype=self.dtype ,device=self.device)
        for i in range(shell_avg.shape[1]):
            lower_rad_threshold = 0.5 + i
            upper_rad_threshold = 1.5 + i
            shell_ind = (self.grid_radius > lower_rad_threshold) & (self.grid_radius < upper_rad_threshold)
            for j in range(n):
                shell_avg[j,i] = torch.mean(spectrum_signals[j][shell_ind])

        if(n == 1):
            return shell_avg[0]
        
        return shell_avg

    def sum_over_shell(self,*shells):
        n = len(shells)
        shell_sum = torch.zeros(len(shells),dtype=self.dtype,device=self.device)
        for i in range(len(shells[0])):
            lower_rad_threshold = 0.5 + i
            upper_rad_threshold = 1.5 + i
            shell_ind = (self.grid_radius > lower_rad_threshold) & (self.grid_radius < upper_rad_threshold)
            num_comps_in_shell = torch.sum(shell_ind)
            for j in range(n):
                shell_sum[j] += num_comps_in_shell * shells[j][i]

        if(n == 1):
            return shell_sum[0]
        
        return shell_sum


def average_fourier_shell(*spectrum_signals):
    return FourierShell.from_tensor(spectrum_signals[0]).avergage_fourier_shell(*spectrum_signals)

def expand_fourier_shell(shell,dim):
    pass

def sum_over_shell(shell,L,dim):
    return FourierShell(L,dim,shell.dtype,shell.device).sum_over_shell(shell)


def vol_fsc(signal1,signal2):
    signal1_fft = centered_fft3(signal1)
    signal2_fft = centered_fft3(signal2)

    correlation,rpsd1,rpsd2 = average_fourier_shell(
        torch.real(signal1_fft * torch.conj(signal2_fft)),
        torch.abs(signal1_fft) ** 2,
        torch.abs(signal2_fft) ** 2
        )
    
    fsc = correlation / torch.sqrt(rpsd1 * rpsd2)

    return fsc

