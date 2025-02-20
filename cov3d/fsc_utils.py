import torch
from aspire.utils import grid_2d, grid_3d
from cov3d.projection_funcs import centered_fft2,centered_fft3

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
        self.grid_radius = torch.tensor(grid_func(L,shifted=False,normalized=False)['r'],dtype=dtype,device=device)
        self.radial_values = torch.arange(0,int(torch.ceil(torch.max(self.grid_radius)).item()))
        #self.shell_ind = []
        #for i in range(len(self.radial_values)):
        #    lower_rad_threshold,upper_rad_threshold = self.radial_avg_interval(i)
        #    self.shell_ind.append((self.grid_radius > lower_rad_threshold) & (self.grid_radius < upper_rad_threshold))


    @staticmethod
    def from_tensor(tensor):
        L = tensor.shape[0]
        dim = tensor.ndim
        return FourierShell(L,dim,tensor.dtype,tensor.device)
    
    def radial_avg_interval(self,radial_index):
        lower_rad_threshold = self.radial_values[radial_index] - 0.5
        upper_rad_threshold = self.radial_values[radial_index] + 0.5 if (radial_index < len(self.radial_values)-1) else self.radial_values[radial_index] + 1

        return lower_rad_threshold,upper_rad_threshold
        
    def _unsqueeze_batch_dim(self,tensor):
        #Add batch dimension if necessery
        if(tensor.ndim == self.dim + 1):
            return tensor
        elif(tensor.ndim == self.dim):
            return tensor.unsqueeze(0)
        else:
            raise Exception(f"Tensor dimension should be either {self.dim} or {self.dim+1} (for batch computation)")

    def avergage_fourier_shell(self,spectrum_signals):
        spectrum_signals = self._unsqueeze_batch_dim(spectrum_signals)
        n = spectrum_signals.shape[0]
        #TODO : what should be done with zero freqeuncy component in odd image length?
        shell_avg = torch.zeros(n, len(self.radial_values), dtype=spectrum_signals.dtype ,device=self.device)
        for i in range(shell_avg.shape[1]):
            lower_rad_threshold,upper_rad_threshold = self.radial_avg_interval(i)
            shell_ind = (self.grid_radius > lower_rad_threshold) & (self.grid_radius < upper_rad_threshold)
            shell_avg[:,i] = torch.sum(spectrum_signals * shell_ind.unsqueeze(0),dim=tuple([-i-1 for i in range(self.dim)])) / torch.sum(shell_ind)

        if(n == 1):
            return shell_avg[0]
        
        return shell_avg
    
    def rpsd(self,signals):        
        signals_psd = torch.abs(centered_fft3(self._unsqueeze_batch_dim(signals)))**2
        return self.avergage_fourier_shell(signals_psd)

    def sum_over_shell(self,shells):
        shells = shells.unsqueeze(0) if shells.ndim == 1 else shells
        n = len(shells)
        shell_sum = torch.zeros(len(shells),dtype=self.dtype,device=self.device)
        for i in range(len(shells[0])):
            lower_rad_threshold,upper_rad_threshold = self.radial_avg_interval(i)
            shell_ind = (self.grid_radius > lower_rad_threshold) & (self.grid_radius < upper_rad_threshold)
            num_comps_in_shell = torch.sum(shell_ind)
            for j in range(n):
                shell_sum[j] += num_comps_in_shell * shells[j][i]

        if(n == 1):
            return shell_sum[0]
        
        return shell_sum
    

    def expand_fourier_shell(self,shells):
        shells = shells.unsqueeze(0) if shells.ndim == 1 else shells
        n = len(shells)
        fourier_signal = torch.zeros((len(shells),)+(self.L,)*self.dim,dtype=self.dtype,device=self.device)
        for i in range(len(shells[0])):
            lower_rad_threshold,upper_rad_threshold = self.radial_avg_interval(i)
            shell_ind = (self.grid_radius > lower_rad_threshold) & (self.grid_radius < upper_rad_threshold)
            fourier_signal[:,shell_ind] = shells[:,i].unsqueeze(1)

        if(n == 1):
            return fourier_signal[0]
        

        return fourier_signal

def concat_tensor_tuple(tensor_tuple):
    return torch.cat([t.unsqueeze(0) for t in tensor_tuple])

def average_fourier_shell(*spectrum_signals):
    return FourierShell.from_tensor(spectrum_signals[0]).avergage_fourier_shell(concat_tensor_tuple(spectrum_signals))

def rpsd(*signals):
    return FourierShell.from_tensor(signals[0]).rpsd(concat_tensor_tuple(signals))

def expand_fourier_shell(shells,L,dim):
    return FourierShell(L,dim,shells.dtype,shells.device).expand_fourier_shell(shells)

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

def covar_rpsd(eigenvectors):
    eigenvecs_fft = centered_fft3(eigenvectors)
    return covar_correlate(eigenvecs_fft,eigenvecs_fft).real

def covar_correlate(fourier_eigenvecs1,fourier_eigenvecs2):
    L = fourier_eigenvecs1.shape[-1]
    dim = 3
    dtype = torch.float32 if fourier_eigenvecs1.dtype == torch.complex64 else torch.float64
    fourier_shell = FourierShell(L,dim,dtype,fourier_eigenvecs1.device)
    correlate_matrix = torch.zeros((len(fourier_shell.radial_values),len(fourier_shell.radial_values)),dtype=fourier_eigenvecs1.dtype,device=fourier_shell.device)
    for i in range(fourier_eigenvecs1.shape[0]):
        s = fourier_shell.avergage_fourier_shell(fourier_eigenvecs1[i] * torch.conj(fourier_eigenvecs2))
        correlate_matrix += s.T @ s.conj()

    return correlate_matrix

def covar_fsc(eigenvecs1,eigenvecs2):
    eigenvecs1_fft = centered_fft3(eigenvecs1)
    eigenvecs2_fft = centered_fft3(eigenvecs2)

    correlation = covar_correlate(eigenvecs1_fft,eigenvecs2_fft).real
    rpsd1 = covar_correlate(eigenvecs1_fft,eigenvecs1_fft).real
    rpsd2 = covar_correlate(eigenvecs2_fft,eigenvecs2_fft).real 

    #Float32 precision might not have the dynamic range to compute the product (since they can be very large). So we normalize the rpsd values by the zero frequency component and correct the normalization at the end
    bottom = torch.sqrt((rpsd1/rpsd1[0,0]) * (rpsd2/rpsd2[0,0])) * torch.sqrt(rpsd1[0,0]) * torch.sqrt(rpsd2[0,0]) 
    fsc = correlation / bottom

    return fsc