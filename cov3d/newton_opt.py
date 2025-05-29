import torch

class BlockNewtonOptimizer(torch.optim.Optimizer):
    """
    Implements a Newton optimizer with backtracking line search. 
    Assumes hessian is block diagonal with respect to the parameters and batch (first dim) of tensor.
    """
    def __init__(self, params, lr=1.0, max_ls_steps=10, c=1e-4, beta=0.1, damping=1e-6,line_search=True):
        defaults = dict(lr=lr, max_ls_steps=max_ls_steps, c=c, beta=beta, damping=damping,line_search=line_search)
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is None:
            raise ValueError("Newton optimizer requires a closure to reevaluate the model.")

        with torch.enable_grad():
            loss = closure()
            loss.sum().backward(create_graph=True)

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            beta = group['beta']
            max_ls_steps = group['max_ls_steps']
            damping = group['damping']
            line_search = group['line_search']

            for param in group['params']:
                if param.grad is None:
                    continue

                orig_param = param.data.clone()
                flat_aggregated_grad = param.grad.sum(dim=0).view(-1, 1)
                n = flat_aggregated_grad.shape[0]
                batch_size = param.shape[0]

                # Compute Hessian
                hessian = torch.zeros((n,) + param.shape, dtype=flat_aggregated_grad.dtype, device=flat_aggregated_grad.device)
                for i in range(n):
                    hessian[i] = torch.autograd.grad(flat_aggregated_grad[i], param, retain_graph=True)[0]
                    
                hessian = hessian.reshape(n,batch_size,n).transpose(0,1)
                # Damping for stability
                hessian = hessian + damping * torch.eye(n, device=param.device).unsqueeze(0)

                # Compute Newton step
                step_dir = torch.linalg.solve(hessian, param.grad.view(batch_size,-1))

                # Backtracking line search
                if(line_search):
                    alpha = lr
                    alpha_step_taken = torch.zeros(batch_size,device=param.device)
                    for _ in range(max_ls_steps):
                        param.data = orig_param - alpha * step_dir.view_as(param.data)

                        with torch.enable_grad():
                            trial_loss = closure()
                            trial_loss.sum().backward(create_graph=True)

                        alpha_step_taken[((trial_loss <= loss - c * alpha * torch.norm(param.grad.view(batch_size,-1),dim=1)**2).to(torch.int) + (alpha_step_taken == 0)) == 2] = alpha 
                        
                        alpha *= beta

                    param.data = orig_param - step_dir.view_as(param.data) * alpha_step_taken.reshape((-1,) + (param.data.ndim-1)*(1,))
                else:
                    # No line search, take the full step
                    param.data = orig_param - lr * step_dir.view_as(param.data)

        return loss




        