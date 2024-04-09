import torch

'''
Reference: SuperGlue CVPR'2020
'''


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    '''
    Perform Sinkhorn Normalization in Log-space for stability
    :param Z:
    :param log_mu:
    :param log_nu:
    :param iters:
    :return:
    '''
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, bins0=None, bins1=None, alpha=None, iters=100):
    """
    Perform Differentiable Optimal Transport in Log-space for stability
    :param scores:
    :param alpha:
    :param iters:
    :return:
    """

    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    if bins0 is None:
        bins0 = alpha.expand(b, m, 1)
    if bins1 is None:
        bins1 = alpha.expand(b, 1, n)
    
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm #multiply probabilities by M + N
    return Z


