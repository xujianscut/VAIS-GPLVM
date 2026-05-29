import os, math, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from compare import GPLVM, load_oil, DEV
from vais_tuned import schedule, vais_loss, transported_latent, recon_from

def gp_refit(m, Hfix, iters=1000, lr=5e-3):
    '''M-step: freeze latents at Hfix, refit GP (um,uL,kernel,sn) on standard SVGP diag ELBO.'''
    Hfix=Hfix.detach()
    for p in [m.a, m.log_s]: p.requires_grad_(False)
    params=[m.um,m.uL,m.log_ls,m.log_sf,m.log_sn,m.Z]
    opt=torch.optim.Adam(params,lr=lr); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters)
    N=m.N
    for it in range(iters):
        idx=torch.randperm(N,device=DEV)[:200]; B=len(idx); scale=N/B
        Lmm=m.chol_kmm(); u=m.sample_u(Lmm)
        ll=m.ll_diag(m.X[idx],Hfix[idx],u,Lmm).sum()
        loss=-(scale*ll-m.kl_u(Lmm))/N
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,50.0); opt.step(); sch.step()
    return m

def main():
    X,y=load_oil(); X=X.to(DEV); N,D=X.shape
    eta,K,kind=3e-3,20,'power'
    torch.manual_seed(0); np.random.seed(0)
    m=GPLVM(N,D).to(DEV); m.X=X
    Xc=X-X.mean(0); U,Sv,V=torch.linalg.svd(Xc,full_matrices=False)
    with torch.no_grad(): m.a.copy_(U[:,:10]*Sv[:10]); m.Z.copy_(m.a[torch.randperm(N)[:30]]); m.log_s.fill_(-2.0)
    betas=schedule(K,kind); opt=torch.optim.Adam(m.parameters(),lr=5e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2000)
    for it in range(2000):
        idx=torch.randperm(N,device=DEV)[:200]
        opt.zero_grad(); vais_loss(m,idx,betas,eta).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),50.0); opt.step(); sch.step()
    mse_base=((recon_from(m,m.a)-X)**2).mean().item(); sn_base=m.log_sn.exp().item()
    Ht=transported_latent(m,betas,eta,S=20,polish=80)
    mse_tp=((recon_from(m,Ht)-X)**2).mean().item()
    print(f'[VAIS]            MSE@a={mse_base:.4f}  sn={sn_base:.3f}',flush=True)
    print(f'[+transport+polish] MSE={mse_tp:.4f}',flush=True)
    gp_refit(m,Ht,iters=1200,lr=5e-3)
    mse_refit=((recon_from(m,Ht)-X)**2).mean().item(); sn_refit=m.log_sn.exp().item()
    print(f'[+GP refit (E->M)] MSE={mse_refit:.4f}  sn={sn_refit:.3f}',flush=True)
    print(f'\nrefs:  MF=0.0163   IWVI=0.0123',flush=True)
    with torch.no_grad(): Hc=Ht.cpu().numpy(); inv=(1/m.log_ls.exp()).cpu().numpy()
    o=np.argsort(-inv)[:2]; plt.figure(figsize=(5.2,4.3))
    for c in range(3):
        i=(y.numpy()==c); plt.scatter(Hc[i,o[0]],Hc[i,o[1]],s=8,label=f'phase {c}')
    plt.title(f'VAIS-tuned (E->M)  MSE={mse_refit:.4f}'); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__),'vais_final.png'),dpi=120); print('saved vais_final.png')
if __name__=='__main__': main()
