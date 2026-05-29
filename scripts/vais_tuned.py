import os, math, argparse, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from compare import GPLVM, load_oil, DEV

def schedule(K, kind='power'):
    if kind=='linear': b=torch.linspace(0,1,K+1)
    elif kind=='sigmoid': b=torch.sigmoid(torch.linspace(-5,5,K+1)); b=(b-b[0])/(b[-1]-b[0])
    elif kind=='power': b=torch.linspace(0,1,K+1)**2
    return b.to(DEV)

def vais_loss(m,idx,betas,eta):
    Lmm=m.chol_kmm(); Xb=m.X[idx]; a=m.a[idx]; s=m.log_s[idx].exp(); B=len(idx); scale=m.N/B
    u=m.sample_u(Lmm); K=len(betas)-1
    H=a+s*torch.randn_like(a); L=-m.lq0_pp(H,a,s).sum()
    def glq(Hin,beta):
        Hin=Hin.detach().requires_grad_(True)
        val=(beta*(m.ll_diag(Xb,Hin,u,Lmm)+m.lp_pp(Hin))+(1-beta)*m.lq0_pp(Hin,a,s)).sum()
        return torch.autograd.grad(val,Hin,create_graph=True)[0]
    for k in range(1,K+1):
        beta=betas[k]; e=torch.randn_like(H); g1=glq(H,beta)
        Hk=H+eta*g1+math.sqrt(2*eta)*e; g2=glq(Hk,beta)
        ebar=math.sqrt(eta/2)*(g1+g2)-e
        L=L-0.5*((ebar**2).sum()-(e**2).sum()); H=Hk
    L=L+m.lp_pp(H).sum()+m.ll_diag(Xb,H,u,Lmm).sum()
    return -(scale*L-m.kl_u(Lmm))/m.N

@torch.no_grad()
def recon_from(m,H): Lmm=m.chol_kmm(); return m.kern(H,m.Z)@torch.cholesky_solve(m.um.t(),Lmm)

def transported_latent(m,betas,eta,S=20,polish=60,plr=2e-2):
    '''S AIS flows -> average endpoint (denoise) -> noiseless MAP polish (grad-ascent on log p(X,H), beta=1).'''
    Lmm=m.chol_kmm().detach(); a=m.a.detach(); s=m.log_s.exp().detach(); K=len(betas)-1
    um=m.um.detach()
    def joint_grad(H,u,beta):
        Hin=H.clone().requires_grad_(True)
        val=(beta*(m.ll_diag(m.X,Hin,u,Lmm)+m.lp_pp(Hin))+(1-beta)*m.lq0_pp(Hin,a,s)).sum()
        return torch.autograd.grad(val,Hin)[0]
    acc=torch.zeros_like(a)
    for _ in range(S):
        u=um+torch.einsum('dij,dj->di',torch.tril(m.uL).detach(),torch.randn_like(um))
        H=a+s*torch.randn_like(a)
        for k in range(1,K+1):
            g=joint_grad(H,u,betas[k]); H=H+eta*g+math.sqrt(2*eta)*torch.randn_like(H)
        acc=acc+H
    H=acc/S
    for t in range(polish):                      # MAP refinement, noise off, decaying step
        g=joint_grad(H,um,1.0); H=H+plr*(1-t/polish)*g
    return H.detach()

def train_vais(X,y,iters,lr,eta,K,kind,seed=0,s0=-2.0,polish=60):
    torch.manual_seed(seed); np.random.seed(seed)
    N,D=X.shape; m=GPLVM(N,D).to(DEV); m.X=X
    Xc=X-X.mean(0); U,Sv,V=torch.linalg.svd(Xc,full_matrices=False)
    with torch.no_grad(): m.a.copy_(U[:,:10]*Sv[:10]); m.Z.copy_(m.a[torch.randperm(N)[:30]]); m.log_s.fill_(s0)
    betas=schedule(K,kind); opt=torch.optim.Adam(m.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters)
    for it in range(iters):
        idx=torch.randperm(N,device=DEV)[:200]
        opt.zero_grad(); vais_loss(m,idx,betas,eta).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),50.0); opt.step(); sch.step()
    ma=((recon_from(m,m.a)-X)**2).mean().item()
    Ht=transported_latent(m,betas,eta,S=20,polish=polish)
    mt=((recon_from(m,Ht)-X)**2).mean().item()
    return m,betas,ma,mt,Ht

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='sweep'); a=ap.parse_args()
    X,y=load_oil(); X=X.to(DEV)
    if a.mode=='sweep':
        print('sweep 900it -> MSE@a / MSE@transported+polish (IWVI ref=0.0123, MF=0.0163)',flush=True)
        best=None
        for eta in [3e-3,6e-3]:
          for K in [15]:
            for kind in ['power','sigmoid']:
                m,betas,ma,mt,Ht=train_vais(X,y,900,5e-3,eta,K,kind)
                print(f'  eta={eta:.1e} K={K} {kind:7s} ->  {ma:.4f} / {mt:.4f}',flush=True)
                if best is None or mt<best[0]: best=(mt,eta,K,kind)
        print(f'\nBEST transported-MSE={best[0]:.4f} @ eta={best[1]} K={best[2]} {best[3]}',flush=True)
    else:
        eta=float(os.environ.get('ETA',3e-3)); K=int(os.environ.get('K',20)); kind=os.environ.get('SCHED','power')
        m,betas,ma,mt,Ht=train_vais(X,y,2500,5e-3,eta,K,kind,polish=80)
        print(f'FINAL VAIS-tuned (eta={eta} K={K} {kind} 2500it +polish): MSE@a={ma:.4f}  MSE@transported={mt:.4f}',flush=True)
        with torch.no_grad(): Hc=Ht.cpu().numpy(); inv=(1/m.log_ls.exp()).cpu().numpy()
        o=np.argsort(-inv)[:2]; plt.figure(figsize=(5,4.2))
        for c in range(3):
            i=(y.numpy()==c); plt.scatter(Hc[i,o[0]],Hc[i,o[1]],s=8,label=f'phase {c}')
        plt.title(f'VAIS-tuned latent  (MSE={mt:.4f})'); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__),'vais_tuned.png'),dpi=120); print('saved vais_tuned.png',flush=True)
if __name__=='__main__': main()
