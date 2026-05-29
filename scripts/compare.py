import os, math, argparse, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DEV='cuda' if torch.cuda.is_available() else 'cpu'
DATA=os.path.join(os.path.dirname(__file__),'data')

def load_oil():
    X=np.loadtxt(os.path.join(DATA,'DataTrn.txt')); L=np.loadtxt(os.path.join(DATA,'DataTrnLbls.txt'))
    y=L.argmax(1); X=(X-X.mean(0))/X.std(0)
    return torch.tensor(X,dtype=torch.float64), torch.tensor(y)

class GPLVM(nn.Module):
    '''Shared sparse-GP GPLVM core; per-point diagonal (FITC) likelihood so MF/IW/VAIS are comparable.'''
    def __init__(self,N,D,Q=10,M=30):
        super().__init__()
        self.N,self.D,self.Q,self.M=N,D,Q,M
        self.log_ls=nn.Parameter(torch.zeros(Q,dtype=torch.float64))
        self.log_sf=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.log_sn=nn.Parameter(torch.tensor(-2.0,dtype=torch.float64))
        self.Z =nn.Parameter(0.5*torch.randn(M,Q,dtype=torch.float64))
        self.a =nn.Parameter(torch.zeros(N,Q,dtype=torch.float64))
        self.log_s=nn.Parameter(-2.0*torch.ones(N,Q,dtype=torch.float64))
        self.um=nn.Parameter(0.1*torch.randn(D,M,dtype=torch.float64))
        self.uL=nn.Parameter(torch.eye(M,dtype=torch.float64).repeat(D,1,1)*0.1)
    def kern(self,A,B):
        ls=self.log_ls.exp(); sf2=(2*self.log_sf).exp()
        d=((A.unsqueeze(-2)/ls-B.unsqueeze(-3)/ls)**2).sum(-1); return sf2*torch.exp(-0.5*d)
    def chol_kmm(self):
        Kmm=self.kern(self.Z,self.Z)+1e-5*torch.eye(self.M,dtype=torch.float64,device=self.Z.device)
        return torch.linalg.cholesky(Kmm)
    def ll_diag(self,Xb,H,u,Lmm):              # returns per-row loglik (over D), shape (.,)
        sf2=(2*self.log_sf).exp(); sn2=(2*self.log_sn).exp()
        knm=self.kern(H,self.Z)
        qnn=(knm*torch.cholesky_solve(knm.t(),Lmm).t()).sum(-1)
        v=(sf2-qnn+sn2).clamp_min(1e-6).unsqueeze(-1)
        mu=knm@torch.cholesky_solve(u.t(),Lmm)
        return (-0.5*((Xb-mu)**2/v+torch.log(v)+math.log(2*math.pi))).sum(-1)
    def lp_pp(self,H):  return (-0.5*H**2-0.5*math.log(2*math.pi)).sum(-1)
    def lq0_pp(self,H,a,s): return (-0.5*((H-a)/s)**2-torch.log(s)-0.5*math.log(2*math.pi)).sum(-1)
    def kl_u(self,Lmm):
        kl=0.0
        for d in range(self.D):
            Sd=torch.tril(self.uL[d]); Sigma=Sd@Sd.t()
            tr=torch.diagonal(torch.cholesky_solve(Sigma,Lmm)).sum(); m=self.um[d:d+1].t()
            quad=(m.t()@torch.cholesky_solve(m,Lmm)).squeeze()
            kl=kl+0.5*(tr+quad-self.M+2*torch.log(torch.diagonal(Lmm)).sum()-2*torch.log(torch.abs(torch.diagonal(Sd))).sum())
        return kl
    def sample_u(self,Lmm):
        return self.um+torch.einsum('dij,dj->di',torch.tril(self.uL),torch.randn_like(self.um))
    @torch.no_grad()
    def reconstruct(self):
        Lmm=self.chol_kmm(); return self.kern(self.a,self.Z)@torch.cholesky_solve(self.um.t(),Lmm)

def mf_loss(m,idx):
    Lmm=m.chol_kmm(); Xb=m.X[idx]; a=m.a[idx]; s=m.log_s[idx].exp(); B=len(idx); scale=m.N/B
    u=m.sample_u(Lmm); H=a+s*torch.randn_like(a)
    ll=m.ll_diag(Xb,H,u,Lmm).sum()
    klh=(0.5*(s**2+a**2-1-2*torch.log(s))).sum()
    L=scale*(ll-klh)-m.kl_u(Lmm)
    return -L/m.N

def iw_loss(m,idx,Kiw=5):
    Lmm=m.chol_kmm(); Xb=m.X[idx]; a=m.a[idx]; s=m.log_s[idx].exp(); B=len(idx); scale=m.N/B
    u=m.sample_u(Lmm)
    eps=torch.randn(Kiw,B,m.Q,dtype=torch.float64,device=DEV)
    H=a.unsqueeze(0)+s.unsqueeze(0)*eps                 # K,B,Q
    Hf=H.reshape(Kiw*B,m.Q); Xf=Xb.unsqueeze(0).expand(Kiw,-1,-1).reshape(Kiw*B,m.D)
    logw=(m.ll_diag(Xf,Hf,u,Lmm)+m.lp_pp(Hf)-m.lq0_pp(Hf,a.repeat(Kiw,1),s.repeat(Kiw,1))).reshape(Kiw,B)
    iwelbo=(torch.logsumexp(logw,0)-math.log(Kiw)).sum()
    L=scale*iwelbo-m.kl_u(Lmm)
    return -L/m.N

def vais_loss(m,idx,Kais=10,eta=3e-3):
    Lmm=m.chol_kmm(); Xb=m.X[idx]; a=m.a[idx]; s=m.log_s[idx].exp(); B=len(idx); scale=m.N/B
    u=m.sample_u(Lmm)
    betas=torch.sigmoid(torch.linspace(-4,4,Kais+1,device=DEV))
    eps0=torch.randn_like(a); H=a+s*eps0
    L=-m.lq0_pp(H,a,s).sum()
    def glq(Hin,beta):
        Hin=Hin.detach().requires_grad_(True)
        val=(beta*(m.ll_diag(Xb,Hin,u,Lmm)+m.lp_pp(Hin))+(1-beta)*m.lq0_pp(Hin,a,s)).sum()
        return torch.autograd.grad(val,Hin,create_graph=True)[0]
    for k in range(1,Kais+1):
        beta=betas[k]; e=torch.randn_like(H); g1=glq(H,beta)
        Hk=H+eta*g1+math.sqrt(2*eta)*e; g2=glq(Hk,beta)
        ebar=math.sqrt(eta/2)*(g1+g2)-e
        L=L-0.5*((ebar**2).sum()-(e**2).sum()); H=Hk
    L=L+m.lp_pp(H).sum()+m.ll_diag(Xb,H,u,Lmm).sum()
    L=scale*L-m.kl_u(Lmm)
    return -L/m.N

@torch.no_grad()
def iw_diagnostics(m,Kiw=25):                 # Table-3 style ESS / weight entropy on full data
    Lmm=m.chol_kmm(); a=m.a; s=m.log_s.exp(); u=m.sample_u(Lmm)
    eps=torch.randn(Kiw,m.N,m.Q,dtype=torch.float64,device=DEV)
    H=a.unsqueeze(0)+s.unsqueeze(0)*eps; Hf=H.reshape(Kiw*m.N,m.Q)
    Xf=m.X.unsqueeze(0).expand(Kiw,-1,-1).reshape(Kiw*m.N,m.D)
    logw=(m.ll_diag(Xf,Hf,u,Lmm)+m.lp_pp(Hf)-m.lq0_pp(Hf,a.repeat(Kiw,1),s.repeat(Kiw,1))).reshape(Kiw,m.N)
    w=torch.softmax(logw,0)                    # normalized weights per point
    ess=(1.0/(w**2).sum(0)).mean().item()
    ent=(-(w*torch.log(w+1e-12)).sum(0)).mean().item()
    return ess,ent

@torch.no_grad()
def nell(m,H,S=80):                            # Negative Expected Log-Likelihood: -E_q(u)[log p(x_n|H_n,u)]
    Lmm=m.chol_kmm(); acc=torch.zeros(m.N,device=DEV)
    for _ in range(S): acc=acc+m.ll_diag(m.X,H,m.sample_u(Lmm),Lmm)
    return -(acc/S).mean().item()

def train(method,X,y,iters=1500,lr=5e-3,B=200,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    N,D=X.shape; m=GPLVM(N,D).to(DEV); m.X=X
    Xc=X-X.mean(0); U,S,V=torch.linalg.svd(Xc,full_matrices=False)
    with torch.no_grad(): m.a.copy_(U[:,:10]*S[:10]); m.Z.copy_(m.a[torch.randperm(N)[:30]])
    opt=torch.optim.Adam(m.parameters(),lr=lr); hist=[]
    fn={'MF':mf_loss,'IWVI':iw_loss,'VAIS':vais_loss}[method]
    for it in range(iters):
        idx=torch.randperm(N,device=DEV)[:B]
        opt.zero_grad(); ne=fn(m,idx); ne.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),50.0); opt.step(); hist.append(ne.item())
    with torch.no_grad(): mse=((m.reconstruct()-X)**2).mean().item()
    nl=nell(m,m.a); ess,ent=iw_diagnostics(m)
    return m,hist,mse,nl,ess,ent

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--iters',type=int,default=1500); a=ap.parse_args()
    X,y=load_oil(); X=X.to(DEV)
    res={}; fig,ax=plt.subplots(1,4,figsize=(16,3.8))
    for j,meth in enumerate(['MF','IWVI','VAIS']):
        m,hist,mse,nl,ess,ent=train(meth,X,y,iters=a.iters)
        res[meth]=(mse,nl,ess,ent)
        print(f'{meth:5s}  MSE={mse:.4f}  NELL={nl:7.2f}  ESS={ess:5.2f}  Hw={ent:.3f}',flush=True)
        with torch.no_grad(): a_=m.a.cpu().numpy(); inv=(1/m.log_ls.exp()).cpu().numpy()
        o=np.argsort(-inv)[:2]
        for c in range(3):
            i=(y.numpy()==c); ax[j].scatter(a_[i,o[0]],a_[i,o[1]],s=7,label=f'phase {c}')
        ax[j].set_title(f'{meth}-GPLVM  (MSE={mse:.3f})'); ax[j].legend(fontsize=6)
    ax[3].axis('off')
    txt='Method   MSE     NELL    ESS    H(w)\n'+'-'*38+'\n'
    for k in ['MF','IWVI','VAIS']:
        v=res[k]; txt+=f'{k:6s} {v[0]:.4f} {v[1]:7.2f} {v[2]:5.2f}  {v[3]:.2f}\n'
    txt+='\n(ESS / H(w): K=25 importance weights;\n higher = less weight collapse)'
    ax[3].text(0.0,0.5,txt,family='monospace',fontsize=10,va='center')
    plt.tight_layout(); out=os.path.join(os.path.dirname(__file__),'compare.png'); plt.savefig(out,dpi=120)
    print('\nsaved',out)
if __name__=='__main__': main()
