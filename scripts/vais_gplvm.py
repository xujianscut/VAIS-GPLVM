import os, math, argparse, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.path.join(os.path.dirname(__file__), 'data')

def load_oil():
    X = np.loadtxt(os.path.join(DATA,'DataTrn.txt'))
    L = np.loadtxt(os.path.join(DATA,'DataTrnLbls.txt'))
    y = L.argmax(1)
    X = (X - X.mean(0)) / X.std(0)
    return torch.tensor(X,dtype=torch.float64), torch.tensor(y)

class VAIS_GPLVM(nn.Module):
    '''VAIS-GPLVM (Xu et al. 2024): annealed IS with time-inhomogeneous unadjusted
       Langevin transitions over the latents, sparse-GP likelihood, Algorithm 1 (minibatch).'''
    def __init__(self, N, D, Q=10, M=30, K=10, eta=3e-3):
        super().__init__()
        self.N,self.D,self.Q,self.M,self.K,self.eta = N,D,Q,M,K,eta
        self.log_ls = nn.Parameter(torch.zeros(Q,dtype=torch.float64))    # ARD
        self.log_sf = nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.log_sn = nn.Parameter(torch.tensor(-2.0,dtype=torch.float64))
        self.Z      = nn.Parameter(0.5*torch.randn(M,Q,dtype=torch.float64))
        self.a    = nn.Parameter(torch.zeros(N,Q,dtype=torch.float64))    # q0 mean (psi)
        self.log_s= nn.Parameter(-2.0*torch.ones(N,Q,dtype=torch.float64))# q0 std
        self.um   = nn.Parameter(0.1*torch.randn(D,M,dtype=torch.float64))# q(u) mean
        self.uL   = nn.Parameter(torch.eye(M,dtype=torch.float64).repeat(D,1,1)*0.1)
        b = torch.linspace(-4,4,K+1); self.register_buffer('beta', torch.sigmoid(b))

    def kern(self,A,B):
        ls = self.log_ls.exp(); sf2 = (2*self.log_sf).exp()
        d = ((A.unsqueeze(-2)/ls - B.unsqueeze(-3)/ls)**2).sum(-1)
        return sf2*torch.exp(-0.5*d)

    def _chol_kmm(self):
        Kmm = self.kern(self.Z,self.Z)+1e-5*torch.eye(self.M,dtype=torch.float64,device=self.Z.device)
        return torch.linalg.cholesky(Kmm)

    def gauss_loglik(self, Xb, H, u, Lmm):     # log N(x_d; Knm Kmm^-1 u_d, Qnn+sn2 I), Eq.13 model
        B=H.shape[0]; sn2=(2*self.log_sn).exp()
        Knm=self.kern(H,self.Z); Knn=self.kern(H,H)
        Qnn=Knm@torch.cholesky_solve(Knm.t(),Lmm)
        I=torch.eye(B,dtype=torch.float64,device=H.device)
        Lc=torch.linalg.cholesky(Knn-Qnn+sn2*I+1e-6*I)
        mu=Knm@torch.cholesky_solve(u.t(),Lmm)
        diff=Xb-mu
        quad=(diff*torch.cholesky_solve(diff,Lc)).sum()
        logdet=2*torch.log(torch.diagonal(Lc)).sum()*self.D
        return -0.5*(quad+logdet+B*self.D*math.log(2*math.pi))

    def logq0(self,H,a,s):
        return (-0.5*((H-a)/s)**2 - torch.log(s) - 0.5*math.log(2*math.pi)).sum()
    def logprior(self,H):
        return (-0.5*H**2 - 0.5*math.log(2*math.pi)).sum()

    def kl_u(self,Lmm):
        kl=0.0
        for d in range(self.D):
            Sd=torch.tril(self.uL[d]); Sigma=Sd@Sd.t()
            tr=torch.diagonal(torch.cholesky_solve(Sigma,Lmm)).sum()
            m=self.um[d:d+1].t()
            quad=(m.t()@torch.cholesky_solve(m,Lmm)).squeeze()
            logdet_p=2*torch.log(torch.diagonal(Lmm)).sum()
            logdet_q=2*torch.log(torch.abs(torch.diagonal(Sd))).sum()
            kl=kl+0.5*(tr+quad-self.M+logdet_p-logdet_q)
        return kl

    def neg_elbo(self, idx):
        Lmm=self._chol_kmm()
        Xb=self.X[idx]; a=self.a[idx]; s=self.log_s[idx].exp(); B=len(idx); scale=self.N/B
        uLtri=torch.tril(self.uL); u=self.um+torch.einsum('dij,dj->di',uLtri,torch.randn_like(self.um))
        eps0=torch.randn_like(a); H=a+s*eps0
        L = -self.logq0(H,a,s)                                  # -log q0(H0)
        def glq(Hin,beta):
            Hin=Hin.detach().requires_grad_(True)
            lp = self.gauss_loglik(Xb,Hin,u,Lmm)+self.logprior(Hin)
            val=beta*lp+(1-beta)*self.logq0(Hin,a,s)
            return torch.autograd.grad(val,Hin,create_graph=True)[0]
        for k in range(1,self.K+1):                             # ULA-AIS transitions
            beta=self.beta[k]; eps=torch.randn_like(H)
            g1=glq(H,beta); Hk=H+self.eta*g1+math.sqrt(2*self.eta)*eps
            g2=glq(Hk,beta); ebar=math.sqrt(self.eta/2)*(g1+g2)-eps
            L=L-0.5*((ebar**2).sum()-(eps**2).sum())            # -R_{k-1}
            H=Hk
        L=L+self.logprior(H)+self.gauss_loglik(Xb,H,u,Lmm)      # +log p(X,H_K)
        L=L-self.kl_u(Lmm)/scale                                # KL(u) shared -> divide back
        return -(L*scale)/self.N                                # per-point neg-ELBO (full-data est.)

    @torch.no_grad()
    def reconstruct(self):
        Lmm=self._chol_kmm(); Knm=self.kern(self.a,self.Z)
        return Knm@torch.cholesky_solve(self.um.t(),Lmm)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--iters',type=int,default=1500)
    ap.add_argument('--lr',type=float,default=5e-3); ap.add_argument('--B',type=int,default=200); a=ap.parse_args()
    X,y=load_oil(); X=X.to(DEV); N,D=X.shape
    Xc=X-X.mean(0); U,S,V=torch.linalg.svd(Xc,full_matrices=False)
    m=VAIS_GPLVM(N,D,Q=10,M=30,K=10,eta=3e-3).to(DEV); m.X=X
    with torch.no_grad(): m.a.copy_(U[:,:10]*S[:10]); m.Z.copy_(m.a[torch.randperm(N)[:30]])
    opt=torch.optim.Adam(m.parameters(),lr=a.lr); hist=[]
    for it in range(a.iters):
        idx=torch.randperm(N,device=DEV)[:a.B]
        opt.zero_grad(); ne=m.neg_elbo(idx); ne.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),50.0); opt.step(); hist.append(ne.item())
        if it%100==0 or it==a.iters-1:
            with torch.no_grad(): mse=((m.reconstruct()-X)**2).mean().item()
            print(f'iter {it:4d}  loss {ne.item():9.4f}  MSE {mse:7.4f}  sn {m.log_sn.exp().item():.3f}',flush=True)
    with torch.no_grad():
        mse=((m.reconstruct()-X)**2).mean().item(); a_=m.a.cpu().numpy(); inv=(1/m.log_ls.exp()).cpu().numpy()
    order=np.argsort(-inv)[:2]
    plt.figure(figsize=(12,3.5))
    plt.subplot(1,3,1)
    for c in range(3):
        i=(y.numpy()==c); plt.scatter(a_[i,order[0]],a_[i,order[1]],s=8,label=f'phase {c}')
    plt.title('2D latent subspace (VAIS-GPLVM)'); plt.legend(fontsize=7); plt.xlabel(f'dim {order[0]}'); plt.ylabel(f'dim {order[1]}')
    plt.subplot(1,3,2); plt.bar(range(10),inv); plt.title('inverse lengthscale (ARD)')
    plt.subplot(1,3,3); plt.plot(hist); plt.title('training loss'); plt.xlabel('iter')
    plt.tight_layout(); out=os.path.join(os.path.dirname(__file__),'result.png'); plt.savefig(out,dpi=120)
    print(f'\nFINAL  MSE(std-space)={mse:.4f}')
    print('saved',out)

if __name__=='__main__': main()
