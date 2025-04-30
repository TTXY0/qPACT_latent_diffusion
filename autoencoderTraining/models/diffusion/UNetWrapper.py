from models.diffusion.openaimodel import UNetModel
import torch as th
class UNetWrapper(UNetModel):
    def __init__(self, *args, k, svd_path, full_latent_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.svd_path = svd_path
        self.full_latent_dim = full_latent_dim
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.U = th.load(svd_path).to(device)
        self.U_k =  self.U[:, :k]
        self.U_k.requires_grad = False
        print("running init of unetwrapper")
        
    def forward(self, x, timesteps, context=None, y=None,**kwargs):
        print("wdasjd;lfasjfsafl")
        print(timesteps.shape)
        x = self.preprocess(x)
        x = super().forward(x, timesteps, context, y,**kwargs)
        x = self.postprocess(x)
        return x
    
    def preprocess(self, x):
        #print(x.shape)
        with th.no_grad():
            batch_size = x.shape[1]
            x_mean = x.mean()
            x = x - x_mean
            x = th.matmul(self.U_k, x)
            #print(x.shape)
            x = x.view(batch_size, self.full_latent_dim
                        [0], self.full_latent_dim[1], self.full_latent_dim[2])  
            #print(x.shape)
            return x
    
    def postprocess(self, x):
        return x 