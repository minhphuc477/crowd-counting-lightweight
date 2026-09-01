#!/usr/bin/env python3
"""D0 bottleneck pipeline for lightweight crowd counting.

Implements:
- D-R / G-R: 1-2 px sampling-phase / shift instability.
- D-K / G-K: inter-person separability collapse through encoder depth.
- D-L / G-L: normalized effective-rank collapse.
- D1: zoom-recovery control.

This carrier is a diagnostic baseline, NOT the final proposed model.

Manifest CSV columns:
  image,points,split[,sizes]
points: Nx2 [x,y] in .npy/.npz/.txt/.json/.mat
sizes: optional Nx1 diameter or Nx2 [w,h]. Definitive D-K needs real head-size proxies.
"""
from __future__ import annotations
import argparse, json, math, random, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


def seed_everything(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _as_points(x):
    a=np.asarray(x,dtype=np.float32)
    if a.size==0: return np.zeros((0,2),np.float32)
    a=a.reshape(-1,a.shape[-1])
    if a.shape[1]<2: raise ValueError(f'Expected Nx2 points, got {a.shape}')
    return a[:,:2].astype(np.float32)


def load_points(path:str)->np.ndarray:
    p=Path(path); s=p.suffix.lower()
    if s=='.npy': return _as_points(np.load(p,allow_pickle=True))
    if s=='.npz':
        z=np.load(p,allow_pickle=True)
        for k in ('points','annPoints','pts','xy'):
            if k in z: return _as_points(z[k])
        raise KeyError(f'{p}: no points key')
    if s=='.txt':
        rows=[]
        for line in p.read_text(encoding='utf-8').splitlines():
            v=line.strip().replace(',',' ').split()
            if len(v)>=2: rows.append([float(v[0]),float(v[1])])
        return np.asarray(rows,np.float32).reshape(-1,2)
    if s=='.json':
        o=json.loads(p.read_text(encoding='utf-8'))
        if isinstance(o,dict):
            for k in ('points','annPoints','pts','xy'):
                if k in o: return _as_points(o[k])
        return _as_points(o)
    if s=='.mat':
        if loadmat is None: raise RuntimeError('scipy required for .mat')
        m=loadmat(p)
        for k in ('annPoints','points','pts','xy'):
            if k in m: return _as_points(m[k])
        if 'image_info' in m:
            x=m['image_info']
            for _ in range(8):
                if isinstance(x,np.ndarray) and x.dtype==object and x.size==1: x=x.flat[0]
                else: break
            try: return _as_points(x)
            except Exception as e: raise ValueError(f'{p}: convert image_info to .npy/.npz') from e
        raise KeyError(f'{p}: unsupported .mat structure')
    raise ValueError(f'Unsupported annotation: {p}')


def load_sizes(path:Optional[str], n:int)->Optional[np.ndarray]:
    if not path or str(path).lower()=='nan': return None
    p=Path(path); s=p.suffix.lower()
    if s=='.npy': a=np.load(p,allow_pickle=True)
    elif s=='.npz':
        z=np.load(p,allow_pickle=True); k=next((k for k in ('sizes','size','wh','head_size') if k in z),None)
        if k is None: raise KeyError(f'{p}: no size key')
        a=z[k]
    elif s=='.txt':
        rows=[]
        for line in p.read_text(encoding='utf-8').splitlines():
            v=[float(x) for x in line.strip().replace(',',' ').split()]
            if v: rows.append(v[:2])
        a=np.asarray(rows,np.float32)
    elif s=='.json':
        o=json.loads(p.read_text(encoding='utf-8'))
        if isinstance(o,dict):
            k=next((k for k in ('sizes','size','wh','head_size') if k in o),None)
            if k is None: raise KeyError(f'{p}: no size key')
            a=np.asarray(o[k],np.float32)
        else: a=np.asarray(o,np.float32)
    else: raise ValueError(f'Unsupported size file: {p}')
    a=np.asarray(a,np.float32)
    if a.ndim==1: a=a[:,None]
    a=a.reshape(a.shape[0],-1)
    if a.shape[0]!=n: raise ValueError(f'{p}: sizes N={a.shape[0]} != points N={n}')
    return (a[:,0] if a.shape[1]==1 else np.sqrt(np.maximum(a[:,0],1e-6)*np.maximum(a[:,1],1e-6))).astype(np.float32)


def pil_to_tensor(im):
    a=np.asarray(im.convert('RGB'),dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1).contiguous()


def norm_imagenet(x):
    mean=x.new_tensor([.485,.456,.406])[:,None,None]; std=x.new_tensor([.229,.224,.225])[:,None,None]
    return (x-mean)/std


def resize_points(x,pts,sizes,scale):
    _,h,w=x.shape; nh=max(2,round(h*scale)); nw=max(2,round(w*scale))
    y=F.interpolate(x[None],size=(nh,nw),mode='bilinear',align_corners=False)[0]
    p=pts.copy()
    if len(p): p[:,0]*=nw/w; p[:,1]*=nh/h
    s=None if sizes is None else sizes*math.sqrt((nw/w)*(nh/h))
    return y,p,s


def pad_min(x,pts,sizes,hmin,wmin):
    _,h,w=x.shape; ph=max(0,hmin-h); pw=max(0,wmin-w)
    if ph or pw:
        mode='reflect' if h>1 and w>1 else 'replicate'
        x=F.pad(x[None],(0,pw,0,ph),mode=mode)[0]
    return x,pts,sizes


def random_crop(x,pts,sizes,crop):
    x,pts,sizes=pad_min(x,pts,sizes,crop,crop); _,h,w=x.shape
    top=random.randint(0,max(0,h-crop)); left=random.randint(0,max(0,w-crop))
    m=(pts[:,0]>=left)&(pts[:,0]<left+crop)&(pts[:,1]>=top)&(pts[:,1]<top+crop)
    p=pts[m].copy(); p[:,0]-=left; p[:,1]-=top
    s=None if sizes is None else sizes[m].copy()
    return x[:,top:top+crop,left:left+crop],p,s


def hflip(x,pts,sizes):
    _,_,w=x.shape; y=torch.flip(x,[2]); p=pts.copy()
    if len(p): p[:,0]=(w-1)-p[:,0]
    return y,p,sizes


def pad_div(x,d=16):
    _,h,w=x.shape; ph=(d-h%d)%d; pw=(d-w%d)%d
    if ph or pw:
        mode='reflect' if h>1 and w>1 else 'replicate'; x=F.pad(x[None],(0,pw,0,ph),mode=mode)[0]
    return x,(h,w)


class CrowdManifestDataset(Dataset):
    def __init__(self,manifest,split,train=False,crop=512,scale_min=.8,scale_max=1.2):
        df=pd.read_csv(manifest); need={'image','points','split'}
        if need-set(df.columns): raise ValueError(f'Missing manifest columns {need-set(df.columns)}')
        self.df=df[df['split'].astype(str)==str(split)].reset_index(drop=True)
        if len(self.df)==0: raise ValueError(f'No rows for split={split}')
        self.train=train; self.crop=crop; self.scale_min=scale_min; self.scale_max=scale_max
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; ip=str(r.image); pp=str(r.points)
        sp=None
        if 'sizes' in self.df.columns and pd.notna(r.get('sizes')) and str(r.get('sizes')).strip(): sp=str(r.get('sizes'))
        x=pil_to_tensor(Image.open(ip)); pts=load_points(pp); sizes=load_sizes(sp,len(pts)) if sp else None
        _,h,w=x.shape; m=(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h); pts=pts[m]
        if sizes is not None: sizes=sizes[m]
        if self.train:
            x,pts,sizes=resize_points(x,pts,sizes,random.uniform(self.scale_min,self.scale_max)); x,pts,sizes=random_crop(x,pts,sizes,self.crop)
            if random.random()<.5: x,pts,sizes=hflip(x,pts,sizes)
            orig=(x.shape[1],x.shape[2])
        else: x,orig=pad_div(x,16)
        return {'image':norm_imagenet(x),'points':torch.from_numpy(pts),'sizes':None if sizes is None else torch.from_numpy(sizes),'name':Path(ip).stem,'orig_hw':torch.tensor(orig)}


def collate(batch):
    return {'image':torch.stack([b['image'] for b in batch]),'points':[b['points'] for b in batch],'sizes':[b['sizes'] for b in batch],'name':[b['name'] for b in batch],'orig_hw':torch.stack([b['orig_hw'] for b in batch])}


def make_div(v,d=8): return max(d,int(v+d/2)//d*d)


class CBA(nn.Module):
    def __init__(self,ci,co,k=3,s=1,g=1,act=True):
        super().__init__(); self.c=nn.Conv2d(ci,co,k,s,k//2,groups=g,bias=False); self.b=nn.BatchNorm2d(co); self.a=nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self,x): return self.a(self.b(self.c(x)))


class DS(nn.Module):
    def __init__(self,ci,co,s=1,e=2.):
        super().__init__(); mid=make_div(ci*e); self.res=(s==1 and ci==co)
        self.f=nn.Sequential(CBA(ci,mid,1),CBA(mid,mid,3,s,g=mid),CBA(mid,co,1,act=False)); self.a=nn.SiLU(inplace=True)
    def forward(self,x):
        y=self.f(x); return self.a(y+x if self.res else y)


class CarrierNet(nn.Module):
    def __init__(self,width=.5,decoder_ch=32):
        super().__init__(); self.width=float(width)
        c0,c1,c2,c3=[make_div(v*width) for v in (24,40,80,128)]
        self.stem=CBA(3,c0,3,2)
        self.stage1=nn.Sequential(DS(c0,c1,2),DS(c1,c1))
        self.stage2=nn.Sequential(DS(c1,c2,2),DS(c2,c2))
        self.stage3=nn.Sequential(DS(c2,c3,2),DS(c3,c3))
        self.l4=nn.Conv2d(c1,decoder_ch,1); self.l8=nn.Conv2d(c2,decoder_ch,1); self.l16=nn.Conv2d(c3,decoder_ch,1)
        self.f8=CBA(decoder_ch,decoder_ch); self.f4=CBA(decoder_ch,decoder_ch); self.head=nn.Conv2d(decoder_ch,1,1)
    def forward(self,x,return_features=False):
        x=self.stem(x); s4=self.stage1(x); s8=self.stage2(s4); s16=self.stage3(s8)
        p8=self.l8(s8)+F.interpolate(self.l16(s16),size=s8.shape[-2:],mode='bilinear',align_corners=False); p8=self.f8(p8)
        p4=self.l4(s4)+F.interpolate(p8,size=s4.shape[-2:],mode='bilinear',align_corners=False); p4=self.f4(p4)
        m=F.softplus(self.head(p4))
        return (m,{'s4':s4,'s8':s8,'s16':s16}) if return_features else m


def nparams(m): return sum(p.numel() for p in m.parameters())


def points_grid(pts,oh,ow,ih,iw,device):
    gt=torch.zeros((1,oh,ow),device=device)
    if pts.numel()==0: return gt
    p=pts.to(device); gx=torch.clamp((p[:,0]/iw*ow).long(),0,ow-1); gy=torch.clamp((p[:,1]/ih*oh).long(),0,oh-1); idx=gy*ow+gx
    gt.view(-1).scatter_add_(0,idx,torch.ones_like(idx,dtype=torch.float32)); return gt


def sum_pool(x,b):
    h,w=x.shape[-2:]; bh=min(b,h); bw=min(b,w); return F.avg_pool2d(x,(bh,bw),(bh,bw))*(bh*bw)


def block_loss(pred,pts_list,input_hw,block=4,lambda_global=.2):
    B,_,oh,ow=pred.shape; ih,iw=input_hw; gts=[]; counts=[]
    for p in pts_list: gts.append(points_grid(p,oh,ow,ih,iw,pred.device)); counts.append(float(len(p)))
    gt=torch.stack(gts); local=F.smooth_l1_loss(sum_pool(pred,block),sum_pool(gt,block)); pc=pred.flatten(1).sum(1); gc=pc.new_tensor(counts); glob=F.smooth_l1_loss(pc,gc)
    return local+lambda_global*glob


def save_ckpt(path,model,opt,epoch,args):
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'width':model.width,'decoder_ch':args.decoder_ch,'args':vars(args)},path)


def load_carrier(path,device):
    c=torch.load(path,map_location='cpu'); m=CarrierNet(c.get('width',.5),c.get('decoder_ch',32)); m.load_state_dict(c['model']); return m.to(device).eval()


def parse_models(spec,device):
    out={}
    for item in spec.split(','):
        if item.strip():
            n,p=item.split('=',1); out[n.strip()]=load_carrier(p.strip(),device)
    if not out: raise ValueError('No models parsed')
    return out


def train_cmd(a):
    seed_everything(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=CrowdManifestDataset(a.manifest,a.split,True,a.crop,a.scale_min,a.scale_max); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True,collate_fn=collate)
    m=CarrierNet(a.width,a.decoder_ch).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.wd); scaler=torch.amp.GradScaler('cuda',enabled=a.amp and dev.type=='cuda'); best=1e99
    print(json.dumps({'device':str(dev),'params':nparams(m),'width':a.width,'n_train':len(ds)},indent=2))
    for ep in range(1,a.epochs+1):
        m.train(); ls=[]
        for b in dl:
            x=b['image'].to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=a.amp and dev.type=='cuda'):
                pred=m(x); loss=block_loss(pred,b['points'],x.shape[-2:],a.block,a.lambda_global)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            if a.grad_clip>0: torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip)
            scaler.step(opt); scaler.update(); ls.append(float(loss.detach().cpu()))
        ml=float(np.mean(ls)); print(f'epoch={ep:04d} loss={ml:.6f}')
        if ml<best: best=ml; save_ckpt(a.out,m,opt,ep,a)
    print('saved',a.out)


@torch.no_grad()
def eval_one(m,ds,dev,workers=2):
    rows=[]; dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=workers,collate_fn=collate)
    for b in dl:
        x=b['image'].to(dev); y=m(x); gt=float(len(b['points'][0])); pred=float(y.sum().cpu()); rows.append({'image':b['name'][0],'pred':pred,'gt':gt,'error':pred-gt,'abs_error':abs(pred-gt)})
    return pd.DataFrame(rows)


def eval_cmd(a):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=load_carrier(a.checkpoint,dev); ds=CrowdManifestDataset(a.manifest,a.split,False); df=eval_one(m,ds,dev,a.workers); df.to_csv(a.out,index=False)
    print({'MAE':df.abs_error.mean(),'RMSE':float(np.sqrt(np.mean(df.error.values**2))),'Bias':df.error.mean(),'N':len(df)})

# ------------------------- spatial helpers -------------------------
def shift_reflect(x,dx,dy,pad):
    h,w=x.shape[-2:]; p=max(pad,abs(dx),abs(dy)); z=F.pad(x,(p,p,p,p),mode='reflect'); return z[...,p-dy:p-dy+h,p-dx:p-dx+w]


def translate_grid(x,sx,sy):
    n,c,h,w=x.shape; th=torch.eye(2,3,dtype=x.dtype,device=x.device)[None].repeat(n,1,1); th[:,0,2]=-2*sx/max(w-1,1); th[:,1,2]=-2*sy/max(h-1,1)
    g=F.affine_grid(th,x.shape,align_corners=True); return F.grid_sample(x,g,mode='bilinear',padding_mode='zeros',align_corners=True)


def crop_inner(x,b):
    if b<=0:return x
    h,w=x.shape[-2:]
    return x[...,b:-b,b:-b] if h>2*b and w>2*b else x[...,0:0,0:0]


def spatial_cos(a,b,border=0):
    a=crop_inner(a,border); b=crop_inner(b,border)
    if a.numel()==0:return np.nan
    a=F.normalize(a.flatten(2),dim=1); b=F.normalize(b.flatten(2),dim=1); return float((a*b).sum(1).mean().cpu())


def stride_of(ih,fh): return ih/float(fh)


def sample_vec(feat,pts,input_hw):
    if len(pts)==0:return feat.new_zeros((0,feat.shape[1]))
    h,w=input_hw; p=pts.to(feat.device).float(); gx=2*p[:,0]/max(w-1,1)-1; gy=2*p[:,1]/max(h-1,1)-1; grid=torch.stack([gx,gy],-1)[None,:,None,:]
    s=F.grid_sample(feat,grid,mode='bilinear',align_corners=True); return s[0,:,:,0].T.contiguous()


def sample_scalar(m,pts,input_hw): return sample_vec(m,pts,input_hw)[:,0]


def nearest(pts):
    n=len(pts)
    if n<2:return np.zeros(0,np.int64),np.zeros(0,np.float32)
    d2=((pts[:,None,:]-pts[None,:,:])**2).sum(-1); np.fill_diagonal(d2,np.inf); idx=d2.argmin(1); return idx.astype(np.int64),np.sqrt(d2[np.arange(n),idx]).astype(np.float32)


# ------------------------- D-R: phase instability -------------------------
@torch.no_grad()
def dr_cmd(a):
    seed_everything(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); models=parse_models(a.models,dev); ds=CrowdManifestDataset(a.manifest,a.split,False)
    shifts=[(dx,dy) for dy in range(-a.max_shift,a.max_shift+1) for dx in range(-a.max_shift,a.max_shift+1)]; rows=[]
    for i in range(len(ds)):
        s=ds[i]; x=s['image'][None].to(dev); sizes=None if s['sizes'] is None else s['sizes'].numpy()
        for mn,m in models.items():
            base,bf=m(x,True)
            for dx,dy in shifts:
                xs=shift_reflect(x,dx,dy,a.max_shift+2); y,ff=m(xs,True); so=stride_of(x.shape[-2],y.shape[-2]); ya=translate_grid(y,-dx/so,-dy/so); bo=math.ceil((a.max_shift+1)/so); yi=crop_inner(ya,bo); bi=crop_inner(base,bo)
                r={'image':s['name'],'model':mn,'dx':dx,'dy':dy,'count_common':float(yi.sum().cpu()) if yi.numel() else np.nan,'gt_count':len(s['points']),'median_head_size':float(np.median(sizes)) if sizes is not None and len(sizes) else np.nan,'mass_l1_aligned':float((bi-yi).abs().mean().cpu()) if bi.shape==yi.shape and bi.numel() else np.nan}
                for st,f0 in bf.items():
                    f=ff[st]; ss=stride_of(x.shape[-2],f.shape[-2]); fa=translate_grid(f,-dx/ss,-dy/ss); b=math.ceil((a.max_shift+1)/ss); r[f'{st}_cos']=spatial_cos(f0,fa,b)
                rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(a.out,index=False)
    sm=df.groupby(['image','model']).agg(ShiftCountStd=('count_common','std'),ShiftCountRange=('count_common',lambda z:float(np.nanmax(z)-np.nanmin(z))),MassL1=('mass_l1_aligned','mean'),s4_cos=('s4_cos','mean'),s8_cos=('s8_cos','mean'),s16_cos=('s16_cos','mean'),gt_count=('gt_count','first'),median_head_size=('median_head_size','first')).reset_index(); sp=str(Path(a.out).with_name(Path(a.out).stem+'_summary.csv')); sm.to_csv(sp,index=False); print(sm.groupby('model')[['ShiftCountStd','ShiftCountRange','MassL1','s4_cos','s8_cos','s16_cos']].mean()); print('saved',a.out,sp)


# ------------------------- zoom control -------------------------
@torch.no_grad()
def zoom_cmd(a):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); models=parse_models(a.models,dev); ds=CrowdManifestDataset(a.manifest,a.split,False); scales=[float(x) for x in a.scales.split(',')]; rows=[]
    for i in range(len(ds)):
        s=ds[i]; x0=s['image'][None].to(dev); gt=len(s['points'])
        for mn,m in models.items():
            for z in scales:
                x=x0 if z==1 else F.interpolate(x0,scale_factor=z,mode='bilinear',align_corners=False); y=m(x); pred=float(y.sum().cpu()); rows.append({'image':s['name'],'model':mn,'zoom':z,'pred':pred,'gt':gt,'error':pred-gt,'abs_error':abs(pred-gt)})
    df=pd.DataFrame(rows); df.to_csv(a.out,index=False); print(df.groupby(['model','zoom'])[['abs_error','error']].mean())


# ------------------------- D-K: pair separability -------------------------
@torch.no_grad()
def prototypes(model,ds,dev,max_heads=20000):
    bag={'s4':[],'s8':[],'s16':[]}; n=0
    for i in range(len(ds)):
        s=ds[i]; pts=s['points']
        if len(pts)==0:continue
        x=s['image'][None].to(dev); _,f=model(x,True)
        for st,v in f.items(): bag[st].append(sample_vec(v,pts,x.shape[-2:]).cpu())
        n+=len(pts)
        if n>=max_heads:break
    out={}
    for st,ch in bag.items():
        z=torch.cat(ch)[:max_heads].float(); out[st]=F.normalize(F.normalize(z,dim=1).mean(0),dim=0).to(dev)
    return out


def headness(feat,p): return ((F.normalize(feat,dim=1)*F.normalize(p,dim=0)[None,:,None,None]).sum(1,keepdim=True)+1)*.5


def local_rect_error(mass,allpts,p1,p2,margin,input_hw):
    ih,iw=input_hw; oh,ow=mass.shape[-2:]; x0=max(0.,min(p1[0],p2[0])-margin); y0=max(0.,min(p1[1],p2[1])-margin); x1=min(float(iw),max(p1[0],p2[0])+margin); y1=min(float(ih),max(p1[1],p2[1])+margin)
    gm=(allpts[:,0]>=x0)&(allpts[:,0]<x1)&(allpts[:,1]>=y0)&(allpts[:,1]<y1); gt=float(gm.sum()); gx0=max(0,math.floor(x0/iw*ow)); gx1=min(ow,math.ceil(x1/iw*ow)); gy0=max(0,math.floor(y0/ih*oh)); gy1=min(oh,math.ceil(y1/ih*oh)); pred=float(mass[...,gy0:gy1,gx0:gx1].sum().cpu()); return pred-gt,gt


@torch.no_grad()
def dk_cmd(a):
    seed_everything(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); models=parse_models(a.models,dev); pds=CrowdManifestDataset(a.manifest,a.prototype_split,False); ds=CrowdManifestDataset(a.manifest,a.split,False); proto={n:prototypes(m,pds,dev,a.max_proto_heads) for n,m in models.items()}; rows=[]; warned=False
    for i in range(len(ds)):
        s=ds[i]; pts=s['points'].numpy(); sizes=None if s['sizes'] is None else s['sizes'].numpy()
        if len(pts)<2:continue
        if sizes is None and not warned: warnings.warn('No sizes: normalized_spacing remains NaN; use real head-size proxy for final D-K claim.'); warned=True
        idx,dist=nearest(pts); seen=set(); pairs=[]
        for q,j in enumerate(idx.tolist()):
            k=tuple(sorted((q,int(j))))
            if k not in seen:seen.add(k);pairs.append(k)
        x=s['image'][None].to(dev); ih,iw=x.shape[-2:]
        for mn,m in models.items():
            mass,f=m(x,True); hm={st:headness(v,proto[mn][st]) for st,v in f.items()}
            for aa,bb in pairs:
                p1,p2=pts[aa],pts[bb]; d=float(np.linalg.norm(p1-p2)); hs=np.nan if sizes is None else float((sizes[aa]+sizes[bb])*.5); ns=np.nan if not np.isfinite(hs) else d/max(hs,1e-6); mid=torch.tensor([[(p1[0]+p2[0])*.5,(p1[1]+p2[1])*.5]],dtype=torch.float32); pp=torch.tensor(np.stack([p1,p2]),dtype=torch.float32); margin=max(a.min_margin_px,a.margin_scale*hs if np.isfinite(hs) else a.min_margin_px); le,lgt=local_rect_error(mass,pts,p1,p2,margin,(ih,iw)); r={'image':s['name'],'model':mn,'a':aa,'b':bb,'spacing_px':d,'pair_head_size':hs,'normalized_spacing':ns,'local_error':le,'local_gt':lgt}
                for st in ('s4','s8','s16'):
                    hp=sample_scalar(hm[st],pp,(ih,iw)).cpu().numpy(); hmid=float(sample_scalar(hm[st],mid,(ih,iw))[0].cpu()); r[f'{st}_merge_ratio']=hmid/max(float(hp.mean()),1e-6); vv=sample_vec(f[st],pp,(ih,iw)); r[f'{st}_pair_cos']=float(F.cosine_similarity(vv[:1],vv[1:2],dim=1)[0].cpu())
                rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(a.out,index=False); col='normalized_spacing' if df.normalized_spacing.notna().sum()>=20 else 'spacing_px'; d=df[np.isfinite(df[col])].copy()
    if len(d)>=a.n_bins*5:
        d['spacing_bin']=pd.qcut(d[col],q=a.n_bins,duplicates='drop'); sm=d.groupby(['model','spacing_bin'],observed=True).agg(n=('local_error','size'),spacing=(col,'mean'),local_bias=('local_error','mean'),local_mae=('local_error',lambda z:float(np.mean(np.abs(z)))),s4_merge=('s4_merge_ratio','mean'),s8_merge=('s8_merge_ratio','mean'),s16_merge=('s16_merge_ratio','mean')).reset_index(); sp=str(Path(a.out).with_name(Path(a.out).stem+'_binned.csv')); sm.to_csv(sp,index=False); print(sm); print('saved',sp)
    print('saved',a.out)


# ------------------------- D-L: effective rank -------------------------
def erank(x):
    x=np.asarray(x,np.float64)
    if x.ndim!=2 or min(x.shape)<2:return {'n':0,'c':0,'rank_cap':0,'entropy_erank':np.nan,'participation_rank':np.nan,'entropy_erank_norm':np.nan,'participation_rank_norm':np.nan}
    x=x-x.mean(0,keepdims=True); sv=np.linalg.svd(x,full_matrices=False,compute_uv=False); pw=sv**2
    if pw.sum()<=1e-12:return {'n':x.shape[0],'c':x.shape[1],'rank_cap':min(x.shape),'entropy_erank':0.,'participation_rank':0.,'entropy_erank_norm':0.,'participation_rank_norm':0.}
    p=pw/pw.sum(); e=float(np.exp(-(p*np.log(p+1e-12)).sum())); pr=float(pw.sum()**2/(np.square(pw).sum()+1e-12)); cap=min(x.shape)
    return {'n':x.shape[0],'c':x.shape[1],'rank_cap':cap,'entropy_erank':e,'participation_rank':pr,'entropy_erank_norm':e/cap,'participation_rank_norm':pr/cap}


@torch.no_grad()
def dl_cmd(a):
    seed_everything(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); models=parse_models(a.models,dev); ds=CrowdManifestDataset(a.manifest,a.split,False); rec=[]
    for i in range(len(ds)):
        s=ds[i]; pts=s['points']; p=pts.numpy(); sizes=None if s['sizes'] is None else s['sizes'].numpy(); idx,dist=nearest(p)
        if len(p)==0 or len(dist)==0:continue
        ns=np.full(len(p),np.nan,np.float32) if sizes is None else dist/np.maximum(sizes,1e-6); x=s['image'][None].to(dev)
        for mn,m in models.items():
            mass,f=m(x,True); ie=float(mass.sum().cpu())-len(p)
            for st,v in f.items():
                z=sample_vec(v,pts,x.shape[-2:]).cpu().numpy()
                for j in range(len(p)): rec.append({'image':s['name'],'model':mn,'stage':st,'head_idx':j,'spacing_px':float(dist[j]),'normalized_spacing':float(ns[j]),'head_size':float(sizes[j]) if sizes is not None else np.nan,'img_error':ie,'feature':z[j].astype(np.float32)})
    meta=pd.DataFrame([{k:v for k,v in r.items() if k!='feature'} for r in rec]); feats=[r['feature'] for r in rec]; col='normalized_spacing' if meta.normalized_spacing.notna().sum()>=a.n_bins*a.min_bin_n else 'spacing_px'; vals=meta.loc[np.isfinite(meta[col]),col].values; edges=np.unique(np.quantile(vals,np.linspace(0,1,a.n_bins+1)))
    if len(edges)<3:raise RuntimeError('Not enough spacing variation')
    meta['spacing_bin']=pd.cut(meta[col],bins=edges,include_lowest=True,duplicates='drop'); out=[]
    for (mn,st,bn),ii in meta.groupby(['model','stage','spacing_bin'],observed=True).groups.items():
        ii=list(ii)
        if len(ii)<a.min_bin_n:continue
        z=np.stack([feats[k] for k in ii]); sub=meta.loc[ii]; out.append({'model':mn,'stage':st,'spacing_bin':str(bn),'spacing_mean':float(sub[col].mean()),'head_size_mean':float(sub.head_size.mean()),'img_abs_error_mean':float(np.abs(sub.img_error).mean()),**erank(z)})
    od=pd.DataFrame(out); od.to_csv(a.out,index=False); print(od); print('saved',a.out)


def selftest_cmd(a):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for w in (.35,.5,1.,1.5):
        m=CarrierNet(w).to(dev).eval(); x=torch.randn(1,3,256,320,device=dev); y,f=m(x,True); print({'width':w,'params':nparams(m),'mass':tuple(y.shape),'s4':tuple(f['s4'].shape),'s8':tuple(f['s8'].shape),'s16':tuple(f['s16'].shape)})
    print('selftest OK')


def add_common(p): p.add_argument('--manifest',required=True); p.add_argument('--split',default='test'); p.add_argument('--workers',type=int,default=2); p.add_argument('--seed',type=int,default=1337)


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('train'); p.add_argument('--manifest',required=True); p.add_argument('--split',default='train'); p.add_argument('--out',required=True); p.add_argument('--width',type=float,default=.5); p.add_argument('--decoder-ch',type=int,default=32); p.add_argument('--crop',type=int,default=512); p.add_argument('--scale-min',type=float,default=.8); p.add_argument('--scale-max',type=float,default=1.2); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--epochs',type=int,default=300); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--wd',type=float,default=1e-4); p.add_argument('--block',type=int,default=4); p.add_argument('--lambda-global',type=float,default=.2); p.add_argument('--grad-clip',type=float,default=5.); p.add_argument('--workers',type=int,default=4); p.add_argument('--amp',action='store_true'); p.add_argument('--seed',type=int,default=1337); p.set_defaults(func=train_cmd)
    p=sub.add_parser('eval'); add_common(p); p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True); p.set_defaults(func=eval_cmd)
    p=sub.add_parser('dr'); add_common(p); p.add_argument('--models',required=True); p.add_argument('--max-shift',type=int,default=2); p.add_argument('--out',required=True); p.set_defaults(func=dr_cmd)
    p=sub.add_parser('zoom'); add_common(p); p.add_argument('--models',required=True); p.add_argument('--scales',default='1,2,4'); p.add_argument('--out',required=True); p.set_defaults(func=zoom_cmd)
    p=sub.add_parser('dk'); add_common(p); p.add_argument('--models',required=True); p.add_argument('--prototype-split',default='train'); p.add_argument('--max-proto-heads',type=int,default=20000); p.add_argument('--margin-scale',type=float,default=1.5); p.add_argument('--min-margin-px',type=float,default=16.); p.add_argument('--n-bins',type=int,default=5); p.add_argument('--out',required=True); p.set_defaults(func=dk_cmd)
    p=sub.add_parser('dl'); add_common(p); p.add_argument('--models',required=True); p.add_argument('--n-bins',type=int,default=4); p.add_argument('--min-bin-n',type=int,default=64); p.add_argument('--out',required=True); p.set_defaults(func=dl_cmd)
    p=sub.add_parser('selftest'); p.set_defaults(func=selftest_cmd)
    a=ap.parse_args(); a.func(a)

if __name__=='__main__': main()
