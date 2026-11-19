import argparse
from pathlib import Path
import numpy as np
from dlmd_e2.data import ExerciseData
from dlmd_e2.model import LinearHead
from dlmd_e2.metrics import auroc
from dlmd_e2.resample import Bootstrap
from dlmd_e2.report import ResultsTable

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--out',default='results.csv'); p.add_argument('--model',default='model.npz'); p.add_argument('--quick',action='store_true'); args=p.parse_args()
    d=ExerciseData(args.data); y=d.y; X=d.features
    tr=np.flatnonzero(d.split=='train'); a=np.flatnonzero(d.split=='test'); b=np.flatnonzero(d.split=='external')
    h=LinearHead().fit(X[tr],y[tr]); r=ResultsTable(); boot=Bootstrap()
    def score(name,rows,pred,site):
        r.add_bootstrap(name,boot.run(lambda i: auroc(y[rows][i],pred[i]),len(rows)),split=site)
    def delta(name,rows,p1,p2,site):
        r.add_bootstrap(name,boot.run(lambda i: auroc(y[rows][i],p1[i])-auroc(y[rows][i],p2[i]),len(rows)),split=site)
    pa=h.predict_proba(X[a]); pb=h.predict_proba(X[b]); score('baseline',a,pa,'A'); score('baseline',b,pb,'B')
    rng=np.random.default_rng(0); v=[]
    for _ in range(1000):
        i=rng.integers(len(a),size=len(a)); j=rng.integers(len(b),size=len(b)); v.append(auroc(y[a][i],pa[i])-auroc(y[b][j],pb[j]))
    r.add_bootstrap('baseline_drop',(np.mean(v),np.std(v,ddof=1),*np.percentile(v,[2.5,97.5])))
    try:
        from dlmd_e2.encoder import Encoder
        enc=Encoder(Path(args.data)/'encoder/resnet18.onnx',threads=2)
        def features(images): return enc.features(images,progress=False)
    except ModuleNotFoundError:
        import cv2
        cv2.setNumThreads(2); net=cv2.dnn.readNetFromONNX(str(Path(args.data)/'encoder/resnet18.onnx'))
        def features(images):
            out=[]
            for im in images:
                net.setInput(im[None,None].astype(np.float32)/255); out.append(net.forward().mean((2,3))[0])
            return np.stack(out)
    def mask(im,side):
        im=im.copy(); im[-32:, :32 if side=='left' else 0]=0
        if side=='right': im[-32:,-32:]=0
        return im
    if args.quick:
        rng=np.random.default_rng(42); a=np.sort(rng.choice(a,100,replace=False)); b=np.sort(rng.choice(b,100,replace=False)); tr=np.sort(rng.choice(tr,300,replace=False))
    rows=np.concatenate([tr,a,b]); F=features(mask(d.image(i),'left') for i in rows); nt=len(tr); na=len(a); fa=F[nt:nt+na]; fb=F[nt+na:]; final=LinearHead().fit(F[:nt],y[tr]); final.save(args.model)
    for site,ix,f in [('A',a,fa),('B',b,fb)]:
        orig=h.predict_proba(X[ix]); left=h.predict_proba(f); right=h.predict_proba(features(mask(d.image(i),'right') for i in ix)); new=final.predict_proba(f)
        score('original_evaluation_subset',ix,orig,site); score('left_mask',ix,left,site); score('right_mask',ix,right,site); score('retrained_left_mask',ix,new,site)
        delta('left_mask_change',ix,left,orig,site); delta('right_mask_change',ix,right,orig,site); delta('retrained_change',ix,new,orig,site)
    print(r); r.write(args.out)
if __name__=='__main__': main()
