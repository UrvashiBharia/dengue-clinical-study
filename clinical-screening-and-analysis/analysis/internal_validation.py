"""Internal validation of the WHOLE model-development procedure by bootstrap (Harrell / Steyerberg).

Each of 200 bootstrap samples repeats every step: univariable screening (p < 0.20, top 10),
exhaustive search of 1-3 predictor logistic models, selection of the best. The model chosen in the
bootstrap sample is then tested on the original data. Mean (bootstrap AUC - original-data AUC) is the
optimism; the calibration slope of the bootstrap model's linear predictor on the original data
estimates the shrinkage needed.

Run:  python analysis/internal_validation.py
"""
import itertools, warnings, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from common import load_data
warnings.filterwarnings("ignore")
df=load_data(); rng=np.random.default_rng(2024)
CAND=["oliguria","arthralgia","bleeding","rash","lethargy","nausea_vomiting","syncope","bodyache","headache","usg_binary","comorbidity","hct","plt","wbc","lymph_pct","alc","pt","ptt","days_fever"]
y_all=df.icu_or_tx.values.astype(float)
Xall=df[CAND].astype(float); Xall=Xall.fillna(Xall.median())
def screen(X,y):
    keep=[]
    for v in CAND:
        x=X[v].values
        if len(np.unique(x))<=2:
            t=pd.crosstab(x,y); 
            if t.shape!=(2,2): continue
            p=stats.fisher_exact(t.values)[1]
        else:
            p=stats.mannwhitneyu(x[y==1],x[y==0]).pvalue
        if p<0.20: keep.append((p,v))
    return [v for p,v in sorted(keep)][:10]
def fit(X,y,cols):
    try: return sm.Logit(y,sm.add_constant(X[cols].values,has_constant='add')).fit(disp=0,method='bfgs',maxiter=300)
    except Exception: return None
def develop(X,y):
    pool=screen(X,y); best=(-1,None,None)
    for k in (1,2,3):
        for c in itertools.combinations(pool,k):
            m=fit(X,y,list(c))
            if m is None: continue
            a=roc_auc_score(y,m.predict(sm.add_constant(X[list(c)].values,has_constant='add')))
            if a>best[0]: best=(a,list(c),m)
    return best
app_auc,cols,m=develop(Xall,y_all)
print("apparent selected:",cols,round(app_auc,3))
opt_auc=[];slopes=[]
for b in range(200):
    i=rng.integers(0,len(y_all),len(y_all)); Xb=Xall.iloc[i].reset_index(drop=True); yb=y_all[i]
    if yb.sum()<5: continue
    ab,cb,mb=develop(Xb,yb)
    lp=sm.add_constant(Xall[cb].values,has_constant='add')@mb.params
    a_orig=roc_auc_score(y_all,lp); opt_auc.append(ab-a_orig)
    cs=sm.Logit(y_all,sm.add_constant(lp)).fit(disp=0).params[1]; slopes.append(cs)
print(f"bootstrap (200) optimism in AUC: {np.mean(opt_auc):.3f} -> optimism-corrected AUC {app_auc-np.mean(opt_auc):.3f}")
print(f"bootstrap calibration slope (shrinkage): {np.mean(slopes):.2f}")
# points score: optimism of the fixed 2/2/1 score itself (no fitting) is zero for fixed weights; report bootstrap CI
sc=(2*df.bleeding.fillna(0)+2*df.usg_binary.fillna(0)+(df.days_fever>=4)).values
print("points score apparent AUC", round(roc_auc_score(y_all,sc),3))
