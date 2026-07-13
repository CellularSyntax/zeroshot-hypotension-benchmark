import numpy as np, csv, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
figs="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/manuscript/figs"
# --- UMAP (canonical CSV AUROC) ---
LABn={"TiRex-2":"tirex2","Chronos-Bolt":"chronos","TimesFM-2.5":"timesfm","Moirai-1.1-R":"moirai","TFT":"tft","PatchTST":"patchtst"}
AU={}
with open(f"{root}/suppl_probe_hypotension.csv") as f:
    for r in csv.DictReader(f): AU[(r["cohort"],LABn[r["model"]],int(r["horizon_min"]))]=float(r["AUROC"])
def umap(tag,cohort_csv,cohort_title,outfile):
    Z=dict(np.load(f"{root}/_expl_umaps_v2_{tag}.npz",allow_pickle=True)); tev=Z["tev"].astype(float)
    lab=lambda h:(np.isfinite(tev)&(tev<=h)).astype(int)
    order=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
    titles={"tirex2":"TiRex-2","chronos":"Chronos-Bolt","timesfm":"TimesFM-2.5","moirai":"Moirai-1.1-R","tft":"TFT (task-trained)","patchtst":"PatchTST (task-trained)"}
    rng=np.random.default_rng(0); GREY="#c3ccd6"; RED="#d1495b"
    fig,axes=plt.subplots(2,6,figsize=(17,6.2))
    for ri,h in enumerate([1,5]):
        y=lab(h)
        for ci,m in enumerate(order):
            ax=axes[ri,ci]; xy=Z[m]; o=rng.permutation(len(xy))
            ax.scatter(xy[o,0],xy[o,1],s=7,c=np.where(y[o]==1,RED,GREY),alpha=np.where(y[o]==1,0.9,0.45),linewidths=0,rasterized=True)
            if ri==0: ax.set_title(titles[m],fontsize=10)
            ax.text(0.5,0.965,f"AUROC {AU[(cohort_csv,m,h)]:.2f}",transform=ax.transAxes,ha="center",va="top",fontsize=8.5,color="#333")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            if ci==0: ax.set_ylabel(f"impending hypotension\n$\\leq${h} min  (n={int(y.sum())})",fontsize=9.5)
    leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor=GREY,markersize=8,label='no impending hypotension'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor=RED,markersize=8,label='impending hypotension at horizon')]
    fig.legend(handles=leg,loc="lower center",ncol=2,frameon=False,fontsize=10,bbox_to_anchor=(0.5,-0.015))
    fig.suptitle(f"{cohort_title}: per-window encoder representations (UMAP), coloured by impending hypotension",fontsize=12.5,y=1.0)
    fig.tight_layout(rect=[0,0.035,1,0.97]); fig.savefig(outfile,bbox_inches="tight"); plt.close(fig); print("saved",outfile)
umap("all2873","VitalDB","VitalDB",f"{figs}/FigS_umap_vitaldb.svg")
umap("mover_art","MOVER","MOVER (external)",f"{figs}/FigS_umap_mover.svg")
# --- RSA ---
RSA=json.load(open(f"{root}/_expl_rsa_both.json"))
LAB={"tirex2":"TiRex-2","chronos":"Chronos","timesfm":"TimesFM","moirai":"Moirai","tft":"TFT","patchtst":"PatchTST"}
fig,axes=plt.subplots(1,2,figsize=(12,5.2))
for ax,(tag,cohort) in zip(axes,[("all2873","VitalDB"),("mover_art","MOVER (external)")]):
    M=np.array(RSA[tag]["cka"]); labs=[LAB[m] for m in RSA[tag]["models"]]
    im=ax.imshow(M,vmin=0,vmax=1,cmap="magma")
    ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs,rotation=45,ha="right",fontsize=9); ax.set_yticklabels(labs,fontsize=9)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v=M[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",color="white" if v<0.55 else "black",fontsize=8.5)
    ax.set_title(cohort,fontsize=11); ax.axhline(3.5,color="#4dd",lw=1.5); ax.axvline(3.5,color="#4dd",lw=1.5)
cbar=fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02); cbar.set_label("linear CKA (representational similarity)",fontsize=9)
fig.suptitle("Cross-model representational similarity (linear CKA) on identical windows",fontsize=12.5,y=1.0)
fig.savefig(f"{figs}/FigS_rsa_cka.svg",bbox_inches="tight"); print("saved rsa svg")
