import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root="results"
# canonical AUROC from the supplementary CSV (single source of truth, figure-style 1.7)
LABn={"TiRex-2":"tirex2","Chronos-Bolt":"chronos","TimesFM-2.5":"timesfm","Moirai-1.1-R":"moirai","TFT":"tft","PatchTST":"patchtst"}
AU={}  # AU[(cohort,model_key,horizon)] = auroc
with open(f"{root}/suppl_probe_hypotension.csv") as f:
    for r in csv.DictReader(f):
        AU[(r["cohort"],LABn[r["model"]],int(r["horizon_min"]))]=float(r["AUROC"])
def render(tag, cohort_csv, cohort_title, outfile):
    Z=dict(np.load(f"{root}/_expl_umaps_v2_{tag}.npz",allow_pickle=True))
    tev=Z["tev"].astype(float)
    lab=lambda h:(np.isfinite(tev)&(tev<=h)).astype(int)
    order=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
    titles={"tirex2":"TiRex-2","chronos":"Chronos-Bolt","timesfm":"TimesFM-2.5",
            "moirai":"Moirai-1.1-R","tft":"TFT (task-trained)","patchtst":"PatchTST (task-trained)"}
    rng=np.random.default_rng(0); GREY="#c3ccd6"; RED="#d1495b"
    fig,axes=plt.subplots(2,6,figsize=(17,6.2))
    for ri,h in enumerate([1,5]):
        y=lab(h)
        for ci,m in enumerate(order):
            ax=axes[ri,ci]; xy=Z[m]; o=rng.permutation(len(xy))
            ax.scatter(xy[o,0],xy[o,1],s=7,c=np.where(y[o]==1,RED,GREY),
                       alpha=np.where(y[o]==1,0.9,0.45),linewidths=0,rasterized=True)
            au=AU[(cohort_csv,m,h)]
            if ri==0: ax.set_title(titles[m],fontsize=10)
            ax.text(0.5,0.965,f"AUROC {au:.2f}",transform=ax.transAxes,ha="center",va="top",fontsize=8.5,color="#333")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            if ci==0:
                ax.set_ylabel(f"impending hypotension\n$\\leq${h} min  (n={int(y.sum())})",fontsize=9.5)
                if ri==0:
                    ax.annotate("",xy=(0.16,0.02),xytext=(0.02,0.02),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",color="#888",lw=0.9))
                    ax.annotate("",xy=(0.02,0.16),xytext=(0.02,0.02),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",color="#888",lw=0.9))
                    ax.text(0.17,0.02,"UMAP-1",transform=ax.transAxes,fontsize=6.5,color="#888",va="center")
                    ax.text(0.03,0.17,"UMAP-2",transform=ax.transAxes,fontsize=6.5,color="#888",rotation=90,ha="center")
    leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor=GREY,markersize=8,label='no impending hypotension'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor=RED,markersize=8,label='impending hypotension at horizon')]
    fig.legend(handles=leg,loc="lower center",ncol=2,frameon=False,fontsize=10,bbox_to_anchor=(0.5,-0.015))
    fig.suptitle(f"{cohort_title}: per-window encoder representations (UMAP), coloured by impending hypotension",fontsize=12.5,y=1.0)
    fig.tight_layout(rect=[0,0.035,1,0.97])
    fig.savefig(outfile,dpi=150,bbox_inches="tight"); plt.close(fig)
    print("saved",outfile)
base="."
render("all2873","VitalDB","VitalDB",f"{base}/FigS_umap_vitaldb.png")
render("mover_art","MOVER","MOVER (external)",f"{base}/FigS_umap_mover.png")
# verify a few against CSV
for k in [("VitalDB","tirex2",1),("VitalDB","tirex2",5),("VitalDB","timesfm",5),("VitalDB","moirai",1)]:
    print("CSV",k,"->",AU[k])
