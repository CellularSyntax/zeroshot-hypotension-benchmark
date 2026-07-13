import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
root="results"
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
            v=M[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                              color="white" if v<0.55 else "black",fontsize=8.5)
    ax.set_title(cohort,fontsize=11)
    # separate FM (0-3) vs trained (4-5) block with lines
    ax.axhline(3.5,color="#4dd",lw=1.5); ax.axvline(3.5,color="#4dd",lw=1.5)
cbar=fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02)
cbar.set_label("linear CKA (representational similarity)",fontsize=9)

fig.savefig("FigS_rsa_cka.png",dpi=150,bbox_inches="tight")
print("saved")
