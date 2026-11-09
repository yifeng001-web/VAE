"""
Quick run: Figure 4-3 (porosity) and Figure 4-4 (S2) only.
Skips Figure 4-5 (PixelCNN generation) which is replaced by nanoGPT.
"""
import os, sys


from thesis_code.figure_scripts.figure_4_3_4_5_evaluation import (
    load_all_models,
    task_4_3_porosity,
    task_4_4_s2
)

def main():
    print("=" * 70)
    print("  Chapter 4 — Figure 4-3 (Porosity) & Figure 4-4 (S2) re-run")
    print("=" * 70)
    
    encoder, decoder, vq_layer, pixelcnn = load_all_models()
    
    real_por, recon_por = task_4_3_porosity(encoder, decoder, vq_layer)
    r_real, s2_real, r_recon, s2_recon = task_4_4_s2(encoder, decoder, vq_layer)
    
    print("\n" + "=" * 70)
    print("  Done! Figures saved to results/evaluation/")
    print("=" * 70)

if __name__ == "__main__":
    main()
