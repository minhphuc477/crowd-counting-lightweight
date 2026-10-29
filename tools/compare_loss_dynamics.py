import pandas as pd

df70 = pd.read_csv('runs/sha_a/rmr_v70/v70_track_b_exp4b_no_solver/train_log.csv')
df71_1 = pd.read_csv('runs/sha_a/rmr_v71/v71_track_b_exp1_pure_radon_ot_l1_sub50/train_log.csv')
df71_4 = pd.read_csv('runs/sha_a/rmr_v71/v71_track_b_exp4_full_synergy_sub50/train_log.csv')

print('=== V70 Exp 4b Loss Breakdown (lambda_l1=0, lambda_bayes=0.15) ===')
for ep in [1, 50, 100, 300, 500, 1000, 1495]:
    sub = df70[df70['epoch'] == ep]
    if len(sub) == 0: continue
    r = sub.iloc[0]
    print(f"Epoch {ep:4d}: Total={r['train_total']:.3f} | CountNB={r['train_count']:.3f} | RegionNB={r['train_region_nb']:.3f} | RadonOT={r['train_radon_ot']:.4f} | Bayes={r['train_bayesian']:.4f} | ValMAE={r['val_mae']:.2f}")

print('\n=== V71 Exp 1 Loss Breakdown (lambda_l1=0.2, lambda_bayes=0.0, Direct) ===')
for ep in [1, 50, 100, 300, 500, 1000, 1500]:
    sub = df71_1[df71_1['epoch'] == ep]
    if len(sub) == 0: continue
    r = sub.iloc[0]
    c_l1 = r['train_count_l1']
    print(f"Epoch {ep:4d}: Total={r['train_total']:.3f} | L1_term={0.2*c_l1:.2f} (raw={c_l1:.1f}) | CountNB={r['train_count']:.3f} | RegionNB={r['train_region_nb']:.3f} | RadonOT={r['train_radon_ot']:.4f} | ValMAE={r['val_mae']:.2f}")

print('\n=== V71 Exp 4 Loss Breakdown (lambda_l1=0.2, lambda_bayes=0.0, PC-BPRO Solver) ===')
for ep in [1, 50, 100, 300, 500, 1000, 1500]:
    sub = df71_4[df71_4['epoch'] == ep]
    if len(sub) == 0: continue
    r = sub.iloc[0]
    c_l1 = r['train_count_l1']
    print(f"Epoch {ep:4d}: Total={r['train_total']:.3f} | L1_term={0.2*c_l1:.2f} (raw={c_l1:.1f}) | CountNB={r['train_count']:.3f} | RegionNB={r['train_region_nb']:.3f} | RadonOT={r['train_radon_ot']:.4f} | ValMAE={r['val_mae']:.2f}")
