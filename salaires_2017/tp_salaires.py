import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. DATASET ────────────────────────────────────────────────────────────────
data = {
    'annee': [
        2008,2008,2008,
        2009,2009,2009,
        2010,2010,2010,
        2011,2011,2011,
        2012,2012,2012,
        2013,2013,2013,
        2014,2014,2014,
    ] * 4,
    'genre': ['Femmes','Hommes','Ensemble'] * 7 * 4,
    'csp': (
        ['Cadres'] * 21 +
        ['Professions_intermediaires'] * 21 +
        ['Employes'] * 21 +
        ['Ouvriers'] * 21
    ),
    'salaire': [
        # Cadres
        3184.29, 4265.84, 3909.41,
        3199.26, 4187.46, 3861.81,
        3312.91, 4278.44, 3964.05,
        3361.43, 4300.52, 3986.60,
        3459.00, 4399.00, 4083.00,
        3462.00, 4376.00, 4066.00,
        3524.00, 4407.00, 4104.00,
        # Professions intermédiaires
        1907.61, 2242.68, 2102.05,
        1928.45, 2240.47, 2105.59,
        1978.84, 2267.71, 2142.71,
        2007.15, 2305.57, 2178.68,
        2054.00, 2380.00, 2241.00,
        2068.00, 2393.00, 2253.00,
        2082.00, 2419.00, 2272.00,
        # Employés
        1422.42, 1564.40, 1463.40,
        1441.83, 1578.87, 1481.57,
        1465.18, 1595.80, 1502.82,
        1507.06, 1643.99, 1547.10,
        1551.00, 1700.00, 1596.00,
        1568.00, 1714.00, 1612.00,
        1584.00, 1739.00, 1631.00,
        # Ouvriers
        1287.83, 1574.40, 1528.83,
        1312.91, 1591.45, 1547.12,
        1343.43, 1612.16, 1569.00,
        1380.09, 1671.76, 1624.83,
        1434.00, 1727.00, 1677.00,
        1441.00, 1735.00, 1686.00,
        1462.00, 1750.00, 1702.00,
    ]
}

df = pd.DataFrame(data)
# Garder seulement Femmes et Hommes (pas Ensemble, qui est une combinaison)
df = df[df['genre'] != 'Ensemble'].reset_index(drop=True)

# ── 2. MODÈLE – encodage one-hot ──────────────────────────────────────────────
df_enc = pd.get_dummies(df, columns=['genre', 'csp'], drop_first=False)
X = df_enc.drop('salaire', axis=1)
y = df_enc['salaire']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. APPRENTISSAGE ──────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = np.mean(np.abs(y_test - y_pred))

print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.1f} €")
print(f"MAE  = {mae:.1f} €")

# ── 4. DESCENTE DE GRADIENT (implémentation manuelle) ─────────────────────────
X_arr = X_train.values.astype(float)
y_arr = y_train.values.astype(float)
X_arr_norm = (X_arr - X_arr.mean(axis=0)) / (X_arr.std(axis=0) + 1e-8)
X_b = np.c_[np.ones(X_arr_norm.shape[0]), X_arr_norm]

theta = np.zeros(X_b.shape[1])
alpha = 0.01
n_epochs = 300
cost_history = []

for _ in range(n_epochs):
    y_hat = X_b @ theta
    error = y_hat - y_arr
    gradient = (2 / len(y_arr)) * X_b.T @ error
    theta -= alpha * gradient
    cost_history.append(np.mean(error ** 2))

# ── 5. GRAPHIQUES ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("TP Machine Learning – Salaires 2008-2014", fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLORS = {
    'Cadres':                      '#6366f1',
    'Professions_intermediaires':  '#14b8a6',
    'Employes':                    '#f59e0b',
    'Ouvriers':                    '#ef4444',
}

# ── Graphe 1 : Évolution des salaires par CSP ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
annees = sorted(df['annee'].unique())
for csp, color in COLORS.items():
    vals_f = [df[(df['annee']==a)&(df['csp']==csp)&(df['genre']=='Femmes')]['salaire'].values[0] for a in annees]
    vals_h = [df[(df['annee']==a)&(df['csp']==csp)&(df['genre']=='Hommes')]['salaire'].values[0] for a in annees]
    label = csp.replace('_', ' ')
    ax1.plot(annees, vals_f, color=color, linestyle='--', marker='o', markersize=4, linewidth=1.5, label=f'{label} (F)')
    ax1.plot(annees, vals_h, color=color, linestyle='-',  marker='s', markersize=4, linewidth=1.5, label=f'{label} (H)')
ax1.set_title("Évolution salaires par CSP", fontsize=11, fontweight='bold')
ax1.set_xlabel("Année")
ax1.set_ylabel("Salaire net moyen (€)")
ax1.legend(fontsize=6, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(annees)

# ── Graphe 2 : Valeurs réelles vs prédites ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_pred, color='#6366f1', alpha=0.7, edgecolors='white', s=60, zorder=3)
lims = [min(y_test.min(), y_pred.min()) - 100, max(y_test.max(), y_pred.max()) + 100]
ax2.plot(lims, lims, 'r--', linewidth=1.5, label='Prédiction parfaite')
ax2.set_title("Réel vs Prédit", fontsize=11, fontweight='bold')
ax2.set_xlabel("Salaire réel (€)")
ax2.set_ylabel("Salaire prédit (€)")
ax2.text(0.05, 0.92, f"R² = {r2:.3f}\nRMSE = {rmse:.0f} €", transform=ax2.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0ff', alpha=0.8))
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Graphe 3 : Résidus ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
residus = y_test.values - y_pred
ax3.scatter(y_pred, residus, color='#f59e0b', alpha=0.7, edgecolors='white', s=60, zorder=3)
ax3.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax3.set_title("Résidus", fontsize=11, fontweight='bold')
ax3.set_xlabel("Valeurs prédites (€)")
ax3.set_ylabel("Résidu (réel − prédit)")
ax3.grid(True, alpha=0.3)

# ── Graphe 4 : Convergence de la descente de gradient ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(range(1, n_epochs + 1), cost_history, color='#ef4444', linewidth=2)
ax4.set_title("Convergence – Descente de gradient", fontsize=11, fontweight='bold')
ax4.set_xlabel("Époque")
ax4.set_ylabel("MSE (coût J(θ))")
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# ── Graphe 5 : Importance des variables (coefficients) ────────────────────
ax5 = fig.add_subplot(gs[1, 1])
coefs = pd.Series(model.coef_, index=X.columns)
coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=True).index)
colors_bar = ['#ef4444' if v < 0 else '#6366f1' for v in coefs_sorted]
ax5.barh(coefs_sorted.index, coefs_sorted.values, color=colors_bar, edgecolor='white', height=0.6)
ax5.axvline(0, color='black', linewidth=0.8)
ax5.set_title("Coefficients du modèle", fontsize=11, fontweight='bold')
ax5.set_xlabel("Valeur du coefficient")
labels_clean = [l.replace('csp_','').replace('genre_','').replace('_',' ') for l in coefs_sorted.index]
ax5.set_yticklabels(labels_clean, fontsize=8)
ax5.grid(True, alpha=0.3, axis='x')

# ── Graphe 6 : Écart salarial F/H par CSP ─────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ecarts = {}
for csp in COLORS:
    vals_f = np.array([df[(df['annee']==a)&(df['csp']==csp)&(df['genre']=='Femmes')]['salaire'].values[0] for a in annees])
    vals_h = np.array([df[(df['annee']==a)&(df['csp']==csp)&(df['genre']=='Hommes')]['salaire'].values[0] for a in annees])
    ecarts[csp] = ((vals_f - vals_h) / vals_h * 100)

for csp, color in COLORS.items():
    ax6.plot(annees, ecarts[csp], color=color, marker='o', markersize=5,
             linewidth=2, label=csp.replace('_', ' '))
ax6.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax6.set_title("Écart salarial F/H par CSP (%)", fontsize=11, fontweight='bold')
ax6.set_xlabel("Année")
ax6.set_ylabel("(Salaire F − H) / H  (%)")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_xticks(annees)

plt.savefig('/mnt/user-data/outputs/tp_salaires_graphes.png', dpi=150, bbox_inches='tight')
print("Graphiques sauvegardés.")
