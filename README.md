# World Cup 2026 — Match Predictor (PRODE 1X2)

Predictor de partidos para el **PRODE del Mundial 2026** (pronóstico 1 / X / 2:
gana local, empate, gana visitante). Entrena un modelo científico en Python,
genera un `model.json` pre-entrenado y lo sirve en un frontend HTML puro que
funciona offline.

> Pensado para un PRODE familiar/entre amigos. No es un producto comercial ni
> asesoramiento de apuestas.

---

## Cómo funciona

Entrenado sobre **~11.000 partidos internacionales (2015–2026, 295 selecciones)**
del dataset público de [martj42](https://github.com/martj42/international_results).

El motor combina dos modelos y los calibra:

1. **Dixon-Coles** (Poisson bivariado, Dixon & Coles 1997) ajustado por
   **máxima verosimilitud penalizada** con `scipy.optimize` (L-BFGS-B),
   **decaimiento temporal** exponencial (los partidos recientes pesan más),
   parámetros separados de **ataque/defensa** por equipo, ventaja de local y la
   corrección `rho` para marcadores bajos.
2. **Modelo de features (ML)** sobre Elo, forma de los últimos 5 partidos,
   head-to-head e importancia del torneo. Regresión logística multinomial en
   NumPy (o `HistGradientBoosting` de scikit-learn si está instalado).
3. **Calibración por temperatura** y **ensemble** ponderado, con el peso elegido
   sobre un tramo de validación.

Las probabilidades se calculan en **forma cerrada** (sin Monte Carlo): exactas,
deterministas e instantáneas.

### Métricas honestas (holdout temporal)

Evaluado sobre los **~1.980 partidos más recientes**, nunca vistos en
entrenamiento (sin fuga de datos). Valores reales del último entrenamiento:

| Métrica            | Valor   |
|--------------------|---------|
| Precisión 1X2      | ~60%    |
| Brier multiclase   | ~0.50   |
| Log-loss           | ~0.86   |

El modelo está **bien calibrado**: cuando dice 66% acierta ~71%, cuando dice 79%
acierta ~88% (ver tabla de calibración en [`analysis.txt`](analysis.txt)).

El techo realista para 1X2 internacional con datos públicos ronda **70–73%**
(los mejores modelos del mundo usan alineaciones y estado físico). Las cifras
reflejan poder predictivo genuino fuera de muestra, no ajuste en muestra.

---

## Uso

### Predecir (frontend)

Abrir [`index.html`](index.html). Tiene dos pestañas:

- **🎯 Predictor**: elegir local y visitante, opcionalmente marcar **"sede
  neutral"** (recomendado para el Mundial), y predecir. Muestra probabilidades
  1/X/2, goles esperados (λ) y la recomendación PRODE con nivel de confianza.
- **📅 Fixture**: el calendario oficial del Mundial 2026 (fase de grupos, 72
  partidos ordenados por fecha) con la predicción del modelo en cada partido.

Ver además el grid de pronósticos pre-calculados en
[`predictions_viewer.html`](predictions_viewer.html).

> Los navegadores bloquean `fetch` sobre `file://`. Servir la carpeta con un
> servidor estático, p. ej.: `py -m http.server 8000` y abrir
> `http://localhost:8000/index.html`.

### Reentrenar

```bash
pip install numpy scipy pandas          # scikit-learn es opcional
py train_advanced.py
```

Regenera `model/model.json`, `predictions.json` y `analysis.txt`.

---

## Estructura

```
worldcup_predictor/
├── index.html               Frontend principal (predicción interactiva)
├── predictions_viewer.html  Grid de pronósticos pre-calculados
├── fixture.json             Fixture oficial de grupos WC2026 + predicciones — generado
├── predictions.json         Top-200 pronósticos (sede neutral) — generado
├── analysis.txt             Reporte estadístico — generado
├── PRODE_GUIDE.txt          Guía de estrategia para el PRODE
├── train_advanced.py        Motor de entrenamiento (única fuente de verdad)
├── data/
│   ├── international_results.csv  Dataset principal (martj42, ~49k partidos 1872–2026)
│   └── wc2026_recent15.csv        Dataset curado de respaldo (48 selecciones)
└── model/
    └── model.json           Modelo pre-entrenado que lee el frontend — generado
```

### Formato del dataset

El entrenador detecta el esquema automáticamente y soporta dos formatos
(usa `international_results.csv` si existe, filtrado a 2015+; si no, el curado):

```
# match-centric (martj42):
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
# team-centric (curado, respaldo):
team, date, opponent, goals_scored, goals_conceded, result, tournament, venue
```

Para actualizar los datos, volver a bajar el CSV de martj42 a
`data/international_results.csv` y reentrenar.

### Contrato de `model.json` (camelCase)

```
lambda_home = exp(teamStrengthHome[home] + teamStrengthAway[away] + homeAdvantage)
lambda_away = exp(teamStrengthHome[away] + teamStrengthAway[home])
```

donde `teamStrengthHome` = ataque y `teamStrengthAway` = −defensa. El archivo
incluye además `rho`, `calibrationTemperature`, `eloRatings`, `validation` y
`matchProbabilities`: probabilidades pre-calculadas del ensemble para cada cruce
entre las 48 selecciones, en forma compacta `[p1, pX, p2, p1n, pXn, p2n]` (las
tres últimas = sede neutral). Se serializa sin indentar para carga rápida
(~126 KB).
