/**
 * =============================================================================
 *  TRAIN.JS — Entrenador standalone del modelo Dixon-Coles
 * =============================================================================
 *  Uso:   node train.js
 *  Input: Descarga results.csv de GitHub automáticamente
 *  Output: model.json + wc2026_recent15.csv
 * =============================================================================
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import https from 'https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============================================================================
// IMPORTAR MOTOR (inline para evitar problemas de ESM)
// ============================================================================

// ---- Constantes ----
const MODEL_CONFIG = {
  CUTOFF_DATE: '2021-01-01',
  XI: 0.0025,
  TOURNAMENT_WEIGHTS: {
    'FIFA World Cup': 1.5,
    'FIFA World Cup qualification': 1.2,
    'Copa América': 1.1,
    'UEFA Euro': 1.1,
    'UEFA Euro qualification': 1.05,
    'AFC Asian Cup': 1.1,
    'AFC Asian Cup qualification': 1.05,
    'African Cup of Nations': 1.1,
    'African Cup of Nations qualification': 1.05,
    'CONCACAF Gold Cup': 1.05,
    'UEFA Nations League': 1.05,
  },
  LEARNING_RATE: 0.0005,
  MAX_ITERATIONS: 800,
  LAMBDA_REG: 0.001,
  REFERENCE_DATE: new Date().toISOString().split('T')[0],
};

const TEAM_NAME_MAP = {
  'Korea Republic': 'South Korea',
  'Republic of Korea': 'South Korea',
  'IR Iran': 'Iran',
  'Congo DR': 'DR Congo',
  "Côte d'Ivoire": 'Ivory Coast',
  "Cote d'Ivoire": 'Ivory Coast',
  'Türkiye': 'Turkey',
  'Czechia': 'Czech Republic',
  'Cape Verde': 'Cabo Verde',
  'Curacao': 'Curaçao',
  'USA': 'United States',
};

const WC2026_TEAMS_NORMALIZED = [
  'Canada', 'Mexico', 'United States',
  'Australia', 'Iraq', 'Iran', 'Japan', 'Jordan', 'South Korea', 'Qatar', 'Saudi Arabia', 'Uzbekistan',
  'Algeria', 'Cabo Verde', 'DR Congo', 'Ivory Coast', 'Egypt', 'Ghana', 'Morocco', 'Senegal', 'South Africa', 'Tunisia',
  'Curaçao', 'Haiti', 'Panama',
  'Argentina', 'Brazil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
  'New Zealand',
  'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Croatia', 'Czech Republic',
  'England', 'France', 'Germany', 'Netherlands', 'Norway', 'Portugal',
  'Scotland', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
];

function normalizeTeamName(name) {
  return TEAM_NAME_MAP[name] || name;
}

// ---- Parseo CSV ----
function parseCSV(csvText) {
  const lines = csvText.trim().split('\n');
  const matches = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = parseCSVLine(lines[i]);
    if (parts.length < 9) continue;
    const homeScore = parseInt(parts[3]);
    const awayScore = parseInt(parts[4]);
    if (isNaN(homeScore) || isNaN(awayScore)) continue;
    matches.push({
      date: parts[0],
      homeTeam: normalizeTeamName(parts[1]),
      awayTeam: normalizeTeamName(parts[2]),
      homeScore, awayScore,
      tournament: parts[5],
      city: parts[6],
      country: parts[7],
      neutral: parts[8].toUpperCase() === 'TRUE',
    });
  }
  return matches;
}

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') inQuotes = !inQuotes;
    else if (ch === ',' && !inQuotes) { result.push(current.trim()); current = ''; }
    else current += ch;
  }
  result.push(current.trim());
  return result;
}

// ---- Matemáticas ----
function logFactorial(n) {
  if (n <= 1) return 0;
  if (n <= 20) { let r = 0; for (let i = 2; i <= n; i++) r += Math.log(i); return r; }
  return n * Math.log(n) - n + 0.5 * Math.log(2 * Math.PI * n);
}

function logPoisson(k, lambda) {
  if (lambda <= 0) lambda = 1e-10;
  return k * Math.log(lambda) - lambda - logFactorial(k);
}

function dixonColesAdj(hg, ag, lh, la, rho) {
  let tau = 1.0;
  if (hg === 0 && ag === 0) tau = 1 - lh * la * rho;
  else if (hg === 0 && ag === 1) tau = 1 + lh * rho;
  else if (hg === 1 && ag === 0) tau = 1 + la * rho;
  else if (hg === 1 && ag === 1) tau = 1 - rho;
  return Math.max(1e-6, tau);
}

function matchLL(hg, ag, lh, la, rho) {
  const tau = dixonColesAdj(hg, ag, lh, la, rho);
  return logPoisson(hg, lh) + logPoisson(ag, la) + Math.log(tau);
}

// ---- Modelo ----
class DixonColesModel {
  constructor() {
    this.teams = {};
    this.homeAdvantage = 0.25;
    this.rho = -0.05;
    this.teamList = [];
    this.trainingMatches = [];
    this.allMatches = [];
  }

  prepareData(matches) {
    this.allMatches = matches;
    const cutoff = MODEL_CONFIG.CUTOFF_DATE;
    const filtered = matches.filter(m => m.date >= cutoff);
    const teamSet = new Set();
    filtered.forEach(m => { teamSet.add(m.homeTeam); teamSet.add(m.awayTeam); });
    this.teamList = Array.from(teamSet).sort();
    this.teams = {};
    this.teamList.forEach(t => { this.teams[t] = { attack: 0.0, defense: 0.0 }; });

    const refDate = new Date(MODEL_CONFIG.REFERENCE_DATE);
    this.trainingMatches = filtered.map(m => {
      const daysDiff = (refDate - new Date(m.date)) / (1000 * 60 * 60 * 24);
      const tw = Math.exp(-MODEL_CONFIG.XI * Math.max(daysDiff, 0));
      const trnw = MODEL_CONFIG.TOURNAMENT_WEIGHTS[m.tournament] || 1.0;
      return { ...m, weight: tw * trnw };
    });

    return { totalMatches: this.trainingMatches.length, totalTeams: this.teamList.length };
  }

  getLambda(atkTeam, defTeam, isHome) {
    const atk = this.teams[atkTeam]?.attack || 0;
    const def = this.teams[defTeam]?.defense || 0;
    const ha = isHome ? this.homeAdvantage : 0;
    const exponent = Math.max(-20, Math.min(20, atk - def + ha));
    return Math.exp(exponent);
  }

  totalLogLikelihood() {
    let total = 0;
    for (const m of this.trainingMatches) {
      if (!this.teams[m.homeTeam] || !this.teams[m.awayTeam]) continue;
      const lh = this.getLambda(m.homeTeam, m.awayTeam, !m.neutral);
      const la = this.getLambda(m.awayTeam, m.homeTeam, false);
      const ll = matchLL(m.homeScore, m.awayScore, lh, la, this.rho);
      if (!isFinite(ll)) continue;
      total += m.weight * ll;
    }
    let reg = 0;
    for (const t of this.teamList) { reg += this.teams[t].attack ** 2 + this.teams[t].defense ** 2; }
    const regularized = total - MODEL_CONFIG.LAMBDA_REG * reg;
    return Number.isFinite(regularized) ? regularized : -1e18;
  }

  optimizationStep() {
    const eps = 1e-5;
    const lr = MODEL_CONFIG.LEARNING_RATE;
    const baseLL = this.totalLogLikelihood();

    for (const tn of this.teamList) {
      this.teams[tn].attack += eps;
      const ga = (this.totalLogLikelihood() - baseLL) / eps;
      this.teams[tn].attack -= eps;

      this.teams[tn].defense += eps;
      const gd = (this.totalLogLikelihood() - baseLL) / eps;
      this.teams[tn].defense -= eps;

      this.teams[tn].attack += lr * ga;
      this.teams[tn].defense += lr * gd;
    }

    this.homeAdvantage += eps;
    let gha = (this.totalLogLikelihood() - baseLL) / eps;
    if (!isFinite(gha)) gha = 0;
    this.homeAdvantage -= eps;
    this.homeAdvantage += lr * gha;
    this.homeAdvantage = Math.max(0.01, Math.min(0.8, this.homeAdvantage));

    this.rho += eps;
    let grho = (this.totalLogLikelihood() - baseLL) / eps;
    if (!isFinite(grho)) grho = 0;
    this.rho -= eps;
    this.rho += lr * 0.1 * grho;
    this.rho = Math.max(-0.4, Math.min(0.4, this.rho));

    const avgDef = this.teamList.reduce((s, t) => s + this.teams[t].defense, 0) / this.teamList.length;
    for (const t of this.teamList) this.teams[t].defense -= avgDef;

    return this.totalLogLikelihood();
  }

  train(onProgress) {
    const maxIter = MODEL_CONFIG.MAX_ITERATIONS;
    let bestLL = this.totalLogLikelihood();
    if (!isFinite(bestLL)) bestLL = -1e18;
    let noImprove = 0;

    for (let i = 0; i < maxIter; i++) {
      const ll = this.optimizationStep();
      if (!isFinite(ll)) {
        if (onProgress) onProgress(i, maxIter, NaN);
        console.warn('[TRAIN] deteniendo porque LL no es finito en la iteración', i);
        break;
      }
      if (onProgress && i % 20 === 0) onProgress(i, maxIter, ll);
      if (ll > bestLL + 0.01) {
        bestLL = ll;
        noImprove = 0;
      } else {
        noImprove++;
        if (noImprove > 100) break;
      }
    }
    return bestLL;
  }

  getRecentMatches(teamName, n = 15) {
    return this.allMatches
      .filter(m => m.homeTeam === teamName || m.awayTeam === teamName)
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, n)
      .map(m => {
        const isHome = m.homeTeam === teamName;
        return {
          date: m.date,
          team: teamName,
          opponent: isHome ? m.awayTeam : m.homeTeam,
          goals_scored: isHome ? m.homeScore : m.awayScore,
          goals_conceded: isHome ? m.awayScore : m.homeScore,
          result: isHome
            ? (m.homeScore > m.awayScore ? 'W' : m.homeScore < m.awayScore ? 'L' : 'D')
            : (m.awayScore > m.homeScore ? 'W' : m.awayScore < m.homeScore ? 'L' : 'D'),
          tournament: m.tournament,
          venue: isHome ? 'Home' : 'Away',
        };
      });
  }

  exportModel() {
    const wcTeams = {};
    const wcSet = new Set(WC2026_TEAMS_NORMALIZED);
    for (const [name, ratings] of Object.entries(this.teams)) {
      if (wcSet.has(name)) {
        wcTeams[name] = {
          attack: Math.round(ratings.attack * 10000) / 10000,
          defense: Math.round(ratings.defense * 10000) / 10000,
        };
      }
    }

    return {
      version: '3.0',
      trained_at: new Date().toISOString(),
      config: {
        cutoff_date: MODEL_CONFIG.CUTOFF_DATE,
        xi_decay: MODEL_CONFIG.XI,
        iterations: MODEL_CONFIG.MAX_ITERATIONS,
      },
      matches_used: this.trainingMatches.length,
      total_teams: this.teamList.length,
      parameters: {
        home_advantage: Math.round(this.homeAdvantage * 10000) / 10000,
        rho: Math.round(this.rho * 10000) / 10000,
      },
      wc2026_teams: wcTeams,
      recent_matches: Object.fromEntries(
        WC2026_TEAMS_NORMALIZED.map(team => [team, this.getRecentMatches(team, 15)])
      ),
      all_teams: Object.fromEntries(
        Object.entries(this.teams).map(([k, v]) => [k, {
          attack: Math.round(v.attack * 10000) / 10000,
          defense: Math.round(v.defense * 10000) / 10000,
        }])
      ),
    };
  }
}

// ============================================================================
// DESCARGA DE CSV
// ============================================================================
function downloadCSV(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': 'WC-Predictor/3.0' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return downloadCSV(res.headers.location).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}`));
        return;
      }
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(data));
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

// ============================================================================
// MAIN
// ============================================================================
async function main() {
  console.log('='.repeat(60));
  console.log('  DIXON-COLES TRAINER — World Cup 2026');
  console.log('='.repeat(60));

  // Step 1: Get CSV data
  const csvPath = join(__dirname, '..', 'data', 'results.csv');
  let csvText;

  if (existsSync(csvPath)) {
    console.log(`\n[DATA] Leyendo CSV local: ${csvPath}`);
    csvText = readFileSync(csvPath, 'utf-8');
  } else {
    console.log('\n[DATA] Descargando CSV de GitHub...');
    const url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv';
    csvText = await downloadCSV(url);
    console.log(`[DATA] Descargado (${(csvText.length / 1024 / 1024).toFixed(1)} MB)`);
  }

  // Step 2: Parse
  console.log('[DATA] Parseando CSV...');
  const matches = parseCSV(csvText);
  console.log(`[DATA] ${matches.length.toLocaleString()} partidos internacionales cargados`);

  // Step 3: Prepare model
  const model = new DixonColesModel();
  const stats = model.prepareData(matches);
  console.log(`\n[MODEL] Datos de entrenamiento: ${stats.totalMatches.toLocaleString()} partidos, ${stats.totalTeams} equipos`);
  console.log('[MODEL] Log-likelihood inicial:', model.totalLogLikelihood());

  // Step 4: Train
  console.log('[MODEL] Entrenando Dixon-Coles...\n');
  const startTime = Date.now();
  const finalLL = model.train((iter, total, ll) => {
    process.stdout.write(`\r  Iteración ${iter}/${total} | LL = ${ll.toFixed(2)}`);
  });
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n\n[MODEL] Entrenamiento completo en ${elapsed}s`);
  console.log(`[MODEL] Log-Likelihood final: ${finalLL.toFixed(4)}`);
  console.log(`[MODEL] Home Advantage (γ): ${model.homeAdvantage.toFixed(4)}`);
  console.log(`[MODEL] Rho (ρ): ${model.rho.toFixed(4)}`);

  // Step 5: Export model.json
  const exported = model.exportModel();
  const modelPath = join(__dirname, 'model.json');
  writeFileSync(modelPath, JSON.stringify(exported, null, 2), 'utf-8');
  console.log(`\n[EXPORT] model.json guardado → ${modelPath}`);
  console.log(`[EXPORT] ${Object.keys(exported.wc2026_teams).length} equipos WC2026 encontrados`);

  // Step 6: Generate recent 15 CSV
  console.log('\n[CSV] Generando últimos 15 partidos por selección...');
  let csvOutput = 'team,date,opponent,goals_scored,goals_conceded,result,tournament,venue\n';
  let teamCount = 0;

  for (const team of WC2026_TEAMS_NORMALIZED) {
    const recent = model.getRecentMatches(team, 15);
    if (recent.length > 0) {
      teamCount++;
      for (const m of recent) {
        csvOutput += `"${m.team}","${m.date}","${m.opponent}",${m.goals_scored},${m.goals_conceded},${m.result},"${m.tournament}","${m.venue}"\n`;
      }
    }
  }

  const dataDir = join(__dirname, '..', 'data');
  if (!existsSync(dataDir)) mkdirSync(dataDir, { recursive: true });
  const recentCsvPath = join(dataDir, 'wc2026_recent15.csv');
  writeFileSync(recentCsvPath, csvOutput, 'utf-8');
  console.log(`[CSV] wc2026_recent15.csv guardado → ${recentCsvPath}`);
  console.log(`[CSV] ${teamCount} selecciones · ${csvOutput.split('\n').length - 2} registros`);

  // Step 7: Show top teams
  console.log('\n' + '='.repeat(60));
  console.log('  TOP 15 SELECCIONES POR RATING DE ATAQUE');
  console.log('='.repeat(60));
  const sorted = Object.entries(exported.wc2026_teams)
    .sort((a, b) => b[1].attack - a[1].attack);

  sorted.slice(0, 15).forEach(([name, r], i) => {
    const rank = String(i + 1).padStart(2);
    const atkBar = '█'.repeat(Math.max(0, Math.round((r.attack + 0.5) * 20)));
    console.log(`  ${rank}. ${name.padEnd(25)} ATK: ${r.attack.toFixed(4).padStart(8)}  DEF: ${r.defense.toFixed(4).padStart(8)}  ${atkBar}`);
  });

  console.log('\n' + '='.repeat(60));
  console.log('  ✓ Entrenamiento completado exitosamente');
  console.log('  → model/model.json       (parámetros del modelo)');
  console.log('  → data/wc2026_recent15.csv (últimos 15 partidos)');
  console.log('='.repeat(60) + '\n');
}

main().catch(err => {
  console.error('\n[ERROR]', err.message);
  process.exit(1);
});
