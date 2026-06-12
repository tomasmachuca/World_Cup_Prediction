/**
 * =============================================================================
 *  WORLD CUP 2026 PREDICTOR — Dual Model Engine
 * =============================================================================
 *  Model 1: Dixon-Coles (Poisson-based statistical model)
 *  Model 2: Elo + Logistic Regression (ML-based model)
 *
 *  Both models are trained independently on the same historical data
 *  and produce separate predictions for comparison.
 * =============================================================================
 */

// ============================================================================
// CONSTANTES DEL MODELO DIXON-COLES
// ============================================================================
const MODEL_CONFIG = {
  // Solo usar partidos desde esta fecha para entrenamiento
  CUTOFF_DATE: '2018-01-01',
  // Decaimiento temporal: exp(-XI * days_since_match)
  XI: 0.0019,
  // Torneos que reciben peso extra
  TOURNAMENT_WEIGHTS: {
    'FIFA World Cup': 2.5,
    'FIFA World Cup qualification': 1.4,
    'Copa América': 1.3,
    'UEFA Euro': 1.3,
    'UEFA Euro qualification': 1.1,
    'AFC Asian Cup': 1.2,
    'AFC Asian Cup qualification': 1.05,
    'African Cup of Nations': 1.2,
    'African Cup of Nations qualification': 1.05,
    'CONCACAF Gold Cup': 1.1,
    'CONCACAF Nations League': 1.05,
    'UEFA Nations League': 1.1,
    'Confederations Cup': 1.3,
    'Copa América qualification': 1.1,
    'Friendly': 0.6,
  },
  // Learning rate para gradient descent
  LEARNING_RATE: 0.003,
  // Iteraciones de optimización
  MAX_ITERATIONS: 3000,
  // Regularización L2
  LAMBDA_REG: 0.002,
  // Referencia temporal
  REFERENCE_DATE: null,
};

// ============================================================================
// LAS 48 SELECCIONES DEL MUNDIAL 2026
// ============================================================================
const WC2026_TEAMS = [
  'Canada', 'Mexico', 'United States',
  'Australia', 'Iraq', 'Iran', 'Japan', 'Jordan', 'South Korea', 'Qatar', 'Saudi Arabia', 'Uzbekistan',
  'Algeria', 'Cabo Verde', 'DR Congo', 'Ivory Coast', "Côte d'Ivoire", 'Egypt', 'Ghana', 'Morocco', 'Senegal', 'South Africa', 'Tunisia',
  'Curaçao', 'Haiti', 'Panama',
  'Argentina', 'Brazil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
  'New Zealand',
  'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Croatia', 'Czech Republic', 'Czechia',
  'England', 'France', 'Germany', 'Netherlands', 'Norway', 'Portugal',
  'Scotland', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Türkiye',
];

// Mapeo de nombres alternativos
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

/**
 * FIFA-inspired prior strengths for initialization.
 * Tier 1 = elite, Tier 5 = weakest at WC.
 * These priors help the optimizer start closer to reality.
 */
const TEAM_PRIORS = {
  // Tier 1 — Elite (attack ~0.35, defense ~-0.25)
  'Argentina': { attack: 0.40, defense: -0.30 },
  'France': { attack: 0.38, defense: -0.28 },
  'Brazil': { attack: 0.36, defense: -0.22 },
  'England': { attack: 0.34, defense: -0.24 },
  'Spain': { attack: 0.35, defense: -0.26 },
  'Germany': { attack: 0.32, defense: -0.20 },
  'Portugal': { attack: 0.33, defense: -0.22 },
  'Netherlands': { attack: 0.30, defense: -0.20 },
  // Tier 2 — Strong
  'Belgium': { attack: 0.26, defense: -0.18 },
  'Croatia': { attack: 0.25, defense: -0.18 },
  'Uruguay': { attack: 0.24, defense: -0.22 },
  'Colombia': { attack: 0.22, defense: -0.15 },
  'Morocco': { attack: 0.20, defense: -0.20 },
  'Japan': { attack: 0.20, defense: -0.15 },
  'South Korea': { attack: 0.18, defense: -0.12 },
  'United States': { attack: 0.18, defense: -0.12 },
  'Mexico': { attack: 0.17, defense: -0.10 },
  'Senegal': { attack: 0.18, defense: -0.14 },
  'Switzerland': { attack: 0.16, defense: -0.16 },
  // Tier 3 — Mid
  'Ecuador': { attack: 0.14, defense: -0.08 },
  'Denmark': { attack: 0.15, defense: -0.12 },
  'Austria': { attack: 0.14, defense: -0.10 },
  'Turkey': { attack: 0.15, defense: -0.08 },
  'Iran': { attack: 0.10, defense: -0.12 },
  'Australia': { attack: 0.10, defense: -0.06 },
  'Tunisia': { attack: 0.10, defense: -0.08 },
  'Egypt': { attack: 0.12, defense: -0.08 },
  'Ivory Coast': { attack: 0.14, defense: -0.06 },
  'Ghana': { attack: 0.10, defense: -0.04 },
  'Norway': { attack: 0.14, defense: -0.08 },
  'Sweden': { attack: 0.12, defense: -0.10 },
  'Canada': { attack: 0.10, defense: -0.06 },
  'Paraguay': { attack: 0.10, defense: -0.10 },
  'Scotland': { attack: 0.10, defense: -0.08 },
  'Czech Republic': { attack: 0.10, defense: -0.08 },
  'DR Congo': { attack: 0.08, defense: -0.04 },
  'South Africa': { attack: 0.08, defense: -0.04 },
  'Algeria': { attack: 0.08, defense: -0.04 },
  // Tier 4 — Lower
  'Panama': { attack: 0.04, defense: 0.02 },
  'Qatar': { attack: 0.04, defense: 0.02 },
  'Saudi Arabia': { attack: 0.06, defense: 0.00 },
  'Iraq': { attack: 0.06, defense: 0.00 },
  'Jordan': { attack: 0.04, defense: 0.02 },
  'Uzbekistan': { attack: 0.06, defense: 0.00 },
  'Bosnia and Herzegovina': { attack: 0.08, defense: -0.04 },
  'New Zealand': { attack: 0.02, defense: 0.04 },
  // Tier 5 — Weakest at WC
  'Cabo Verde': { attack: 0.00, defense: 0.06 },
  'Curaçao': { attack: -0.02, defense: 0.08 },
  'Haiti': { attack: -0.04, defense: 0.10 },
};

function normalizeTeamName(name) {
  return TEAM_NAME_MAP[name] || name;
}

// ============================================================================
// PARSEO DE CSV
// ============================================================================
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
      homeScore,
      awayScore,
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
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

// ============================================================================
// DIXON-COLES: FUNCIONES MATEMÁTICAS
// ============================================================================
function logFactorial(n) {
  if (n <= 1) return 0;
  if (n <= 20) {
    let result = 0;
    for (let i = 2; i <= n; i++) result += Math.log(i);
    return result;
  }
  return n * Math.log(n) - n + 0.5 * Math.log(2 * Math.PI * n);
}

function logPoisson(k, lambda) {
  if (lambda <= 0) lambda = 1e-10;
  return k * Math.log(lambda) - lambda - logFactorial(k);
}

function dixonColesAdjustment(homeGoals, awayGoals, lambdaHome, lambdaAway, rho) {
  if (homeGoals === 0 && awayGoals === 0) {
    return 1 - lambdaHome * lambdaAway * rho;
  } else if (homeGoals === 0 && awayGoals === 1) {
    return 1 + lambdaHome * rho;
  } else if (homeGoals === 1 && awayGoals === 0) {
    return 1 + lambdaAway * rho;
  } else if (homeGoals === 1 && awayGoals === 1) {
    return 1 - rho;
  }
  return 1.0;
}

function matchLogLikelihood(homeGoals, awayGoals, lambdaHome, lambdaAway, rho) {
  const logPHome = logPoisson(homeGoals, lambdaHome);
  const logPAway = logPoisson(awayGoals, lambdaAway);
  const tau = dixonColesAdjustment(homeGoals, awayGoals, lambdaHome, lambdaAway, rho);

  if (tau <= 0) return -1000;

  return logPHome + logPAway + Math.log(tau);
}

// ============================================================================
// DIXON-COLES: MODELO CORREGIDO
// ============================================================================
class DixonColesModel {
  constructor(config = {}) {
    this.config = { ...MODEL_CONFIG, ...config };
    this.config.REFERENCE_DATE = this.config.REFERENCE_DATE || new Date().toISOString().split('T')[0];

    this.teams = {};
    this.homeAdvantage = 0.27;
    this.rho = -0.04;
    this.teamList = [];
    this.trainingMatches = [];
    this.allMatches = [];
  }

  loadFromJSON(json) {
    if (json.parameters) {
      this.homeAdvantage = Number.isFinite(json.parameters.home_advantage) ? json.parameters.home_advantage : this.homeAdvantage;
      this.rho = Number.isFinite(json.parameters.rho) ? json.parameters.rho : this.rho;
    }

    const teamData = json.all_teams || json.wc2026_teams || {};
    this.teams = {};
    for (const [team, ratings] of Object.entries(teamData)) {
      this.teams[team] = {
        attack: Number.isFinite(ratings.attack) ? ratings.attack : 0,
        defense: Number.isFinite(ratings.defense) ? ratings.defense : 0,
      };
    }
    this.teamList = Object.keys(this.teams).sort();
  }

  prepareData(matches) {
    this.allMatches = matches;

    const cutoff = this.config.CUTOFF_DATE;
    const filtered = matches.filter(m => m.date >= cutoff);

    const teamSet = new Set();
    filtered.forEach(m => {
      teamSet.add(m.homeTeam);
      teamSet.add(m.awayTeam);
    });

    this.teamList = Array.from(teamSet).sort();

    // Initialize with FIFA-inspired priors instead of zeros
    this.teams = {};
    this.teamList.forEach(t => {
      const prior = TEAM_PRIORS[t];
      if (prior) {
        this.teams[t] = { attack: prior.attack, defense: prior.defense };
      } else {
        this.teams[t] = { attack: 0.0, defense: 0.0 };
      }
    });

    const refDate = new Date(this.config.REFERENCE_DATE);

    this.trainingMatches = filtered.map(m => {
      const matchDate = new Date(m.date);
      const daysDiff = (refDate - matchDate) / (1000 * 60 * 60 * 24);
      const timeWeight = Math.exp(-this.config.XI * Math.max(daysDiff, 0));

      const tournamentKey = m.tournament;
      let tournWeight = this.config.TOURNAMENT_WEIGHTS[tournamentKey] || 1.0;

      // Detect friendlies
      if (!this.config.TOURNAMENT_WEIGHTS[tournamentKey] &&
          (tournamentKey.includes('Friendly') || tournamentKey.includes('friendly'))) {
        tournWeight = 0.6;
      }

      return {
        ...m,
        weight: timeWeight * tournWeight,
      };
    });

    return {
      totalMatches: this.trainingMatches.length,
      totalTeams: this.teamList.length,
    };
  }

  getLambda(attackTeam, defenseTeam, isHome) {
    const attack = this.teams[attackTeam]?.attack || 0;
    const defense = this.teams[defenseTeam]?.defense || 0;
    const ha = isHome ? this.homeAdvantage : 0;

    // lambda = exp(attack_i - defense_j + home_advantage)
    // Higher attack = more goals scored
    // Lower (more negative) defense = fewer goals conceded
    // So attack_i - defense_j: if defTeam has negative defense (good),
    // subtracting it makes lambda smaller = fewer goals for attacker
    return Math.exp(attack - defense + ha);
  }

  totalLogLikelihood() {
    let totalLL = 0;

    for (const m of this.trainingMatches) {
      if (!this.teams[m.homeTeam] || !this.teams[m.awayTeam]) continue;

      const lambdaHome = this.getLambda(m.homeTeam, m.awayTeam, !m.neutral);
      const lambdaAway = this.getLambda(m.awayTeam, m.homeTeam, false);

      const ll = matchLogLikelihood(m.homeScore, m.awayScore, lambdaHome, lambdaAway, this.rho);
      totalLL += m.weight * ll;
    }

    // L2 regularization towards priors (not towards zero)
    let regPenalty = 0;
    for (const t of this.teamList) {
      const prior = TEAM_PRIORS[t] || { attack: 0, defense: 0 };
      regPenalty += (this.teams[t].attack - prior.attack * 0.3) ** 2;
      regPenalty += (this.teams[t].defense - prior.defense * 0.3) ** 2;
    }
    totalLL -= this.config.LAMBDA_REG * regPenalty;

    return totalLL;
  }

  /**
   * Gradient descent optimization step with proper normalization constraints
   */
  optimizationStep() {
    const eps = 1e-5;
    const lr = this.config.LEARNING_RATE;
    const baseLL = this.totalLogLikelihood();

    // --- Gradients for each team's attack/defense ---
    for (const teamName of this.teamList) {
      // Attack gradient
      this.teams[teamName].attack += eps;
      const llAttack = this.totalLogLikelihood();
      this.teams[teamName].attack -= eps;
      const gradAttack = (llAttack - baseLL) / eps;

      // Defense gradient
      this.teams[teamName].defense += eps;
      const llDefense = this.totalLogLikelihood();
      this.teams[teamName].defense -= eps;
      const gradDefense = (llDefense - baseLL) / eps;

      // Gradient clipping to prevent explosions
      const maxGrad = 5.0;
      const clippedGradAttack = Math.max(-maxGrad, Math.min(maxGrad, gradAttack));
      const clippedGradDefense = Math.max(-maxGrad, Math.min(maxGrad, gradDefense));

      this.teams[teamName].attack += lr * clippedGradAttack;
      this.teams[teamName].defense += lr * clippedGradDefense;
    }

    // --- Home advantage gradient ---
    this.homeAdvantage += eps;
    const llHA = this.totalLogLikelihood();
    this.homeAdvantage -= eps;
    const gradHA = (llHA - baseLL) / eps;
    this.homeAdvantage += lr * gradHA;
    // Clamp home advantage to reasonable range
    this.homeAdvantage = Math.max(0.05, Math.min(0.6, this.homeAdvantage));

    // --- Rho gradient (slower learning) ---
    this.rho += eps;
    const llRho = this.totalLogLikelihood();
    this.rho -= eps;
    const gradRho = (llRho - baseLL) / eps;
    this.rho += lr * 0.1 * gradRho;
    this.rho = Math.max(-0.3, Math.min(0.3, this.rho));

    // --- CRITICAL FIX: Normalize BOTH attack AND defense (sum = 0) ---
    const avgAttack = this.teamList.reduce((s, t) => s + this.teams[t].attack, 0) / this.teamList.length;
    const avgDefense = this.teamList.reduce((s, t) => s + this.teams[t].defense, 0) / this.teamList.length;
    for (const t of this.teamList) {
      this.teams[t].attack -= avgAttack;
      this.teams[t].defense -= avgDefense;
    }

    return this.totalLogLikelihood();
  }

  train(onProgress = null) {
    const maxIter = this.config.MAX_ITERATIONS;
    let bestLL = -Infinity;
    let noImproveCount = 0;

    for (let i = 0; i < maxIter; i++) {
      const ll = this.optimizationStep();

      if (onProgress && i % 20 === 0) {
        onProgress(i, maxIter, ll);
      }

      // Early stopping with tolerance
      if (ll > bestLL + 0.005) {
        bestLL = ll;
        noImproveCount = 0;
      } else {
        noImproveCount++;
        if (noImproveCount > 200) {
          if (onProgress) onProgress(i, maxIter, ll);
          break;
        }
      }
    }

    return bestLL;
  }

  getRecentMatches(teamName, n = 15) {
    const normalized = normalizeTeamName(teamName);
    return this.allMatches
      .filter(m => m.homeTeam === normalized || m.awayTeam === normalized)
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, n)
      .map(m => {
        const isHome = m.homeTeam === normalized;
        return {
          date: m.date,
          opponent: isHome ? m.awayTeam : m.homeTeam,
          goalsFor: isHome ? m.homeScore : m.awayScore,
          goalsAgainst: isHome ? m.awayScore : m.homeScore,
          result: isHome
            ? (m.homeScore > m.awayScore ? 'W' : m.homeScore < m.awayScore ? 'L' : 'D')
            : (m.awayScore > m.homeScore ? 'W' : m.awayScore < m.homeScore ? 'L' : 'D'),
          tournament: m.tournament,
          venue: isHome ? 'Home' : 'Away',
        };
      });
  }

  predictMatch(teamA, teamB, nSims = 100000, neutral = true) {
    const normA = normalizeTeamName(teamA);
    const normB = normalizeTeamName(teamB);

    if (!this.teams[normA]) throw new Error(`Equipo no encontrado: ${teamA}`);
    if (!this.teams[normB]) throw new Error(`Equipo no encontrado: ${teamB}`);

    const lambdaA = this.getLambda(normA, normB, !neutral);
    const lambdaB = this.getLambda(normB, normA, false);

    // Monte Carlo simulation
    const results = { A: 0, Draw: 0, B: 0 };
    const scores = {};
    const goalsA = [];
    const goalsB = [];

    for (let i = 0; i < nSims; i++) {
      const gA = poissonRandom(lambdaA);
      const gB = poissonRandom(lambdaB);

      goalsA.push(gA);
      goalsB.push(gB);

      if (gA > gB) results.A++;
      else if (gA < gB) results.B++;
      else results.Draw++;

      const scoreKey = `${gA}-${gB}`;
      scores[scoreKey] = (scores[scoreKey] || 0) + 1;
    }

    const topScores = Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([score, count]) => ({
        score,
        prob: (count / nSims * 100),
      }));

    const avgGoalsA = goalsA.reduce((a, b) => a + b, 0) / nSims;
    const avgGoalsB = goalsB.reduce((a, b) => a + b, 0) / nSims;

    return {
      teamA: normA,
      teamB: normB,
      lambdaA: Math.round(lambdaA * 10000) / 10000,
      lambdaB: Math.round(lambdaB * 10000) / 10000,
      probA: Math.round(results.A / nSims * 10000) / 100,
      probDraw: Math.round(results.Draw / nSims * 10000) / 100,
      probB: Math.round(results.B / nSims * 10000) / 100,
      topScores,
      avgGoalsA: Math.round(avgGoalsA * 100) / 100,
      avgGoalsB: Math.round(avgGoalsB * 100) / 100,
      nSimulations: nSims,
      ratingA: this.teams[normA],
      ratingB: this.teams[normB],
      homeAdvantage: Math.round(this.homeAdvantage * 10000) / 10000,
      rho: Math.round(this.rho * 10000) / 10000,
    };
  }

  exportModel() {
    const wcTeams = {};
    const normalizedWC = new Set(WC2026_TEAMS.map(normalizeTeamName));

    for (const [name, ratings] of Object.entries(this.teams)) {
      if (normalizedWC.has(name)) {
        wcTeams[name] = {
          attack: Math.round(ratings.attack * 10000) / 10000,
          defense: Math.round(ratings.defense * 10000) / 10000,
        };
      }
    }

    return {
      version: '4.0',
      trained_at: new Date().toISOString(),
      config: {
        cutoff_date: this.config.CUTOFF_DATE,
        xi_decay: this.config.XI,
        learning_rate: this.config.LEARNING_RATE,
        iterations: this.config.MAX_ITERATIONS,
        regularization: this.config.LAMBDA_REG,
      },
      matches_used: this.trainingMatches.length,
      total_teams: this.teamList.length,
      parameters: {
        home_advantage: Math.round(this.homeAdvantage * 10000) / 10000,
        rho: Math.round(this.rho * 10000) / 10000,
      },
      wc2026_teams: wcTeams,
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
// ELO + ML MODEL (Machine Learning / Logistic Regression)
// ============================================================================

/**
 * ELO-based prediction model with logistic regression.
 * Computes Elo ratings from historical match data, then uses
 * features (Elo diff, form, goal stats) in a logistic model.
 */
class EloMLModel {
  constructor() {
    this.elo = {};           // { teamName: eloRating }
    this.form = {};          // { teamName: { wins, draws, losses, gf, ga, last10 } }
    this.K_BASE = 40;        // Base K-factor
    this.K_TOURNAMENT = {
      'FIFA World Cup': 60,
      'FIFA World Cup qualification': 45,
      'Copa América': 50,
      'UEFA Euro': 50,
      'UEFA Euro qualification': 40,
      'AFC Asian Cup': 45,
      'African Cup of Nations': 45,
      'CONCACAF Gold Cup': 40,
      'UEFA Nations League': 40,
    };
    this.INITIAL_ELO = 1500;
    this.allMatches = [];
    this.trainingData = [];   // { eloA, eloB, formA, formB, ... , result }
    this.weights = null;      // Logistic regression weights
    this.teamList = [];
  }

  /**
   * Train the Elo system and build ML features
   */
  train(matches, onProgress = null) {
    this.allMatches = matches;

    // Sort matches chronologically
    const sorted = [...matches].sort((a, b) => a.date.localeCompare(b.date));

    // Initialize all teams
    const teamSet = new Set();
    sorted.forEach(m => { teamSet.add(m.homeTeam); teamSet.add(m.awayTeam); });
    this.teamList = Array.from(teamSet).sort();

    // Initialize Elo with priors
    this.elo = {};
    this.teamList.forEach(t => {
      const prior = TEAM_PRIORS[t];
      if (prior) {
        // Map prior attack/defense to Elo: stronger teams start higher
        const strength = (prior.attack - prior.defense) * 400;
        this.elo[t] = this.INITIAL_ELO + strength;
      } else {
        this.elo[t] = this.INITIAL_ELO;
      }
    });

    // Track recent form per team
    const recentResults = {}; // { team: [last N results as 1/0.5/0] }
    this.teamList.forEach(t => { recentResults[t] = []; });

    // Track goal stats
    const goalStats = {}; // { team: { gf: [], ga: [] } }
    this.teamList.forEach(t => { goalStats[t] = { gf: [], ga: [] }; });

    // Process matches and build training data
    this.trainingData = [];
    const cutoffDate = '2018-01-01';

    for (let idx = 0; idx < sorted.length; idx++) {
      const m = sorted[idx];

      if (onProgress && idx % 5000 === 0) {
        onProgress(idx, sorted.length);
      }

      const home = m.homeTeam;
      const away = m.awayTeam;

      if (!this.elo[home]) this.elo[home] = this.INITIAL_ELO;
      if (!this.elo[away]) this.elo[away] = this.INITIAL_ELO;

      const eloHome = this.elo[home];
      const eloAway = this.elo[away];

      // Expected scores
      const expHome = 1 / (1 + Math.pow(10, (eloAway - eloHome) / 400));
      const expAway = 1 - expHome;

      // Actual result
      let actualHome, actualAway;
      if (m.homeScore > m.awayScore) {
        actualHome = 1; actualAway = 0;
      } else if (m.homeScore < m.awayScore) {
        actualHome = 0; actualAway = 1;
      } else {
        actualHome = 0.5; actualAway = 0.5;
      }

      // Goal difference bonus (capped at 3)
      const goalDiff = Math.min(Math.abs(m.homeScore - m.awayScore), 3);
      const gdMultiplier = 1 + goalDiff * 0.1;

      // K-factor based on tournament
      const K = (this.K_TOURNAMENT[m.tournament] || this.K_BASE) * gdMultiplier;

      // If after cutoff, save as training data for logistic regression
      if (m.date >= cutoffDate) {
        const formHome = this._getFormScore(recentResults[home]);
        const formAway = this._getFormScore(recentResults[away]);
        const gfHome = this._getAvg(goalStats[home]?.gf || []);
        const gaHome = this._getAvg(goalStats[home]?.ga || []);
        const gfAway = this._getAvg(goalStats[away]?.gf || []);
        const gaAway = this._getAvg(goalStats[away]?.ga || []);

        this.trainingData.push({
          eloDiff: eloHome - eloAway,
          formDiff: formHome - formAway,
          gfDiff: gfHome - gfAway,
          gaDiff: gaHome - gaAway,
          neutral: m.neutral ? 1 : 0,
          result: actualHome, // 1 = home win, 0.5 = draw, 0 = away win
        });
      }

      // Update Elo ratings
      this.elo[home] += K * (actualHome - expHome);
      this.elo[away] += K * (actualAway - expAway);

      // Update recent results (keep last 10)
      if (!recentResults[home]) recentResults[home] = [];
      if (!recentResults[away]) recentResults[away] = [];
      recentResults[home].push(actualHome);
      recentResults[away].push(actualAway);
      if (recentResults[home].length > 10) recentResults[home].shift();
      if (recentResults[away].length > 10) recentResults[away].shift();

      // Update goal stats (keep last 15)
      if (!goalStats[home]) goalStats[home] = { gf: [], ga: [] };
      if (!goalStats[away]) goalStats[away] = { gf: [], ga: [] };
      goalStats[home].gf.push(m.homeScore);
      goalStats[home].ga.push(m.awayScore);
      goalStats[away].gf.push(m.awayScore);
      goalStats[away].ga.push(m.homeScore);
      if (goalStats[home].gf.length > 15) { goalStats[home].gf.shift(); goalStats[home].ga.shift(); }
      if (goalStats[away].gf.length > 15) { goalStats[away].gf.shift(); goalStats[away].ga.shift(); }
    }

    // Save final form and goal stats
    this.form = {};
    this.teamList.forEach(t => {
      this.form[t] = {
        formScore: this._getFormScore(recentResults[t] || []),
        avgGF: this._getAvg(goalStats[t]?.gf || []),
        avgGA: this._getAvg(goalStats[t]?.ga || []),
      };
    });

    // Train logistic regression on the training data
    this._trainLogisticRegression();

    return {
      totalTeams: this.teamList.length,
      trainingSize: this.trainingData.length,
    };
  }

  _getFormScore(results) {
    if (results.length === 0) return 0.5;
    // Weighted average with more weight on recent
    let total = 0, wTotal = 0;
    for (let i = 0; i < results.length; i++) {
      const w = 1 + i * 0.2; // More recent = higher weight
      total += results[i] * w;
      wTotal += w;
    }
    return total / wTotal;
  }

  _getAvg(arr) {
    if (arr.length === 0) return 1.0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  /**
   * Train a simple logistic regression using gradient descent.
   * Features: [eloDiff, formDiff, gfDiff, gaDiff, neutral, bias]
   * Target: P(home win) vs P(away win), with draws handled via ordinal thresholds.
   */
  _trainLogisticRegression() {
    // We'll train TWO logistic regressions:
    // Model A: P(home win) vs P(not home win) = P(draw or away win)
    // Model B: P(away win) vs P(not away win) = P(home win or draw)
    // Then: P(home) = sigmoid_A, P(away) = sigmoid_B, P(draw) = 1 - P(home) - P(away)

    const nFeatures = 5; // eloDiff, formDiff, gfDiff, gaDiff, neutral
    this.weightsWin = new Float64Array(nFeatures + 1); // +1 for bias
    this.weightsLoss = new Float64Array(nFeatures + 1);

    const lr = 0.0001;
    const iterations = 500;
    const data = this.trainingData;

    if (data.length === 0) return;

    // Normalize features
    const means = new Float64Array(nFeatures);
    const stds = new Float64Array(nFeatures);

    for (let f = 0; f < nFeatures; f++) {
      const vals = data.map(d => this._getFeature(d, f));
      means[f] = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - means[f]) ** 2, 0) / vals.length;
      stds[f] = Math.sqrt(variance) || 1;
    }

    this.featureMeans = means;
    this.featureStds = stds;

    // Train win model (result == 1)
    for (let iter = 0; iter < iterations; iter++) {
      const gradW = new Float64Array(nFeatures + 1);

      for (const d of data) {
        const features = this._normalizeFeatures(d);
        const z = this._dotProduct(features, this.weightsWin);
        const pred = this._sigmoid(z);
        const target = d.result === 1 ? 1 : 0;
        const error = pred - target;

        for (let j = 0; j < features.length; j++) {
          gradW[j] += error * features[j];
        }
      }

      for (let j = 0; j < this.weightsWin.length; j++) {
        this.weightsWin[j] -= lr * (gradW[j] / data.length + 0.001 * this.weightsWin[j]);
      }
    }

    // Train loss model (result == 0 means away win)
    for (let iter = 0; iter < iterations; iter++) {
      const gradL = new Float64Array(nFeatures + 1);

      for (const d of data) {
        const features = this._normalizeFeatures(d);
        const z = this._dotProduct(features, this.weightsLoss);
        const pred = this._sigmoid(z);
        const target = d.result === 0 ? 1 : 0;
        const error = pred - target;

        for (let j = 0; j < features.length; j++) {
          gradL[j] += error * features[j];
        }
      }

      for (let j = 0; j < this.weightsLoss.length; j++) {
        this.weightsLoss[j] -= lr * (gradL[j] / data.length + 0.001 * this.weightsLoss[j]);
      }
    }
  }

  _getFeature(d, index) {
    switch (index) {
      case 0: return d.eloDiff;
      case 1: return d.formDiff;
      case 2: return d.gfDiff;
      case 3: return d.gaDiff;
      case 4: return d.neutral;
      default: return 0;
    }
  }

  _normalizeFeatures(d) {
    return [
      (d.eloDiff - this.featureMeans[0]) / this.featureStds[0],
      (d.formDiff - this.featureMeans[1]) / this.featureStds[1],
      (d.gfDiff - this.featureMeans[2]) / this.featureStds[2],
      (d.gaDiff - this.featureMeans[3]) / this.featureStds[3],
      (d.neutral - this.featureMeans[4]) / this.featureStds[4],
      1.0, // bias
    ];
  }

  _dotProduct(features, weights) {
    let sum = 0;
    for (let i = 0; i < features.length; i++) {
      sum += features[i] * weights[i];
    }
    return sum;
  }

  exportState() {
    return {
      elo: this.elo,
      form: this.form,
      weightsWin: this.weightsWin ? Array.from(this.weightsWin) : null,
      weightsLoss: this.weightsLoss ? Array.from(this.weightsLoss) : null,
      featureMeans: this.featureMeans ? Array.from(this.featureMeans) : null,
      featureStds: this.featureStds ? Array.from(this.featureStds) : null,
    };
  }

  loadFromState(state) {
    if (!state) return;
    this.elo = { ...state.elo };
    this.form = { ...state.form };
    if (state.weightsWin) this.weightsWin = new Float64Array(state.weightsWin);
    if (state.weightsLoss) this.weightsLoss = new Float64Array(state.weightsLoss);
    if (state.featureMeans) this.featureMeans = new Float64Array(state.featureMeans);
    if (state.featureStds) this.featureStds = new Float64Array(state.featureStds);
  }

  _sigmoid(z) {
    if (z > 20) return 1 - 1e-9;
    if (z < -20) return 1e-9;
    return 1 / (1 + Math.exp(-z));
  }

  /**
   * Predict match outcome
   */
  predictMatch(teamA, teamB, neutral = true) {
    const normA = normalizeTeamName(teamA);
    const normB = normalizeTeamName(teamB);

    const eloA = this.elo[normA] || this.INITIAL_ELO;
    const eloB = this.elo[normB] || this.INITIAL_ELO;
    const formA = this.form[normA] || { formScore: 0.5, avgGF: 1, avgGA: 1 };
    const formB = this.form[normB] || { formScore: 0.5, avgGF: 1, avgGA: 1 };

    const dataPoint = {
      eloDiff: eloA - eloB,
      formDiff: formA.formScore - formB.formScore,
      gfDiff: formA.avgGF - formB.avgGF,
      gaDiff: formA.avgGA - formB.avgGA,
      neutral: neutral ? 1 : 0,
    };

    // Use logistic regression if trained
    let probA, probB, probDraw;

    if (this.weightsWin && this.featureMeans) {
      const features = this._normalizeFeatures(dataPoint);
      const rawProbWin = this._sigmoid(this._dotProduct(features, this.weightsWin));
      const rawProbLoss = this._sigmoid(this._dotProduct(features, this.weightsLoss));

      // Normalize probabilities
      // rawProbWin is P(teamA wins), rawProbLoss is P(teamB wins)
      const totalRaw = rawProbWin + rawProbLoss;

      if (totalRaw >= 1) {
        // Scale down proportionally
        const scale = 0.95 / totalRaw;
        probA = rawProbWin * scale;
        probB = rawProbLoss * scale;
        probDraw = 1 - probA - probB;
      } else {
        probA = rawProbWin;
        probB = rawProbLoss;
        probDraw = 1 - probA - probB;
      }

      // Ensure draw is reasonable (at least 10% for evenly matched, less for mismatches)
      const eloDiffAbs = Math.abs(dataPoint.eloDiff);
      const minDraw = Math.max(0.08, 0.30 - eloDiffAbs / 2000);
      const maxDraw = 0.40;

      if (probDraw < minDraw) {
        const deficit = minDraw - probDraw;
        probDraw = minDraw;
        // Reduce proportionally from A and B
        const ratio = probA / (probA + probB);
        probA -= deficit * ratio;
        probB -= deficit * (1 - ratio);
      }
      if (probDraw > maxDraw) {
        const excess = probDraw - maxDraw;
        probDraw = maxDraw;
        const ratio = probA / (probA + probB);
        probA += excess * ratio;
        probB += excess * (1 - ratio);
      }
    } else {
      // Fallback to pure Elo
      const expA = 1 / (1 + Math.pow(10, (eloB - eloA) / 400));
      probA = expA * 0.75;
      probB = (1 - expA) * 0.75;
      probDraw = 1 - probA - probB;
    }

    // Ensure all probabilities are valid
    probA = Math.max(0.01, probA);
    probB = Math.max(0.01, probB);
    probDraw = Math.max(0.05, probDraw);
    const total = probA + probB + probDraw;
    probA /= total;
    probB /= total;
    probDraw /= total;

    return {
      teamA: normA,
      teamB: normB,
      probA: Math.round(probA * 10000) / 100,
      probDraw: Math.round(probDraw * 10000) / 100,
      probB: Math.round(probB * 10000) / 100,
      eloA: Math.round(eloA),
      eloB: Math.round(eloB),
      formA: Math.round(formA.formScore * 100) / 100,
      formB: Math.round(formB.formScore * 100) / 100,
      avgGfA: Math.round(formA.avgGF * 100) / 100,
      avgGfB: Math.round(formB.avgGF * 100) / 100,
      avgGaA: Math.round(formA.avgGA * 100) / 100,
      avgGaB: Math.round(formB.avgGA * 100) / 100,
    };
  }

  /**
   * Get top N teams by Elo
   */
  getTopTeams(n = 20) {
    return Object.entries(this.elo)
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([name, rating]) => ({ name, elo: Math.round(rating) }));
  }
}


// ============================================================================
// UTILIDADES
// ============================================================================
function poissonRandom(lambda) {
  if (lambda <= 0) return 0;
  if (lambda > 30) {
    const normal = Math.sqrt(lambda) * boxMullerRandom() + lambda;
    return Math.max(0, Math.round(normal));
  }

  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;

  do {
    k++;
    p *= Math.random();
  } while (p > L);

  return k - 1;
}

function boxMullerRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ============================================================================
// EXPORT
// ============================================================================
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DixonColesModel, EloMLModel, parseCSV, WC2026_TEAMS, normalizeTeamName, poissonRandom };
}
