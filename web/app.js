const state = {
  genres: [],
  activeGenres: new Set(["Comedy", "Drama"]),
  mode: "balanced",
  hasRecommendations: localStorage.getItem("magicMovieCompletedQuiz") === "true",
  lastRecommendations: [],
};

const quizState = {
  index: 0,
  answers: {},
  isTurning: false,
};

const watchlist = new Map(JSON.parse(localStorage.getItem("magicMovieWatchlist") || "[]"));

const quizQuestions = [
  {
    id: "mood",
    color: "#ff4718",
    question: "Q: What mood are you in right now?",
    options: ["Comforting", "Funny", "Emotional", "Intense", "Mysterious", "Thought-provoking", "Romantic", "Dark"],
  },
  {
    id: "effort",
    color: "#5aa7ff",
    question: "Q: How much mental effort do you want?",
    options: ["Easy watch", "Some thinking", "Challenging"],
  },
  {
    id: "risk",
    color: "#7c63ff",
    question: "Q: Do you want a safe pick or an adventurous pick?",
    options: ["Safe pick", "Balanced", "Adventurous"],
  },
  {
    id: "popularity",
    color: "#d35d91",
    question: "Q: Mainstream favorite or hidden gem?",
    options: ["Mainstream", "A mix", "Hidden gem"],
  },
  {
    id: "intensity",
    color: "#0f9f8f",
    question: "Q: How emotionally intense should it be?",
    options: ["Light", "Medium", "Emotionally heavy"],
  },
  {
    id: "ending",
    color: "#b65f27",
    question: "Q: What type of ending do you want?",
    options: ["Happy", "Bittersweet", "Ambiguous", "Dark", "No preference"],
  },
  {
    id: "time",
    color: "#2f855a",
    question: "Q: How much time do you have?",
    options: ["Under 90 minutes", "90-120 minutes", "Over 2 hours", "No preference"],
  },
  {
    id: "genres",
    color: "#ef3f6b",
    question: "Q: Pick genres you want today.",
    multiple: true,
    options: ["Comedy", "Romance", "Drama", "Sci-Fi", "Thriller", "Horror", "Action", "Animation", "Documentary", "Mystery"],
  },
];

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function init() {
  initLanding();
  initQuiz();
  const boot = await fetchJson("/api/bootstrap");
  state.genres = boot.genres;
  $("#datasetSummary").innerHTML = `
    <strong>${boot.ratingsCount.toLocaleString()} ratings</strong>
    <span>${boot.moviesCount} movies, ${boot.users.length} users</span>
  `;

  $("#user").innerHTML = boot.users
    .slice(0, 40)
    .map((user) => `<option value="${user}">User ${user}</option>`)
    .join("");

  $("#genres").innerHTML = state.genres
    .map((genre) => `<button class="chip ${state.activeGenres.has(genre) ? "active" : ""}" data-genre="${genre}">${genre}</button>`)
    .join("");

  bindEvents();
  await Promise.all([loadRecommendations(), loadModels(), loadDiagnostics()]);
}

function initLanding() {
  const startButton = $("#startQuiz");
  startButton.addEventListener("click", () => {
    document.body.classList.remove("landing-visible");
    document.body.classList.add("quiz-visible");
    $("#quizShell").setAttribute("aria-hidden", "false");
    renderQuizCard();
  });

  $("#quizLogoHome").addEventListener("click", showLanding);
  $("#resultsLogoHome").addEventListener("click", showLanding);
  $("#backToQuiz").addEventListener("click", () => showQuiz({ preserveAnswers: true }));
  $("#openStatsGuide").addEventListener("click", openStatsGuide);
  $("#closeStatsGuide").addEventListener("click", closeStatsGuide);
  $("#themeToggle").addEventListener("click", toggleTheme);
  $("#watchlistToggle").addEventListener("click", openWatchlistPanel);
  $("#closeWatchlist").addEventListener("click", closeWatchlistPanel);
  $("#clearWatchlist").addEventListener("click", clearWatchlist);
  $("#returnToResults").addEventListener("click", showResultsPage);
  $("#statsGuideOverlay").addEventListener("click", (event) => {
    if (event.target.id === "statsGuideOverlay") closeStatsGuide();
  });
  $("#watchlistOverlay").addEventListener("click", (event) => {
    if (event.target.id === "watchlistOverlay") closeWatchlistPanel();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeStatsGuide();
    if (event.key === "Escape") closeWatchlistPanel();
  });

  $$("[data-draggable]").forEach((item) => makeDraggable(item, $("#dragBounds")));
  renderWatchlist();
}

function initQuiz() {
  $("#previousQuestion").addEventListener("click", () => {
    if (quizState.index === 0) return;
    turnQuizPage(-1);
  });

  $("#nextQuestion").addEventListener("click", async () => {
    if (!hasCurrentAnswer()) return;
    if (quizState.index === quizQuestions.length - 1) {
      await finishQuiz();
      return;
    }
    turnQuizPage(1);
  });

  $("#quizCard").addEventListener("click", (event) => {
    const option = event.target.closest("[data-option]");
    if (!option) return;
    selectQuizOption(option.dataset.option);
  });

  renderQuizCard();
}

function renderQuizCard() {
  const question = quizQuestions[quizState.index];
  const nextQuestion = quizQuestions[quizState.index + 1];
  const nextNextQuestion = quizQuestions[quizState.index + 2];
  const selected = normalizeAnswer(quizState.answers[question.id]);
  const card = $("#quizCard");
  const shadowOne = $(".card-shadow-one");
  const shadowTwo = $(".card-shadow-two");
  card.classList.remove("page-flip-next", "page-flip-prev", "page-enter-next", "page-enter-prev");
  card.style.setProperty("--card-color", question.color);
  shadowTwo.style.setProperty("--shadow-two-color", nextQuestion?.color || "#cddf65");
  shadowOne.style.setProperty("--shadow-one-color", nextNextQuestion?.color || nextQuestion?.color || "#f0c1a8");
  card.classList.toggle("is-multiple", Boolean(question.multiple));
  card.innerHTML = `
    <h2 class="question-title">${question.question}</h2>
    ${question.multiple ? `<p class="question-hint">you can pick multiple</p>` : ""}
    <div class="option-grid ${question.options.length % 3 === 0 ? "option-grid-three" : ""}">
      ${question.options
        .map((option) => `
          <button
            class="option-button ${selected.includes(option) ? "selected" : ""}"
            data-option="${option}"
            type="button"
          >${option}</button>
        `)
        .join("")}
    </div>
  `;
  $("#previousQuestion").disabled = quizState.index === 0;
  $("#nextQuestion").disabled = !hasCurrentAnswer();
  $("#nextQuestion").textContent = quizState.index === quizQuestions.length - 1 ? "show picks" : "next";
  $("#nextQuestion").hidden = !question.multiple;
  $("#returnToResults").hidden = !state.hasRecommendations;
  $("#quizProgress").textContent = `${quizState.index + 1} / ${quizQuestions.length}`;
}

async function turnQuizPage(direction) {
  const nextIndex = quizState.index + direction;
  if (quizState.isTurning || nextIndex < 0 || nextIndex >= quizQuestions.length) return;
  quizState.isTurning = true;
  const card = $("#quizCard");
  card.classList.add(direction > 0 ? "page-flip-next" : "page-flip-prev");
  await delay(230);
  quizState.index = nextIndex;
  renderQuizCard();
  card.classList.add(direction > 0 ? "page-enter-next" : "page-enter-prev");
  await delay(230);
  card.classList.remove("page-enter-next", "page-enter-prev");
  quizState.isTurning = false;
}

function selectQuizOption(option) {
  const question = quizQuestions[quizState.index];
  if (question.multiple) {
    const values = new Set(normalizeAnswer(quizState.answers[question.id]));
    if (values.has(option)) {
      values.delete(option);
    } else {
      values.add(option);
    }
    quizState.answers[question.id] = Array.from(values);
  } else {
    quizState.answers[question.id] = option;
    renderQuizCard();
    window.setTimeout(() => advanceQuiz(), 170);
    return;
  }
  renderQuizCard();
}

async function advanceQuiz() {
  if (!hasCurrentAnswer()) return;
  if (quizState.index === quizQuestions.length - 1) {
    await finishQuiz();
    return;
  }
  await turnQuizPage(1);
}

async function finishQuiz() {
  applyQuizPreferences();
  await Promise.all([loadRecommendations(), loadModels(), loadDiagnostics()]);
  state.hasRecommendations = true;
  localStorage.setItem("magicMovieCompletedQuiz", "true");
  showResultsPage();
}

function applyQuizPreferences() {
  const moodMap = {
    Comforting: "warm",
    Funny: "fun",
    Emotional: "thoughtful",
    Intense: "intense",
    Mysterious: "curious",
    "Thought-provoking": "thoughtful",
    Romantic: "warm",
    Dark: "intense",
  };
  const genreMap = {
    Comedy: "Comedy",
    Romance: "Romance",
    Drama: "Drama",
    "Sci-Fi": "Sci-Fi",
    Thriller: "Thriller",
    Horror: "Thriller",
    Action: "Action",
    Animation: "Adventure",
    Documentary: "Documentary",
    Mystery: "Thriller",
  };
  $("#mood").value = moodMap[quizState.answers.mood] || "thoughtful";
  $("#confidence").value = quizState.answers.risk === "Safe pick" ? "0.9" : quizState.answers.risk === "Adventurous" ? "0.15" : "0.5";
  $("#adventure").value = quizState.answers.risk === "Adventurous" ? "0.9" : quizState.answers.risk === "Safe pick" ? "0.1" : "0.45";
  $("#hiddenGem").value = quizState.answers.popularity === "Hidden gem" ? "0.95" : quizState.answers.popularity === "Mainstream" ? "0.1" : "0.55";
  state.mode = quizState.answers.risk === "Safe pick"
    ? "safe"
    : quizState.answers.risk === "Adventurous"
      ? "adventurous"
      : quizState.answers.popularity === "Hidden gem"
        ? "hidden_gem"
        : "balanced";

  const selectedGenres = normalizeAnswer(quizState.answers.genres)
    .map((genre) => genreMap[genre])
    .filter((genre) => state.genres.includes(genre));
  if (selectedGenres.length > 0) {
    state.activeGenres = new Set(selectedGenres);
  }
  $$("#genres .chip").forEach((button) => {
    button.classList.toggle("active", state.activeGenres.has(button.dataset.genre));
  });
  $$("#mode button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.mode);
  });
}

function hasCurrentAnswer() {
  const question = quizQuestions[quizState.index];
  return normalizeAnswer(quizState.answers[question.id]).length > 0;
}

function normalizeAnswer(answer) {
  if (!answer) return [];
  return Array.isArray(answer) ? answer : [answer];
}

function makeDraggable(item, bounds) {
  let drag = null;
  let touchedEdge = false;

  item.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    item.classList.remove("edge-bounce");
    drag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startLeft: item.offsetLeft,
      startTop: item.offsetTop,
      maxLeft: bounds.clientWidth - item.offsetWidth,
      maxTop: bounds.clientHeight - item.offsetHeight,
    };
    item.classList.add("dragging");
    item.setPointerCapture(event.pointerId);
  });

  item.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const rawLeft = drag.startLeft + event.clientX - drag.startX;
    const rawTop = drag.startTop + event.clientY - drag.startY;
    const boundedLeft = clamp(rawLeft, 0, drag.maxLeft);
    const boundedTop = clamp(rawTop, 0, drag.maxTop);
    const bounceLeft = boundedLeft + rubberBand(rawLeft - boundedLeft);
    const bounceTop = boundedTop + rubberBand(rawTop - boundedTop);
    touchedEdge = touchedEdge || rawLeft !== boundedLeft || rawTop !== boundedTop;
    item.style.left = `${bounceLeft}px`;
    item.style.top = `${bounceTop}px`;
    item.style.right = "auto";
    item.style.bottom = "auto";
  });

  item.addEventListener("pointerup", endDrag);
  item.addEventListener("pointercancel", endDrag);

  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const finalLeft = clamp(item.offsetLeft, 0, drag.maxLeft);
    const finalTop = clamp(item.offsetTop, 0, drag.maxTop);
    item.classList.remove("dragging");
    item.releasePointerCapture(event.pointerId);
    item.style.left = `${finalLeft}px`;
    item.style.top = `${finalTop}px`;
    if (touchedEdge) {
      item.classList.add("edge-bounce");
      window.setTimeout(() => item.classList.remove("edge-bounce"), 280);
    }
    drag = null;
    touchedEdge = false;
  }
}

function bindEvents() {
  $("#genres").addEventListener("click", async (event) => {
    if (!event.target.matches("button")) return;
    const genre = event.target.dataset.genre;
    if (state.activeGenres.has(genre)) {
      state.activeGenres.delete(genre);
    } else {
      state.activeGenres.add(genre);
    }
    event.target.classList.toggle("active");
    await loadRecommendations();
  });

  ["user", "mood", "confidence", "hiddenGem", "adventure"].forEach((id) => {
    $(`#${id}`).addEventListener("input", debounce(loadRecommendations, 180));
  });

  $("#mode").addEventListener("click", async (event) => {
    if (!event.target.matches("button")) return;
    state.mode = event.target.dataset.mode;
    $$("#mode button").forEach((button) => button.classList.toggle("active", button.dataset.mode === state.mode));
    await loadRecommendations();
  });

  $$(".tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".tabs button").forEach((item) => item.classList.remove("active"));
      $$(".tab").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $(`#${button.dataset.tab}`).classList.add("active");
    });
  });
}

async function loadRecommendations() {
  const params = new URLSearchParams({
    user: $("#user").value,
    genres: Array.from(state.activeGenres).join(","),
    mood: $("#mood").value,
    confidence: $("#confidence").value,
    hiddenGem: $("#hiddenGem").value,
    adventure: $("#adventure").value,
    effort: effortPreference(),
    mode: state.mode,
  });
  const data = await fetchJson(`/api/recommend?${params}`);
  renderRecommendations(data.recommendations);
  renderContributions(data.contributions, data.recommendations[0]);
}

function effortPreference() {
  const answer = quizState.answers.effort;
  if (answer === "Easy watch") return "0.15";
  if (answer === "Challenging") return "0.9";
  if (answer === "Some thinking") return "0.5";
  return "0.5";
}

function renderRecommendations(rows) {
  state.lastRecommendations = rows;
  renderMovieCards(rows);
  $("#recommendationRows").innerHTML = rows
    .map((row) => `
      <tr>
        <td>
          <div class="movie-title">${row.title} <span class="subtle">(${row.year})</span></div>
          <div class="subtle">${row.genres}</div>
        </td>
        <td class="score">${fmt(row.predicted_rating)}</td>
        <td>${fmt(row.naive_low_rating)}-${fmt(row.naive_high_rating)}</td>
        <td>${fmt(row.bootstrap_low_rating)}-${fmt(row.bootstrap_high_rating)}</td>
        <td class="score">${fmt(row.decision_score)}</td>
      </tr>
    `)
    .join("");
}

function renderMovieCards(rows) {
  const cardColors = ["#ff4718", "#8fa52d", "#4f8f86", "#4f82d8", "#d9476d", "#9b5d2e", "#7253c7", "#2f6f64"];
  $("#movieCardGrid").innerHTML = rows
    .map((row, index) => {
      const intervalWidth = row.bootstrap_high_rating - row.bootstrap_low_rating;
      const uncertainty = intervalWidth > 1 ? "High" : intervalWidth > 0.65 ? "Medium" : "Low";
      const movieId = String(row.movie_id);
      const isSaved = watchlist.has(movieId);
      const saveButton = `
        <button class="movie-save-star ${isSaved ? "saved" : ""}" type="button" aria-label="${isSaved ? "Remove from" : "Add to"} watchlist" data-save-movie="${movieId}">★</button>
      `;
      return `
        <article class="movie-flip-card" tabindex="0" role="button" data-movie-id="${movieId}" style="--movie-card-color: ${cardColors[index % cardColors.length]}" aria-label="Flip stats card for ${escapeHtml(row.title)}">
          <span class="movie-card-inner">
            <span class="movie-card-face movie-card-front">
              ${saveButton}
              <dl class="movie-card-copy">
                <dt><strong>Movie name:</strong></dt>
                <dd>${escapeHtml(row.title)} (${row.year})</dd>
                <dt><strong>Genres:</strong></dt>
                <dd>${escapeHtml(row.genres.replaceAll("|", ", "))}</dd>
                <dt><strong>Runtime:</strong></dt>
                <dd>${row.runtime} min</dd>
                <dt><strong>IMDb rating:</strong></dt>
                <dd>${fmt(row.imdb_rating)}</dd>
              </dl>
              ${posterMarkup(row)}
            </span>
            <span class="movie-card-face movie-card-back">
              ${saveButton}
              <dl class="stat-list">
                <dt><strong>Predicted rating:</strong></dt>
                <dd>${fmt(row.predicted_rating)} / 5</dd>
                <dt><strong>95% prediction interval:</strong></dt>
                <dd>[${fmt(row.bootstrap_low_rating)}, ${fmt(row.bootstrap_high_rating)}]</dd>
                <dt><strong>Uncertainty level:</strong></dt>
                <dd>${uncertainty}</dd>
                <dt><strong>Final recommendation score:</strong></dt>
                <dd>${fmt(row.decision_score)}</dd>
              </dl>
            </span>
          </span>
        </article>
      `;
    })
    .join("");

  $$(".movie-save-star").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const row = rows.find((item) => String(item.movie_id) === button.dataset.saveMovie);
      if (row) toggleSavedMovie(row);
      renderMovieCards(rows);
    });
  });

  $$(".movie-flip-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.closest(".movie-save-star")) return;
      card.classList.toggle("flipped");
    });
    card.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      card.classList.toggle("flipped");
    });
  });
}

function toggleTheme() {
  const isDark = document.body.classList.toggle("dark-mode");
  $("#themeToggle").setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
}

function toggleSavedMovie(row) {
  const movieId = String(row.movie_id);
  if (watchlist.has(movieId)) {
    watchlist.delete(movieId);
  } else {
    watchlist.set(movieId, {
      title: row.title,
      year: row.year,
      genres: row.genres,
      predicted_rating: row.predicted_rating,
      decision_score: row.decision_score,
    });
  }
  localStorage.setItem("magicMovieWatchlist", JSON.stringify(Array.from(watchlist.entries())));
  renderWatchlist();
}

function openWatchlistPanel() {
  const overlay = $("#watchlistOverlay");
  overlay.classList.add("open");
  overlay.setAttribute("aria-hidden", "false");
}

function closeWatchlistPanel() {
  const overlay = $("#watchlistOverlay");
  overlay.classList.remove("open");
  overlay.setAttribute("aria-hidden", "true");
}

function renderWatchlist() {
  $("#watchlistCount").textContent = watchlist.size;
  $("#clearWatchlist").disabled = watchlist.size === 0;
  const items = Array.from(watchlist.values()).sort((a, b) => Number(b.decision_score || 0) - Number(a.decision_score || 0));
  $("#watchlistItems").innerHTML = items.length
    ? items
        .map((item) => `
          <div class="watchlist-item">
            <strong>${escapeHtml(item.title)} (${item.year})</strong>
            <span class="watchlist-score">${fmt(item.decision_score || item.predicted_rating)}</span>
          </div>
        `)
        .join("")
    : `<p class="watchlist-empty">Star a recommendation to save it here.</p>`;
}

function clearWatchlist() {
  watchlist.clear();
  localStorage.setItem("magicMovieWatchlist", JSON.stringify([]));
  renderWatchlist();
  if (state.lastRecommendations.length) {
    renderMovieCards(state.lastRecommendations);
  }
}

function posterMarkup(row) {
  if (!row.poster_url) {
    return `<span class="poster-slot" aria-hidden="true"></span>`;
  }
  return `
    <span class="poster-slot">
      <span class="poster-frame">
        <img src="${escapeHtml(row.poster_url)}" alt="Poster for ${escapeHtml(row.title)}" loading="lazy" />
      </span>
    </span>
  `;
}

function openStatsGuide() {
  const overlay = $("#statsGuideOverlay");
  overlay.classList.add("open");
  overlay.setAttribute("aria-hidden", "false");
}

function closeStatsGuide() {
  const overlay = $("#statsGuideOverlay");
  overlay.classList.remove("open");
  overlay.setAttribute("aria-hidden", "true");
}

function renderContributions(rows, topPick) {
  $("#contributionRows").innerHTML = rows
    .map((row) => `
      <tr>
        <td>${row.factor}</td>
        <td class="score">${signed(row.effect)}</td>
      </tr>
    `)
    .join("");
  $("#explanationText").textContent = topPick.explanation;
}

async function loadModels() {
  const data = await fetchJson("/api/models");
  drawBarChart($("#modelChart"), data.models.map((item) => item.name), data.models.map((item) => item.rmse), "RMSE");
  $("#coefficientRows").innerHTML = data.coefficients
    .map((row) => `
      <tr>
        <td>${row.feature}</td>
        <td class="score">${signed(row.estimate)}</td>
        <td>${Number(row.p_value).toFixed(3)}</td>
      </tr>
    `)
    .join("");
}

async function loadDiagnostics() {
  const data = await fetchJson("/api/diagnostics");
  drawBarChart(
    $("#diagnosticChart"),
    data.residualHistogram.map((item) => item.x.toFixed(2)),
    data.residualHistogram.map((item) => item.y),
    "Residual count",
  );
  $("#assumptions").innerHTML = data.assumptions.map((item) => `<li>${item}</li>`).join("");
}

function drawBarChart(canvas, labels, values, label) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight || 260;
  const scale = window.devicePixelRatio || 1;
  canvas.width = width * scale;
  canvas.height = height * scale;
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, width, height);

  const padding = { left: 46, right: 16, top: 24, bottom: 46 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const max = Math.max(...values) * 1.15 || 1;
  const barW = chartW / values.length;

  ctx.fillStyle = "#14213d";
  ctx.font = "700 13px system-ui";
  ctx.fillText(label, padding.left, 16);

  ctx.strokeStyle = "#d9e1ea";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + chartH);
  ctx.lineTo(padding.left + chartW, padding.top + chartH);
  ctx.stroke();

  values.forEach((value, index) => {
    const x = padding.left + index * barW + barW * 0.18;
    const barH = (value / max) * chartH;
    const y = padding.top + chartH - barH;
    ctx.fillStyle = index % 2 ? "#b45309" : "#0f766e";
    ctx.fillRect(x, y, barW * 0.64, barH);
    ctx.fillStyle = "#607085";
    ctx.font = "11px system-ui";
    ctx.save();
    ctx.translate(x + barW * 0.28, padding.top + chartH + 12);
    ctx.rotate(-Math.PI / 5);
    ctx.fillText(labels[index], 0, 0);
    ctx.restore();
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed: ${url}`);
  return response.json();
}

function fmt(value) {
  return Number(value).toFixed(2);
}

function signed(value) {
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function debounce(fn, wait) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), wait);
  };
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

function rubberBand(distance) {
  const limit = 28;
  return clamp(distance * 0.32, -limit, limit);
}

function showLanding() {
  document.body.classList.remove("quiz-visible", "app-visible");
  document.body.classList.add("landing-visible");
  $("#quizShell").setAttribute("aria-hidden", "true");
  $("#appShell").setAttribute("aria-hidden", "true");
  quizState.index = 0;
  quizState.answers = {};
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showQuiz(options = {}) {
  document.body.classList.remove("landing-visible", "app-visible");
  document.body.classList.add("quiz-visible");
  $("#quizShell").setAttribute("aria-hidden", "false");
  $("#appShell").setAttribute("aria-hidden", "true");
  quizState.index = 0;
  if (!options.preserveAnswers) {
    quizState.answers = {};
  }
  renderQuizCard();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showResultsPage() {
  document.body.classList.remove("landing-visible", "quiz-visible");
  document.body.classList.add("app-visible");
  $("#quizShell").setAttribute("aria-hidden", "true");
  $("#appShell").setAttribute("aria-hidden", "false");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.addEventListener("resize", debounce(() => {
  loadModels();
  loadDiagnostics();
}, 150));

init();
