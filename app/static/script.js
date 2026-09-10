const inputText = document.getElementById("inputText");
const rail = document.getElementById("suggestionRail");
const errorBox = document.getElementById("errorBox");
const engineButtons = document.querySelectorAll(".engine-btn");

let currentSuggestions = [];
let currentEngine = "transformer";
let debounceTimer = null;

engineButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    engineButtons.forEach((b) => {
      b.classList.remove("is-active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("is-active");
    btn.setAttribute("aria-selected", "true");
    currentEngine = btn.dataset.engine;
    fetchSuggestions();
  });
});

async function fetchSuggestions() {
  const text = inputText.value.trim();
  if (!text) {
    renderEmpty();
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, engine: currentEngine, top_k: 5 }),
    });
    const data = await response.json();

    if (data.error) {
      showError(data.error);
      currentSuggestions = [];
      renderSuggestions();
      return;
    }

    hideError();
    currentSuggestions = data.suggestions || [];
    renderSuggestions();
  } catch (err) {
    showError("Could not reach the prediction server. Is it running?");
  }
}

function renderEmpty() {
  currentSuggestions = [];
  rail.innerHTML = '<p class="rail-empty">Suggestions will line up here as you type.</p>';
}

function renderSuggestions() {
  if (currentSuggestions.length === 0) {
    renderEmpty();
    return;
  }

  const maxProb = Math.max(...currentSuggestions.map((s) => s.probability));
  rail.innerHTML = "";

  currentSuggestions.forEach((s, i) => {
    const key = document.createElement("button");
    key.className = "key" + (i === 0 ? " is-top" : "");
    key.type = "button";

    const pct = maxProb > 0 ? Math.max(6, (s.probability / maxProb) * 100) : 6;
    key.innerHTML = `
      <span class="word">${s.word}</span>
      <span class="prob-track"><span class="prob-fill" style="width:${pct}%"></span></span>
    `;
    key.addEventListener("click", () => insertWord(s.word));
    rail.appendChild(key);
  });
}

function insertWord(word) {
  const trimmed = inputText.value.replace(/\s+$/, "");
  inputText.value = trimmed + (trimmed ? " " : "") + word + " ";
  inputText.focus();
  fetchSuggestions();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

inputText.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    if (currentSuggestions.length > 0) {
      insertWord(currentSuggestions[0].word);
    }
  }
});

inputText.addEventListener("keyup", (event) => {
  if (event.key === " ") {
    fetchSuggestions();
    return;
  }
  // Light debounce so suggestions also refresh a moment after any typing,
  // not just on space, without hammering the API on every keystroke.
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchSuggestions, 450);
});
