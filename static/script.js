const featureGrid = document.querySelector("#feature-grid");
const featureCount = document.querySelector("#feature-count");
const form = document.querySelector("#prediction-form");
const formMessage = document.querySelector("#form-message");
const statsMessage = document.querySelector("#stats-message");
const predictButton = document.querySelector("#predict-button");
const resultCard = document.querySelector("#result-card");

let featureNames = [];

function formatLabel(name) {
  const [base, suffix] = name.split("_");
  const labelBase = base.replace(/\b\w/g, (letter) => letter.toUpperCase());
  return suffix ? `${labelBase} (${suffix})` : labelBase;
}

function showMessage(element, message = "") {
  element.textContent = message;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Something went wrong. Please try again.");
  }
  return data;
}

function buildInputs() {
  featureGrid.replaceChildren();
  featureNames.forEach((name, index) => {
    const field = document.createElement("div");
    field.className = "feature-field";

    const label = document.createElement("label");
    label.htmlFor = `feature-${index}`;
    label.textContent = formatLabel(name);

    const input = document.createElement("input");
    input.id = `feature-${index}`;
    input.name = name;
    input.type = "number";
    input.step = "any";
    input.inputMode = "decimal";
    input.required = true;
    input.setAttribute("aria-label", formatLabel(name));

    field.append(label, input);
    featureGrid.append(field);
  });
  featureCount.textContent = `${featureNames.length} features`;
}

async function loadFeatures() {
  try {
    const data = await fetchJson("/features");
    featureNames = data.features;
    buildInputs();
  } catch (error) {
    featureCount.textContent = "Features unavailable";
    showMessage(formMessage, error.message);
  }
}

async function loadSample(type) {
  showMessage(formMessage);
  try {
    const sample = await fetchJson(`/sample?type=${encodeURIComponent(type)}`);
    sample.values.forEach((value, index) => {
      const input = form.elements.namedItem(sample.features[index]);
      input.value = value;
    });
    resultCard.hidden = true;
  } catch (error) {
    showMessage(formMessage, error.message);
  }
}

function showPrediction(modelKey, result) {
  const name = modelKey === "logistic_regression" ? "logistic" : "svm";
  const badge = document.querySelector(`#${name}-prediction`);
  const probabilityText = document.querySelector(`#${name}-probability`);
  const bar = document.querySelector(`#${name}-bar`);
  const percentage = result.probability * 100;

  badge.textContent = result.prediction;
  badge.className = `prediction-badge ${result.prediction.toLowerCase()}`;
  probabilityText.textContent = `${percentage.toFixed(1)}%`;
  bar.style.width = `${percentage}%`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showMessage(formMessage);

  const values = featureNames.map((name) => {
    const input = form.elements.namedItem(name);
    return input.value.trim() === "" ? null : Number(input.value);
  });
  if (values.some((value) => value === null || !Number.isFinite(value))) {
    showMessage(formMessage, "Please enter a valid number for every feature.");
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "Predicting…";
  try {
    const results = await fetchJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    showPrediction("logistic_regression", results.logistic_regression);
    showPrediction("svm", results.svm);
    resultCard.hidden = false;
    resultCard.classList.remove("is-visible");
    requestAnimationFrame(() => resultCard.classList.add("is-visible"));
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showMessage(formMessage, error.message);
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict";
  }
});

document.querySelector("#benign-sample").addEventListener("click", () => loadSample("benign"));
document.querySelector("#malignant-sample").addEventListener("click", () => loadSample("malignant"));
document.querySelector("#clear-button").addEventListener("click", () => {
  form.reset();
  resultCard.hidden = true;
  showMessage(formMessage);
});

function renderStats(stats) {
  const labels = [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
  ];
  const models = [
    ["logistic_regression", "Logistic Regression"],
    ["svm", "Support Vector Machine"],
  ];

  document.querySelector("#stats-grid").replaceChildren(
    ...models.map(([key, title]) => {
      const article = document.createElement("article");
      article.className = "stats-model";
      const heading = document.createElement("h3");
      heading.textContent = title;
      const table = document.createElement("table");
      table.className = "stats-table";
      const body = document.createElement("tbody");

      labels.forEach(([metric, label]) => {
        const row = document.createElement("tr");
        const name = document.createElement("th");
        const value = document.createElement("td");
        name.scope = "row";
        name.textContent = label;
        value.textContent = `${(stats[key][metric] * 100).toFixed(1)}%`;
        row.append(name, value);
        body.append(row);
      });

      table.append(body);
      article.append(heading, table);
      return article;
    }),
  );
}

async function loadStats() {
  try {
    renderStats(await fetchJson("/stats"));
  } catch (error) {
    showMessage(statsMessage, error.message);
  }
}

loadFeatures();
loadStats();
