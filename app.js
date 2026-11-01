(function () {
  "use strict";

  const FEATURE_COUNT = 6;
  const MIN_IMPORT_RECORDS = 12;
  const DEMO_SOURCE = "Bundled demo dataset";
  const elements = {
    irradiance: document.getElementById("irradiance"),
    irradianceOutput: document.getElementById("irradiance-output"),
    temperature: document.getElementById("temperature"),
    temperatureOutput: document.getElementById("temperature-output"),
    prediction: document.getElementById("prediction-value"),
    predictionContext: document.getElementById("prediction-context"),
    rangeWarning: document.getElementById("range-warning"),
    sourceName: document.getElementById("source-name"),
    recordCount: document.getElementById("record-count"),
    mae: document.getElementById("mae-value"),
    r2: document.getElementById("r2-value"),
    tableRows: document.getElementById("dataset-rows"),
    tableCount: document.getElementById("table-count"),
    chartTemperature: document.getElementById("chart-temperature"),
    chart: document.getElementById("efficiency-chart"),
    upload: document.getElementById("csv-upload"),
    download: document.getElementById("download-button"),
    reset: document.getElementById("reset-button"),
    status: document.getElementById("status-message")
  };
  let dataset = [];
  let model = null;
  let currentSource = DEMO_SOURCE;
  let resizeObserver;

  function featureVector(irradiance, temperature) {
    const light = irradiance / 1000;
    const heat = (temperature - 25) / 25;
    return [1, light, heat, light * light, heat * heat, light * heat];
  }

  function solveLinearSystem(matrix, vector) {
    const size = vector.length;
    const rows = matrix.map((row, index) => [...row, vector[index]]);

    for (let column = 0; column < size; column += 1) {
      let pivot = column;
      for (let row = column + 1; row < size; row += 1) {
        if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
      }
      if (Math.abs(rows[pivot][column]) < 1e-10) {
        throw new Error("The dataset does not contain enough variation in sunlight and temperature to fit this model.");
      }
      [rows[column], rows[pivot]] = [rows[pivot], rows[column]];

      const divisor = rows[column][column];
      for (let entry = column; entry <= size; entry += 1) rows[column][entry] /= divisor;

      for (let row = 0; row < size; row += 1) {
        if (row === column) continue;
        const factor = rows[row][column];
        for (let entry = column; entry <= size; entry += 1) {
          rows[row][entry] -= factor * rows[column][entry];
        }
      }
    }

    return rows.map((row) => row[size]);
  }

  function fitModel(records) {
    if (records.length < FEATURE_COUNT) {
      throw new Error("At least 6 records are needed to train the prediction model.");
    }

    const matrix = Array.from({ length: FEATURE_COUNT }, () => Array(FEATURE_COUNT).fill(0));
    const vector = Array(FEATURE_COUNT).fill(0);

    records.forEach((record) => {
      const features = featureVector(record.irradiance, record.temperature);
      for (let row = 0; row < FEATURE_COUNT; row += 1) {
        vector[row] += features[row] * record.efficiency;
        for (let column = 0; column < FEATURE_COUNT; column += 1) {
          matrix[row][column] += features[row] * features[column];
        }
      }
    });

    return solveLinearSystem(matrix, vector);
  }

  function predict(coefficients, irradiance, temperature) {
    const features = featureVector(irradiance, temperature);
    return features.reduce((sum, feature, index) => sum + feature * coefficients[index], 0);
  }

  function evaluateModel(records) {
    const testRecords = records.filter((_, index) => index % 5 === 0);
    const trainRecords = records.filter((_, index) => index % 5 !== 0);
    const coefficients = fitModel(trainRecords);
    const errors = testRecords.map((record) => record.efficiency - predict(coefficients, record.irradiance, record.temperature));
    const mae = errors.reduce((sum, error) => sum + Math.abs(error), 0) / errors.length;
    const mean = testRecords.reduce((sum, record) => sum + record.efficiency, 0) / testRecords.length;
    const total = testRecords.reduce((sum, record) => sum + (record.efficiency - mean) ** 2, 0);
    const residual = errors.reduce((sum, error) => sum + error ** 2, 0);
    const r2 = total === 0 ? 0 : 1 - residual / total;
    return { mae, r2 };
  }

  function validateRecords(records) {
    if (!Array.isArray(records) || records.length < MIN_IMPORT_RECORDS) {
      throw new Error(`Please provide at least ${MIN_IMPORT_RECORDS} valid data rows.`);
    }
    records.forEach((record, index) => {
      if (![record.irradiance, record.temperature, record.efficiency].every(Number.isFinite)) {
        throw new Error(`Row ${index + 1} contains a missing or invalid number.`);
      }
      if (record.irradiance < 0 || record.irradiance > 2000) {
        throw new Error(`Row ${index + 1}: sunlight intensity must be between 0 and 2,000 W/m².`);
      }
      if (record.temperature < -50 || record.temperature > 120) {
        throw new Error(`Row ${index + 1}: temperature must be between −50 and 120 °C.`);
      }
      if (record.efficiency < 0 || record.efficiency > 100) {
        throw new Error(`Row ${index + 1}: efficiency must be between 0% and 100%.`);
      }
    });
  }

  function parseCsv(text) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;

    for (let index = 0; index < text.length; index += 1) {
      const character = text[index];
      if (quoted) {
        if (character === '"' && text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else if (character === '"') {
          quoted = false;
        } else {
          field += character;
        }
      } else if (character === '"') {
        quoted = true;
      } else if (character === ",") {
        row.push(field.trim());
        field = "";
      } else if (character === "\n" || character === "\r") {
        if (character === "\r" && text[index + 1] === "\n") index += 1;
        row.push(field.trim());
        if (row.some((cell) => cell !== "")) rows.push(row);
        row = [];
        field = "";
      } else {
        field += character;
      }
    }
    if (quoted) throw new Error("The CSV contains an unclosed quoted field.");
    row.push(field.trim());
    if (row.some((cell) => cell !== "")) rows.push(row);
    if (rows.length < 2) throw new Error("The CSV needs a header row and at least one data row.");

    const normalizeHeader = (value) => value.replace(/^\uFEFF/, "").toLowerCase().replace(/[^a-z0-9]/g, "");
    const headers = rows[0].map(normalizeHeader);
    const findColumn = (aliases) => headers.findIndex((header) => aliases.includes(header));
    const irradianceColumn = findColumn(["irradiancewm2", "irradiance", "sunlightintensity", "solarirradiance", "lightintensity"]);
    const temperatureColumn = findColumn(["temperaturec", "temperature", "paneltemperature", "moduletemperature"]);
    const efficiencyColumn = findColumn(["efficiencypct", "efficiency", "efficiencypercent", "panelefficiency"]);

    if ([irradianceColumn, temperatureColumn, efficiencyColumn].some((column) => column < 0)) {
      throw new Error("CSV headers must identify irradiance, temperature, and efficiency. See the required column names above.");
    }

    return rows.slice(1).map((cells, index) => ({
      irradiance: Number(cells[irradianceColumn]),
      temperature: Number(cells[temperatureColumn]),
      efficiency: Number(cells[efficiencyColumn]),
      recordNumber: index + 1
    }));
  }

  function showStatus(message, isError) {
    elements.status.textContent = message;
    elements.status.classList.toggle("error", Boolean(isError));
    elements.status.hidden = false;
  }

  function clearStatus() {
    elements.status.textContent = "";
    elements.status.classList.remove("error");
    elements.status.hidden = true;
  }

  function loadDataset(records, source) {
    validateRecords(records);
    const metrics = evaluateModel(records);
    const coefficients = fitModel(records);
    dataset = records.map((record) => ({
      irradiance: record.irradiance,
      temperature: record.temperature,
      efficiency: record.efficiency
    }));
    model = { coefficients, metrics };
    currentSource = source;
    render();
  }

  function renderTable() {
    const visibleRecords = dataset.slice(-8).reverse();
    elements.tableRows.innerHTML = visibleRecords.map((record, index) => {
      const recordNumber = dataset.length - index;
      return `<tr><td>#${String(recordNumber).padStart(3, "0")}</td><td>${record.irradiance.toLocaleString()} W/m²</td><td>${record.temperature.toFixed(1)} °C</td><td>${record.efficiency.toFixed(2)}%</td></tr>`;
    }).join("");
    elements.tableCount.textContent = `Showing ${visibleRecords.length} of ${dataset.length} observations`;
  }

  function currentPrediction() {
    return predict(model.coefficients, Number(elements.irradiance.value), Number(elements.temperature.value));
  }

  function renderPrediction() {
    const irradiance = Number(elements.irradiance.value);
    const temperature = Number(elements.temperature.value);
    const estimate = currentPrediction();
    const boundedEstimate = Math.max(0, Math.min(100, estimate));
    const minIrradiance = Math.min(...dataset.map((record) => record.irradiance));
    const maxIrradiance = Math.max(...dataset.map((record) => record.irradiance));
    const minTemperature = Math.min(...dataset.map((record) => record.temperature));
    const maxTemperature = Math.max(...dataset.map((record) => record.temperature));
    const outsideRange = irradiance < minIrradiance || irradiance > maxIrradiance || temperature < minTemperature || temperature > maxTemperature;

    elements.irradianceOutput.innerHTML = `${irradiance.toLocaleString()} <span>W/m²</span>`;
    elements.temperatureOutput.innerHTML = `${temperature} <span>°C</span>`;
    elements.prediction.textContent = boundedEstimate.toFixed(1);
    elements.predictionContext.textContent = outsideRange ? "Outside the dataset's observed range" : "Based on your dataset";
    elements.rangeWarning.hidden = !outsideRange;
    elements.chartTemperature.textContent = `${temperature} °C`;
  }

  function renderMetrics() {
    elements.sourceName.textContent = currentSource;
    elements.recordCount.textContent = dataset.length.toLocaleString();
    elements.mae.textContent = model.metrics.mae.toFixed(2);
    elements.r2.textContent = model.metrics.r2.toFixed(3);
    elements.reset.hidden = currentSource === DEMO_SOURCE;
    renderTable();
  }

  function drawChart() {
    const canvas = elements.chart;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width === 0 || bounds.height === 0) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * ratio);
    canvas.height = Math.round(bounds.height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    const width = bounds.width;
    const height = bounds.height;
    const plot = { left: 43, right: width - 12, top: 12, bottom: height - 31 };
    const xMax = Math.max(1000, Math.ceil(Math.max(...dataset.map((record) => record.irradiance)) / 200) * 200);
    const values = dataset.map((record) => record.efficiency);
    const currentEfficiency = currentPrediction();
    const lowY = Math.max(0, Math.floor((Math.min(...values, currentEfficiency) - 2) / 2) * 2);
    const highY = Math.min(100, Math.ceil((Math.max(...values, currentEfficiency) + 2) / 2) * 2);
    const yMin = Math.min(lowY, 8);
    const yMax = Math.max(highY, yMin + 4);
    const x = (value) => plot.left + (value / xMax) * (plot.right - plot.left);
    const y = (value) => plot.bottom - ((value - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);

    context.clearRect(0, 0, width, height);
    context.font = '9px "DM Sans", sans-serif';
    context.textBaseline = "middle";
    context.textAlign = "right";
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = yMin + ((yMax - yMin) * tick) / 4;
      const position = y(value);
      context.beginPath();
      context.strokeStyle = "#edf0eb";
      context.lineWidth = 1;
      context.moveTo(plot.left, position);
      context.lineTo(plot.right, position);
      context.stroke();
      context.fillStyle = "#929c94";
      context.fillText(value.toFixed(0), plot.left - 9, position);
    }

    context.textAlign = "center";
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = (xMax * tick) / 4;
      context.fillStyle = "#929c94";
      context.fillText(value.toFixed(0), x(value), plot.bottom + 15);
    }
    context.fillStyle = "#8b958d";
    context.font = '8px "DM Sans", sans-serif';
    context.fillText("SUNLIGHT INTENSITY (W/m²)", (plot.left + plot.right) / 2, height - 5);

    dataset.forEach((record) => {
      context.beginPath();
      context.fillStyle = "rgba(114, 157, 117, .55)";
      context.arc(x(record.irradiance), y(record.efficiency), 3, 0, Math.PI * 2);
      context.fill();
    });

    context.beginPath();
    context.strokeStyle = "#34794a";
    context.lineWidth = 2;
    for (let point = 0; point <= 60; point += 1) {
      const irradiance = (xMax * point) / 60;
      const positionX = x(irradiance);
      const positionY = y(predict(model.coefficients, irradiance, Number(elements.temperature.value)));
      if (point === 0) context.moveTo(positionX, positionY);
      else context.lineTo(positionX, positionY);
    }
    context.stroke();

    const selectedX = x(Number(elements.irradiance.value));
    const selectedY = y(currentEfficiency);
    context.beginPath();
    context.fillStyle = "#ffffff";
    context.strokeStyle = "#e3a243";
    context.lineWidth = 2;
    context.arc(selectedX, selectedY, 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }

  function render() {
    renderMetrics();
    renderPrediction();
    drawChart();
  }

  function downloadCsv() {
    const header = "irradiance_w_m2,temperature_c,efficiency_pct";
    const rows = dataset.map((record) => `${record.irradiance},${record.temperature},${record.efficiency}`);
    const blob = new Blob([[header, ...rows].join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = currentSource === DEMO_SOURCE ? "solar-efficiency-demo.csv" : "solar-efficiency-dataset.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  elements.irradiance.addEventListener("input", render);
  elements.temperature.addEventListener("input", render);
  elements.download.addEventListener("click", downloadCsv);
  elements.upload.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    try {
      const records = parseCsv(await file.text());
      validateRecords(records);
      loadDataset(records, file.name);
      showStatus(`Trained a new model with ${records.length} observations from ${file.name}.`, false);
    } catch (error) {
      showStatus(error instanceof Error ? error.message : "The dataset could not be loaded.", true);
    } finally {
      elements.upload.value = "";
    }
  });
  elements.reset.addEventListener("click", () => {
    try {
      loadDataset(window.SOLAR_DEMO_DATA, DEMO_SOURCE);
      clearStatus();
    } catch (error) {
      showStatus(error instanceof Error ? error.message : "The demo dataset could not be restored.", true);
    }
  });

  try {
    if (!Array.isArray(window.SOLAR_DEMO_DATA)) {
      throw new Error("The bundled dataset could not be found. Ensure solar-dataset.js is next to index.html.");
    }
    loadDataset(window.SOLAR_DEMO_DATA, DEMO_SOURCE);
    if ("ResizeObserver" in window) {
      resizeObserver = new ResizeObserver(drawChart);
      resizeObserver.observe(elements.chart);
    } else {
      window.addEventListener("resize", drawChart);
    }
  } catch (error) {
    showStatus(error instanceof Error ? error.message : "The prediction model could not be initialized.", true);
  }
}());
