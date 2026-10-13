/**
 * Performance Visualizations and Live Benchmarking Charts
 * Uses Chart.js for responsive, crisp visual comparisons.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Performance Page Charts if canvas elements exist
    initAccuracyF1Chart();
    initLatencyChart();
    initAccuracyParamsChart();
    initLiveBenchmarkRunner();
});

/**
 * Chart 1: Grouped Bar Chart - Test Accuracy vs Macro F1
 */
function initAccuracyF1Chart() {
    const ctx = document.getElementById("accuracyF1Chart");
    if (!ctx || !window.chartData) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: window.chartData.labels,
            datasets: [
                {
                    label: "Test Accuracy (%)",
                    data: window.chartData.accuracies,
                    backgroundColor: "rgba(5, 150, 105, 0.85)",
                    borderColor: "rgba(4, 120, 87, 1)",
                    borderWidth: 1.5,
                    borderRadius: 6
                },
                {
                    label: "Macro F1 Score (%)",
                    data: window.chartData.f1_scores,
                    backgroundColor: "rgba(37, 99, 235, 0.85)",
                    borderColor: "rgba(29, 78, 216, 1)",
                    borderWidth: 1.5,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top",
                    labels: { font: { family: "Inter", weight: "600", size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.raw}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: "Percentage (%)", font: { weight: "600" } },
                    grid: { color: "rgba(226, 232, 240, 0.8)" }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Chart 2: Horizontal Bar Chart - Inference Latency (ms/image)
 */
function initLatencyChart() {
    const ctx = document.getElementById("latencyChart");
    if (!ctx || !window.chartData) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: window.chartData.labels,
            datasets: [
                {
                    label: "Inference Latency (ms/image)",
                    data: window.chartData.latencies,
                    backgroundColor: [
                        "rgba(5, 150, 105, 0.85)",
                        "rgba(16, 185, 129, 0.85)",
                        "rgba(217, 119, 6, 0.85)",
                        "rgba(59, 130, 246, 0.85)",
                        "rgba(148, 163, 184, 0.85)"
                    ],
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `Latency: ${context.raw} ms/image (lower is faster)`
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: { display: true, text: "Latency (ms) — Lower is Better", font: { weight: "600" } },
                    grid: { color: "rgba(226, 232, 240, 0.8)" }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Chart 3: Bubble / Scatter Plot - Accuracy vs Total Parameters
 */
function initAccuracyParamsChart() {
    const ctx = document.getElementById("accuracyParamsChart");
    if (!ctx || !window.chartData || !window.chartData.scatter_data) return;

    new Chart(ctx, {
        type: "bubble",
        data: {
            datasets: window.chartData.scatter_data.map((item, index) => {
                const colors = [
                    "rgba(5, 150, 105, 0.85)",
                    "rgba(16, 185, 129, 0.85)",
                    "rgba(217, 119, 6, 0.85)",
                    "rgba(59, 130, 246, 0.85)",
                    "rgba(148, 163, 184, 0.85)"
                ];
                return {
                    label: item.model_name,
                    data: [{ x: item.x, y: item.y, r: item.r }],
                    backgroundColor: colors[index % colors.length],
                    borderColor: colors[index % colors.length].replace("0.85", "1"),
                    borderWidth: 1.5,
                    latency: item.latency,
                    params: item.params
                };
            })
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { font: { family: "Inter", weight: "500", size: 11 } }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const ds = context.dataset;
                            return `${ds.label}: ${context.raw.y}% Acc, ${context.raw.x}M params, ${ds.latency} ms/img`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: "linear",
                    title: { display: true, text: "Total Parameters (Millions)", font: { weight: "600" } },
                    grid: { color: "rgba(226, 232, 240, 0.8)" }
                },
                y: {
                    min: 40,
                    max: 100,
                    title: { display: true, text: "Test Accuracy (%)", font: { weight: "600" } },
                    grid: { color: "rgba(226, 232, 240, 0.8)" }
                }
            }
        }
    });
}

/**
 * Live Benchmarking Suite Controller
 */
function initLiveBenchmarkRunner() {
    const runBtn = document.getElementById("runLiveBenchmarkBtn");
    const modelSelect = document.getElementById("benchModelSelect");
    const iterationsInput = document.getElementById("benchIterationsInput");
    const resultsPanel = document.getElementById("liveBenchResultsPanel");

    if (!runBtn) return;

    runBtn.addEventListener("click", async () => {
        const modelKey = modelSelect ? modelSelect.value : "resnet50";
        const iters = iterationsInput ? parseInt(iterationsInput.value) || 50 : 50;

        runBtn.disabled = true;
        const originalHtml = runBtn.innerHTML;
        runBtn.innerHTML = `<div class="spinner" style="width: 16px; height: 16px; border-width: 2px; display: inline-block; vertical-align: middle; margin-right: 6px;"></div> Benchmarking (${iters} passes)...`;

        try {
            const resp = await fetch("/api/benchmark", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: modelKey,
                    warm_up: 15,
                    iterations: iters
                })
            });

            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.error || "Benchmark failed.");
            }

            const res = data.data;

            // Render live benchmark metrics
            document.getElementById("liveMeanLatency").textContent = `${res.mean_latency_ms} ms`;
            document.getElementById("liveMedianLatency").textContent = `${res.median_latency_ms} ms`;
            document.getElementById("liveStdLatency").textContent = `±${res.std_latency_ms} ms`;
            document.getElementById("liveFps").textContent = `${res.fps} img/s`;
            document.getElementById("liveDevice").textContent = `${res.device_name} (${res.device})`;
            document.getElementById("liveRecordedBaseline").textContent = `${res.recorded_latency_ms} ms`;

            if (resultsPanel) {
                resultsPanel.style.display = "block";
                resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            showToast(`Benchmark complete for ${res.model_name}!`, "success");

        } catch (error) {
            console.error("Benchmarking error:", error);
            showToast(error.message, "danger");
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = originalHtml;
        }
    });
}
