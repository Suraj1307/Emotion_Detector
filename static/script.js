document.addEventListener("DOMContentLoaded", function () {
    const textInput = document.getElementById("textInput");
    const charCount = document.getElementById("charCount");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const emotionForm = document.getElementById("emotionForm");
    const spinner = document.getElementById("spinner");
    const analyzingStatus = document.getElementById("analyzingStatus");
    const typingFeedback = document.getElementById("typingFeedback");
    const clearBtn = document.getElementById("clearBtn");
    const copyResultBtn = document.getElementById("copyResultBtn");
    const printResultBtn = document.getElementById("printResultBtn");
    const confidenceGauge = document.getElementById("confidenceGauge");
    const batchForm = document.getElementById("batchForm");
    const batchBtn = document.getElementById("batchBtn");

    const updateInputState = () => {
        if (!textInput || !charCount) return;
        const value = textInput.value || "";
        const length = value.length;
        charCount.textContent = length;

        if (analyzeBtn) {
            analyzeBtn.disabled = value.trim().length === 0;
        }

        if (!typingFeedback) return;
        if (length === 0) {
            typingFeedback.textContent = "Best results with 20-200 characters.";
        } else if (length < 20) {
            typingFeedback.textContent = "Text is short. Add more details for better signal.";
        } else if (length <= 200) {
            typingFeedback.textContent = "Good length for stable prediction.";
        } else {
            typingFeedback.textContent = "Long text detected. Keep only the most relevant lines.";
        }
    };

    if (textInput) {
        textInput.addEventListener("input", updateInputState);
        updateInputState();
    }

    if (clearBtn && textInput) {
        clearBtn.addEventListener("click", () => {
            textInput.value = "";
            updateInputState();
            textInput.focus();
        });
    }

    if (emotionForm) {
        emotionForm.addEventListener("submit", function () {
            if (spinner) spinner.style.display = "inline-block";
            if (analyzingStatus) analyzingStatus.textContent = "Analyzing...";
        });
    }

    if (batchForm && batchBtn) {
        batchForm.addEventListener("submit", function () {
            batchBtn.textContent = "Analyzing batch...";
            batchBtn.disabled = true;
        });
    }

    document.querySelectorAll(".example-btn").forEach((button) => {
        button.addEventListener("click", () => {
            if (!textInput) return;
            const text = button.getAttribute("data-example") || "";
            textInput.value = text;
            updateInputState();
            textInput.focus();
        });
    });

    document.querySelectorAll(".copy-example-btn").forEach((button) => {
        button.addEventListener("click", async () => {
            const text = button.getAttribute("data-copy") || "";
            if (!text) return;
            try {
                await navigator.clipboard.writeText(text);
                button.textContent = "Copied";
                setTimeout(() => {
                    button.textContent = "Copy";
                }, 1000);
            } catch (err) {
                button.textContent = "Failed";
            }
        });
    });

    if (copyResultBtn) {
        copyResultBtn.addEventListener("click", async () => {
            const resultCard = document.getElementById("resultCard");
            if (!resultCard) return;
            const text = resultCard.innerText;
            try {
                await navigator.clipboard.writeText(text);
                copyResultBtn.textContent = "Copied";
                setTimeout(() => {
                    copyResultBtn.textContent = "Copy Result";
                }, 1200);
            } catch (err) {
                copyResultBtn.textContent = "Copy Failed";
            }
        });
    }

    if (printResultBtn) {
        printResultBtn.addEventListener("click", () => {
            window.print();
        });
    }

    if (confidenceGauge) {
        const value = parseFloat(confidenceGauge.getAttribute("data-confidence") || "0");
        const safe = Math.max(0, Math.min(100, value));
        confidenceGauge.style.setProperty("--gauge-value", safe + "%");
        confidenceGauge.setAttribute("aria-label", `Confidence ${safe}%`);
    }

    document.querySelectorAll(".prob-fill").forEach((bar) => {
        const prob = parseFloat(bar.getAttribute("data-prob") || "0");
        bar.style.width = `${Math.max(0, Math.min(100, prob))}%`;
    });

    const trendContainer = document.getElementById("trendContainer");
    if (trendContainer) {
        const raw = trendContainer.getAttribute("data-history");
        let data = [];
        try {
            data = raw ? JSON.parse(raw) : [];
        } catch (err) {
            data = [];
        }

        if (!Array.isArray(data) || data.length === 0) {
            trendContainer.innerHTML = '<div class="helper-text">No trend data yet.</div>';
            return;
        }

        const maxCount = Math.max(...data.map((d) => d.count), 1);
        trendContainer.innerHTML = data
            .map((d) => {
                const width = Math.round((d.count / maxCount) * 100);
                return `
                    <div class="trend-item">
                        <span class="trend-label">${d.emotion}</span>
                        <div class="trend-bar"><div class="trend-fill" style="width:${width}%"></div></div>
                        <span class="trend-value">${d.count}</span>
                    </div>
                `;
            })
            .join("");
    }
});
