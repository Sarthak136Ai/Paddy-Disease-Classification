/**
 * Main Application JavaScript
 * Global interactions, mobile menu, and notification toasts.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Mobile navigation toggle
    const mobileToggle = document.querySelector(".mobile-toggle");
    const navLinks = document.querySelector(".nav-links");

    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener("click", () => {
            navLinks.classList.toggle("open");
        });
    }

    // Auto-dismiss alerts
    const alerts = document.querySelectorAll(".alert-dismissible");
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = "0";
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    });
});

/**
 * Utility: Show dynamic toast message
 */
function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: ${type === 'danger' ? '#dc2626' : (type === 'success' ? '#059669' : '#0f172a')};
        color: #fff;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 9999;
        transition: all 0.3s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
