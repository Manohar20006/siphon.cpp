document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("action-btn");
  const statusText = document.getElementById("status-text");
  btn.addEventListener("click", () => {
    statusText.textContent = "Status updated: Button was clicked successfully!";
    btn.textContent = "Clicked!";
  });
});
