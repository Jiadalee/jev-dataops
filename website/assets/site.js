"use strict";

// Small progressive enhancements. Content and navigation work without JavaScript.
const menu = document.querySelector(".menu-toggle");
const navigation = document.getElementById("site-nav");
menu?.addEventListener("click", () => {
  const open = menu.getAttribute("aria-expanded") !== "true";
  menu.setAttribute("aria-expanded", String(open));
  navigation.classList.toggle("open", open);
});
for (const link of navigation?.querySelectorAll("a") || []) {
  link.addEventListener("click", () => {
    menu.setAttribute("aria-expanded", "false");
    navigation.classList.remove("open");
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && menu?.getAttribute("aria-expanded") === "true") {
    menu.setAttribute("aria-expanded", "false");
    navigation.classList.remove("open");
    menu.focus();
  }
});

for (const list of document.querySelectorAll('[role="tablist"]')) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  function activate(tab, focus = false) {
    for (const item of tabs) {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.dataset.tab).hidden = !selected;
    }
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault(); activate(tabs[next], true);
    });
  });
}

const gpuControl = document.getElementById("gpu-count-control");
if (gpuControl) {
  function updateGPUCommands() {
    const selected = Number(gpuControl.querySelector('input[name="training-gpus"]:checked')?.value);
    const count = [1, 2, 4, 8].includes(selected) ? selected : 1;
    const batch = count * 4;
    for (const command of document.querySelectorAll("[data-gpu-command]")) {
      command.textContent = command.textContent
        .replace(/--n-gpus \d+/, `--n-gpus ${count}`)
        .replace(/--batch-size \d+/, `--batch-size ${batch}`);
    }
    const workers = document.getElementById("gpu-workers");
    workers.replaceChildren();
    workers.setAttribute("aria-label", `${count} GPU ${count === 1 ? "worker" : "workers"} on one host`);
    for (let index = 1; index <= count; index++) {
      const worker = document.createElement("span");
      worker.className = "gpu-worker";
      worker.textContent = `GPU ${index}`;
      workers.append(worker);
    }
    document.getElementById("gpu-plan-summary").textContent =
      `${count} ${count === 1 ? "GPU" : "GPUs"} · global batch ${batch}. ` +
      `SFT: 4 records per GPU. verl: ${batch} prompts per global batch, before rollout expansion.`;
  }
  gpuControl.addEventListener("change", updateGPUCommands);
  updateGPUCommands();
}

let toastTimer;
function announce(message) {
  const toast = document.getElementById("site-toast");
  clearTimeout(toastTimer);
  toast.textContent = message; toast.hidden = false;
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3500);
}
for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const code = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      announce("Commands copied to your clipboard.");
    } catch (_) {
      const range = document.createRange(); range.selectNodeContents(code);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      announce("Commands selected. Use your browser's copy shortcut.");
    }
  });
}
