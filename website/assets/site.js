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
