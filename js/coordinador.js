// js/coordinador.js
(() => {
  // Menú lateral
  const sideMenu = document.querySelector('aside');
  const menuBtn  = document.getElementById('menu-btn');
  const closeBtn = document.getElementById('close-btn');

  if (menuBtn && sideMenu) {
    menuBtn.addEventListener('click', () => { sideMenu.style.display = 'block'; });
  }
  if (closeBtn && sideMenu) {
    closeBtn.addEventListener('click', () => { sideMenu.style.display = 'none'; });
  }

  // IMPORTANTE:
  // No manejar tema aquí. El tema se administra en el script del HTML
  // con el contenedor #theme-toggle (usa localStorage y data-theme).
})();
