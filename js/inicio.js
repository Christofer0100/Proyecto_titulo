

document.querySelector("form").addEventListener("submit", function (event) {
  event.preventDefault(); // evita que se recargue la página
  window.location.href = "/coordinador.html"; // redirige a home.html
});


