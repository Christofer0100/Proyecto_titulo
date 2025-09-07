

document.querySelector("form").addEventListener("submit", function (event) {
  event.preventDefault(); // evita que se recargue la página
  window.location.href = "/coordinador.html"; // redirige a home.html
});


const token = localStorage.getItem("coord_token");
if (!token) {
  window.location.href = "inicio_sesion.html";
}

// Al llamar a la API, envía el token:
fetch("https://TU-DOMINIO/api/solicitudes/", {
  headers: {
    "Accept": "application/json",
    "Authorization": "Bearer " + token
  }
});