# app/webhooks.py
from __future__ import annotations
import re
from datetime import time, datetime
from typing import Any, Dict, Optional

from django.conf import settings
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from . import models  # tus modelos del archivo models.py


# ---------------- utilidades ----------------
def _s(val: Any) -> Optional[str]:
    """str limpio (quita espacios y un '=' inicial que n8n a veces antepone)."""
    if not isinstance(val, str):
        return None
    out = val.strip().lstrip("=")
    return out or None


def _phone(val: Any) -> Optional[str]:
    s = _s(val)
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    # Normaliza a +56XXXXXXXXX si viene con 56
    if digits.startswith("56"):
        return f"+{digits}"
    # Si ya viene con +, déjalo tal cual
    if s.startswith("+"):
        return s
    return digits


def _to_int(val: Any, default: int = 1) -> int:
    try:
        return int(str(val).strip().lstrip("="))
    except Exception:
        return default


def _to_time(val: Any) -> Optional[time]:
    """Acepta '13', '13:00', '13:00:00'."""
    s = _s(val)
    if not s:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            pass
    # Solo hora
    try:
        h = int(s)
        if 0 <= h <= 23:
            return time(hour=h)
    except Exception:
        pass
    return None


def _require_token(request):
    """Token por header. En DEBUG también acepta ?token= o body.token para pruebas."""
    expected = getattr(settings, "WEBHOOK_TOKEN", None)
    if not expected:
        return None
    received = request.META.get("HTTP_X_WEBHOOK_TOKEN")
    if settings.DEBUG and not received:
        try:
            received = request.GET.get("token") or (request.data.get("token") if isinstance(request.data, dict) else None)
        except Exception:
            received = None
    if received != expected:
        return Response({"ok": False, "error": "Token inválido"}, status=401)
    return None


# --------------- WEBHOOK --------------------
@api_view(["POST"])
@csrf_exempt
def whatsapp_webhook(request):
    """
    Recibe el formulario desde n8n y crea:
      - Tenista (por numero)
      - Origen (salida)
      - Destino (lugar)
      - Solicitud (form_*, FKs, estado=NUEVA)
    Devuelve lo insertado desde BD.
    """
    # seguridad
    token_error = _require_token(request)
    if token_error:
        return token_error

    body = request.data or {}
    if not isinstance(body, dict):
        return Response({"ok": False, "error": "JSON inválido"}, status=400)

    # --------- lee campos del formulario ---------
    # Teléfonos
    from_phone = _phone(body.get("from_phone"))          # WhatsApp que escribe
    form_telefono = _s(body.get("telefono")) or from_phone

    if not from_phone:
        return Response({"ok": False, "error": "from_phone requerido"}, status=400)

    # Datos de contacto del formulario
    form_nombres = _s(body.get("nombres")) or "Tenista"
    form_apellidos = _s(body.get("apellidos"))
    form_correo = _s(body.get("correo"))

    # Viaje
    pasajeros = _to_int(body.get("pasajeros"), default=1)
    hora_salida = _to_time(body.get("hora_salida"))
    observaciones = _s(body.get("observaciones"))

    # Origen/Destino: aceptamos string o {direccion:..}/{salida:..}/{lugar:..}
    def _pick_origen(v) -> Optional[str]:
        if isinstance(v, dict):
            return _s(v.get("salida")) or _s(v.get("direccion"))
        return _s(v)

    def _pick_destino(v) -> Optional[str]:
        if isinstance(v, dict):
            return _s(v.get("lugar")) or _s(v.get("direccion"))
        return _s(v)

    origen_txt = _pick_origen(body.get("origen"))
    # si venía "destinos": [ {...} ] nos quedamos con el primero
    if "destinos" in body and isinstance(body["destinos"], list) and body["destinos"]:
        destino_txt = _pick_destino(body["destinos"][0])
    else:
        destino_txt = _pick_destino(body.get("destino"))

    # --------- UPSERT Tenista / Origen / Destino ---------
    tenista, _ = models.Tenista.objects.get_or_create(
        numero=from_phone,
        defaults={
            "nombre": form_nombres,
            "apellido": form_apellidos or "",
            "correo": form_correo,
        },
    )
    # actualiza datos faltantes si llegaron ahora
    changed = False
    if form_correo and not tenista.correo:
        tenista.correo = form_correo; changed = True
    if form_nombres and tenista.nombre in (None, "", "Tenista"):
        tenista.nombre = form_nombres; changed = True
    if form_apellidos and not tenista.apellido:
        tenista.apellido = form_apellidos; changed = True
    if changed:
        tenista.save()

    origen_obj = None
    if origen_txt:
        origen_obj, _ = models.Origen.objects.get_or_create(salida=origen_txt)

    destino_obj = None
    if destino_txt:
        destino_obj, _ = models.Destino.objects.get_or_create(lugar=destino_txt)

    # --------- crear SOLICITUD ---------
    solicitud = models.Solicitud.objects.create(
        form_nombres=form_nombres,
        form_apellidos=form_apellidos or "",
        form_correo=form_correo,
        form_telefono=form_telefono or "",
        pasajeros=pasajeros,
        hora_salida=hora_salida,
        observaciones=observaciones,
        origen=origen_obj,   # FK (puede ser None)
        destino=destino_obj, # FK (puede ser None)
        tenista=tenista,     # FK
        idioma_detectado="es",
        raw_form=body,       # guarda el JSON crudo por auditoría
        estado=models.SolicitudEstado.NUEVA,
        created_at=timezone.now(),  # tu campo no tiene default → lo seteamos
    )

    # --------- respuesta (leído desde BD) ---------
    sol = (models.Solicitud.objects
           .select_related("tenista", "origen", "destino")
           .get(pk=solicitud.pk))

    resp = {
        "ok": True,
        "solicitud": {
            "id": sol.id,
            "estado": sol.estado,
            "created_at": sol.created_at.isoformat() if sol.created_at else None,
            "pasajeros": sol.pasajeros,
            "hora_salida": sol.hora_salida.isoformat() if sol.hora_salida else None,
            "observaciones": sol.observaciones,
            "origen": getattr(sol.origen, "salida", None),
            "destino": getattr(sol.destino, "lugar", None),
            "form": {
                "nombres": sol.form_nombres,
                "apellidos": sol.form_apellidos,
                "correo": sol.form_correo,
                "telefono": sol.form_telefono,
            },
            "tenista": {
                "id": sol.tenista.id,
                "nombre": sol.tenista.nombre,
                "apellido": sol.tenista.apellido,
                "correo": sol.tenista.correo,
                "numero": sol.tenista.numero,
            },
        },
    }
    return Response(resp, status=status.HTTP_201_CREATED)



from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Reusa estos helpers si los tienes en el archivo,
# o comenta las llamadas si no los necesitas.
def _serialize_solicitud(sol):
    """Convierte una Solicitud en un dict legible (anidando FKs)."""
    return {
        "id": sol.id,
        "estado": sol.estado,
        "created_at": sol.created_at.isoformat() if getattr(sol, "created_at", None) else None,
        "pasajeros": sol.pasajeros,
        "hora_salida": sol.hora_salida.isoformat() if sol.hora_salida else None,
        "observaciones": sol.observaciones,
        "origen": getattr(sol.origen, "salida", None),
        "destino": getattr(sol.destino, "lugar", None),
        "form": {
            "nombres": sol.form_nombres,
            "apellidos": sol.form_apellidos,
            "correo": sol.form_correo,
            "telefono": sol.form_telefono,
        },
        "tenista": {
            "id": sol.tenista.id if sol.tenista else None,
            "nombre": getattr(sol.tenista, "nombre", None),
            "apellido": getattr(sol.tenista, "apellido", None),
            "correo": getattr(sol.tenista, "correo", None),
            "numero": getattr(sol.tenista, "numero", None),
        },
    }

@api_view(["GET"])
def solicitud_detail(request, pk: int):
    from . import models
    sol = (models.Solicitud.objects
           .select_related("tenista", "origen", "destino")
           .filter(pk=pk).first())
    if not sol:
        return Response({"ok": False, "error": "No encontrada"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"ok": True, "solicitud": _serialize_solicitud(sol)}, status=200)


from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(["GET"])
def tenista_por_numero(request, numero=None):
    from . import models
    numero = numero or request.GET.get("numero") or request.GET.get("from_phone")
    if not numero:
        return Response({"ok": False, "error": "numero requerido"}, status=400)

    t = models.Tenista.objects.filter(numero=numero).first()
    if not t:
        return Response({"ok": True, "tenista": None}, status=200)

    return Response({"ok": True, "tenista": {
        "id": t.id, "nombre": t.nombre, "apellido": t.apellido,
        "correo": t.correo, "numero": t.numero
    }})