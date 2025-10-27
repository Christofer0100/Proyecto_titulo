# app/views.py
from django.http import JsonResponse
from django.utils import timezone
from django.db import transaction
from django.db.models import Q
from rest_framework import viewsets, generics, filters, permissions, status
from rest_framework.decorators import api_view, permission_classes
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import logging
import json
from datetime import datetime, timedelta  # <- NUEVO

from .models import (
    Coordinador, CoordinadorToken, Conductor, Tenista, Origen, Destino,
    Solicitud, Reserva, ReservaEstado
)
from .serializers import (
    CoordinadorSerializer, ConductorSerializer, TenistaSerializer,
    OrigenSerializer, DestinoSerializer,
    SolicitudReadNestedSerializer, SolicitudWriteSerializer,
    ReservaReadNestedSerializer, ReservaWriteSerializer,
    SolicitudListSerializer, ConductorListSerializer,
)

logger = logging.getLogger(__name__)


# ---------- Utilidades ----------
def _parse_iso_dt(value: str):
    """
    Convierte string ISO 8601 a datetime aware (zona actual si es naive).
    Acepta 'Z' al final como UTC.
    """
    if not value:
        return None
    try:
        s = value.strip()
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_current_timezone())
        return dt
    except Exception:
        return None


# ---------- Base ----------
class BaseViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    ordering_fields = ["id"]
    search_fields = ["id"]


# ---------- Catálogos ----------
class CoordinadorViewSet(BaseViewSet):
    queryset = Coordinador.objects.all().order_by("-id")
    serializer_class = CoordinadorSerializer
    search_fields = ["nombre", "correo"]
    ordering_fields = ["id", "created_at"]


class ConductorViewSet(BaseViewSet):
    """
    GET /api/conductores/?disponibles=1[&fecha_hora=ISO]
      -> excluye conductores ocupados (ASIGNADA, EN_CURSO) dentro de ±1h
         respecto de 'fecha_hora' o 'now()' si no se envía.
    """
    queryset = Conductor.objects.all().order_by("-id")
    serializer_class = ConductorSerializer
    search_fields = ["nombre", "apellido", "mail", "telefono", "patente"]
    ordering_fields = ["id", "created_at"]

    def get_queryset(self):
        qs = super().get_queryset()
        # Si quieres solo activos por defecto, descomenta:
        # qs = qs.filter(activo=True)

        disponibles = self.request.query_params.get("disponibles")
        if str(disponibles) == "1":
            fecha_param = self.request.query_params.get("fecha_hora")
            fecha = _parse_iso_dt(fecha_param) or timezone.now()

            window_start = fecha - timedelta(hours=1)
            window_end = fecha + timedelta(hours=1)

            ocupados_ids = (
                Reserva.objects.filter(
                    estado__in=[ReservaEstado.ASIGNADA, ReservaEstado.EN_CURSO],
                    fecha_hora_agendada__range=(window_start, window_end),
                ).values_list("conductor_id", flat=True)
            )
            qs = qs.exclude(id__in=ocupados_ids)

        return qs


class TenistaViewSet(BaseViewSet):
    queryset = Tenista.objects.all().order_by("-id")
    serializer_class = TenistaSerializer
    search_fields = ["nombre", "apellido", "numero", "correo"]


class OrigenViewSet(BaseViewSet):
    queryset = Origen.objects.all().order_by("salida")
    serializer_class = OrigenSerializer
    search_fields = ["salida"]
    ordering_fields = ["id"]


class DestinoViewSet(BaseViewSet):
    queryset = Destino.objects.all().order_by("lugar")
    serializer_class = DestinoSerializer
    search_fields = ["lugar"]
    ordering_fields = ["id"]


# ---------- Solicitudes ----------
class SolicitudViewSet(BaseViewSet):
    queryset = Solicitud.objects.select_related("origen", "destino", "tenista").order_by("-id")
    search_fields = ["form_telefono", "form_correo", "form_nombres", "form_apellidos", "estado"]
    ordering_fields = ["id", "created_at"]

    def get_serializer_class(self):
        if getattr(self, "action", None) in ["list", "retrieve"]:
            return SolicitudReadNestedSerializer
        return SolicitudWriteSerializer


# ---------- Reservas ----------
class ReservaViewSet(viewsets.ModelViewSet):
    """
    Soporta filtros por:
      - ?conductor=<ID>
      - ?solicitud=<ID>  (o ?solicitud_id=<ID>)
      - ?estado=ASIGNADA|EN_CURSO|...
    """
    queryset = Reserva.objects.select_related("solicitud", "coordinador", "conductor").order_by("-id")
    search_fields = ["estado", "conductor__nombre", "conductor__apellido", "solicitud__form__telefono"]
    ordering_fields = ["id", "fecha_hora_agendada", "created_at", "updated_at"]

    def get_serializer_class(self):
        if getattr(self, "action", None) in ["list", "retrieve"]:
            return ReservaReadNestedSerializer
        return ReservaWriteSerializer

    def get_queryset(self):
        qs = super().get_queryset()

        conductor_id = self.request.query_params.get("conductor")
        if conductor_id:
            qs = qs.filter(conductor__id=conductor_id)

        sid = self.request.query_params.get("solicitud_id") or self.request.query_params.get("solicitud")
        if sid:
            qs = qs.filter(solicitud_id=sid)

        estado = self.request.query_params.get("estado")
        if estado:
            qs = qs.filter(estado=estado)

        return qs


# ---------- List APIs (si las usas en UI) ----------
class SolicitudListAPI(generics.ListAPIView):
    queryset = Solicitud.objects.select_related("tenista", "origen", "destino").order_by("-created_at", "-id")
    serializer_class = SolicitudListSerializer
    permission_classes = [permissions.AllowAny]


class ConductorListAPI(generics.ListAPIView):
    queryset = Conductor.objects.all().order_by("-id")
    serializer_class = ConductorListSerializer
    permission_classes = [permissions.AllowAny]


# ---------- Auth / Tokens ----------
def _emit_token(coord: Coordinador) -> CoordinadorToken:
    # Desactiva tokens vencidos del coordinador
    CoordinadorToken.objects.filter(
        coordinador=coord, is_active=True, expires_at__lt=timezone.now()
    ).update(is_active=False)

    from secrets import token_urlsafe
    key = token_urlsafe(32)
    t = CoordinadorToken.objects.create(
        coordinador=coord,
        key=key,
        expires_at=timezone.now() + timezone.timedelta(days=1),
        is_active=True,
    )
    return t


@api_view(['POST'])
@permission_classes([AllowAny])
def coordinador_login(request):
    try:
        email = (request.data.get('email') or '').strip().lower()
        password = request.data.get('password') or ''

        if not email or not password:
            return Response({"ok": False, "error": "email y password son requeridos"}, status=400)

        coord = Coordinador.objects.filter(correo__iexact=email).first()
        if not coord or not coord.password_hash:
            return Response({"ok": False, "error": "Credenciales inválidas"}, status=401)

        from django.contrib.auth.hashers import check_password
        if not check_password(password, coord.password_hash):
            return Response({"ok": False, "error": "Credenciales inválidas"}, status=401)

        token = _emit_token(coord)
        return Response({
            "ok": True,
            "token": token.key,
            "expires_at": token.expires_at.isoformat(),
            "coordinador": {"id": coord.id, "correo": coord.correo, "nombre": getattr(coord, "nombre", None)},
        })

    except Exception:
        logger.exception("Error en login de coordinador")
        return Response({"ok": False, "error": "Error interno"}, status=500)


# ---------- Asignación de conductor ----------
@api_view(["POST"])
@permission_classes([AllowAny])  # ajusta si usas auth
@transaction.atomic
def asignar_conductor_a_solicitud(request, pk: int):
    """
    Body esperado (JSON):
    {
      "conductor_id": 3,
      "fecha_hora_agendada": "2025-09-07T15:30:00Z",  (opcional pero recomendado)
      "coordinador_id": 1                             (opcional)
    }
    Regla: bloquear solo si el conductor tiene otra reserva ASIGNADA/EN_CURSO
           en la ventana de ±1 hora de 'fecha_hora_agendada' (o now()).
    """
    data = request.data or {}
    conductor_id = data.get("conductor_id")
    if not conductor_id:
        return Response({"ok": False, "error": "conductor_id es requerido"}, status=400)

    try:
        sol = Solicitud.objects.select_for_update().get(pk=pk)
    except Solicitud.DoesNotExist:
        return Response({"ok": False, "error": "Solicitud no encontrada"}, status=404)

    try:
        conductor = Conductor.objects.get(pk=conductor_id, activo=True)
    except Conductor.DoesNotExist:
        return Response({"ok": False, "error": "Conductor no válido o inactivo"}, status=400)

    # Determinar la fecha/hora a usar para la validación y la reserva
    fecha_param = data.get("fecha_hora_agendada")
    fecha = _parse_iso_dt(fecha_param) or timezone.now()

    # Ventana de ±1 hora
    window_start = fecha - timedelta(hours=1)
    window_end = fecha + timedelta(hours=1)

    # Bloquear solo si hay otra reserva del mismo conductor en esa ventana
    conflicto = Reserva.objects.filter(
        conductor_id=conductor.id,
        estado__in=[ReservaEstado.ASIGNADA, ReservaEstado.EN_CURSO],
        fecha_hora_agendada__range=(window_start, window_end),
    ).exists()
    if conflicto:
        return Response(
            {"ok": False, "error": "El conductor tiene otra reserva en la última/próxima hora"},
            status=409
        )

    # Crear o actualizar la reserva para esta solicitud
    reserva, _created = Reserva.objects.get_or_create(
        solicitud=sol,
        defaults={
            "fecha_hora_agendada": fecha,
            "estado": ReservaEstado.ASIGNADA,
            "created_at": timezone.now(),
            "updated_at": timezone.now(),
        },
    )
    reserva.conductor = conductor
    reserva.estado = ReservaEstado.ASIGNADA
    reserva.fecha_hora_agendada = fecha
    reserva.updated_at = timezone.now()

    coord_id = data.get("coordinador_id")
    if coord_id:
        reserva.coordinador_id = int(coord_id)

    reserva.save()

    return Response({"ok": True, "solicitud_id": sol.id, "reserva": ReservaReadNestedSerializer(reserva).data})


# ---------- Tenistas & Login conductor (custom) ----------
class TenistaListView(generics.ListAPIView):
    queryset = Tenista.objects.all().order_by("id")
    serializer_class = TenistaSerializer


@csrf_exempt
def login_conductor(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Método no permitido"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
        email = data.get("email")
        password = data.get("password")

        from django.contrib.auth.hashers import check_password
        conductor = Conductor.objects.get(mail=email)
        if not check_password(password, conductor.password_hash):
            return JsonResponse({"ok": False, "error": "Credenciales inválidas"}, status=401)

        return JsonResponse({
            "ok": True,
            "conductor": {
                "id": conductor.id,
                "nombre": conductor.nombre,
                "apellido": conductor.apellido,
                "mail": conductor.mail
            }
        })
    except Conductor.DoesNotExist:
        return JsonResponse({"ok": False, "error": "Conductor no encontrado"}, status=404)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
