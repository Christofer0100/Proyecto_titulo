from django.shortcuts import render
# Create your views here.
from rest_framework import generics, permissions
from .models import Solicitud, Conductor
from .serializers import SolicitudListSerializer, ConductorListSerializer
from rest_framework import viewsets, filters
from rest_framework.permissions import AllowAny
from .models import (
    Coordinador, Conductor, Tenista, Origen, Destino,
    Solicitud, Reserva
)
from .serializers import (
    CoordinadorSerializer, ConductorSerializer, TenistaSerializer,
    OrigenSerializer, DestinoSerializer,
    SolicitudReadNestedSerializer, SolicitudWriteSerializer,
    ReservaReadNestedSerializer, ReservaWriteSerializer
)
from app.models import Coordinador, CoordinadorToken, Solicitud, Tenista, Origen, Destino
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.hashers import check_password, make_password
from django.utils import timezone
from django.db import models
import secrets
from datetime import timedelta

from .models import Coordinador
# app/views.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from django.contrib.auth.hashers import check_password
import logging

from app.models import Coordinador, CoordinadorToken


class BaseViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    ordering_fields = ["id"]
    search_fields = ["id"]


class CoordinadorViewSet(BaseViewSet):
    queryset = Coordinador.objects.all().order_by("-id")
    serializer_class = CoordinadorSerializer
    search_fields = ["nombre", "correo"]
    ordering_fields = ["id", "created_at"]


class ConductorViewSet(BaseViewSet):
    queryset = Conductor.objects.all().order_by("-id")
    serializer_class = ConductorSerializer
    search_fields = ["nombre", "apellido", "mail", "telefono", "patente"]
    ordering_fields = ["id", "created_at"]


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


class SolicitudViewSet(BaseViewSet):
    queryset = Solicitud.objects.select_related("origen", "destino", "tenista").order_by("-id")
    search_fields = ["form_telefono", "form_correo", "form_nombres", "form_apellidos", "estado"]
    ordering_fields = ["id", "created_at"]

    def get_serializer_class(self):
        if self.action in ["list", "retrieve"]:
            return SolicitudReadNestedSerializer
        return SolicitudWriteSerializer


class ReservaViewSet(BaseViewSet):
    queryset = Reserva.objects.select_related("solicitud", "coordinador", "conductor").order_by("-id")
    search_fields = ["estado", "conductor__nombre", "conductor__apellido", "solicitud__form_telefono"]
    ordering_fields = ["id", "fecha_hora_agendada", "created_at", "updated_at"]

    def get_serializer_class(self):
        if self.action in ["list", "retrieve"]:
            return ReservaReadNestedSerializer
        return ReservaWriteSerializer


class SolicitudListAPI(generics.ListAPIView):
    queryset = Solicitud.objects.select_related("tenista", "origen", "destino").order_by("-created_at", "-id")
    serializer_class = SolicitudListSerializer
    permission_classes = [permissions.AllowAny]  # o cambia a IsAuthenticated

class ConductorListAPI(generics.ListAPIView):
    queryset = Conductor.objects.all().order_by("-id")
    serializer_class = ConductorListSerializer
    permission_classes = [permissions.AllowAny]


logger = logging.getLogger(__name__)

def _emit_token(coord):
    # elimina tokens viejos si quieres, o reusa el más reciente válido
    CoordinadorToken.objects.filter(coordinador=coord, is_active=True, expires_at__lt=timezone.now()).update(is_active=False)

    # crea token nuevo
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
            return Response(
                {"ok": False, "error": "email y password son requeridos"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        coord = Coordinador.objects.filter(correo__iexact=email).first()
        if not coord:
            return Response(
                {"ok": False, "error": "Coordinador no encontrado"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        if not coord.password_hash:
            return Response(
                {"ok": False, "error": "El coordinador no tiene contraseña definida"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        if not check_password(password, coord.password_hash):
            return Response(
                {"ok": False, "error": "Credenciales inválidas"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        token = _emit_token(coord)
        return Response(
            {
                "ok": True,
                "token": token.key,
                "expires_at": token.expires_at.isoformat(),
                "coordinador": {
                    "id": coord.id,
                    "correo": coord.correo,
                    "nombre": getattr(coord, "nombre", None),
                },
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.exception("Error en login de coordinador")
        return Response(
            {"ok": False, "error": "Error interno"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )