from django.shortcuts import render

# Create your views here.
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
