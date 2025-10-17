# app/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from app.webhooks import whatsapp_webhook, solicitud_detail, tenista_por_numero
from app.views import (
    CoordinadorViewSet, ConductorViewSet, TenistaViewSet,
    OrigenViewSet, DestinoViewSet, SolicitudViewSet, ReservaViewSet,
    coordinador_login, login_conductor, asignar_conductor_a_solicitud,
)

router = DefaultRouter()
router.register(r'coordinadores', CoordinadorViewSet, basename='coordinador')
router.register(r'conductores',   ConductorViewSet,   basename='conductor')
router.register(r'tenistas',      TenistaViewSet,     basename='tenista')
router.register(r'origenes',      OrigenViewSet,      basename='origen')
router.register(r'destinos',      DestinoViewSet,     basename='destino')
router.register(r'solicitudes',   SolicitudViewSet,   basename='solicitud')
router.register(r'reservas',      ReservaViewSet,     basename='reserva')

urlpatterns = [
    path('', include(router.urls)),

    # Webhooks / utilidades
    path("webhooks/whatsapp/", whatsapp_webhook, name="whatsapp_webhook"),
    path("solicitudes/<int:pk>/", solicitud_detail, name="solicitud-detail"),
    path("api/tenistas/por-numero/", tenista_por_numero),
    path("api/tenistas/por-numero/<path:numero>/", tenista_por_numero),

    # Acciones personalizadas
    path("solicitudes/<int:pk>/asignar/", asignar_conductor_a_solicitud, name="solicitud-asignar"),

    # Auth
    path("auth/coordinador/login/", coordinador_login, name="coordinador-login"),
    path("auth/conductor/login/",   login_conductor,   name="conductor-login"),
]
