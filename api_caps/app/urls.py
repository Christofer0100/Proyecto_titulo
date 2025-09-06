# app/urls.py
from django.urls import path, include
from django.urls import path
from app.webhooks import whatsapp_webhook
from rest_framework.routers import DefaultRouter
from app.views import (
    CoordinadorViewSet, ConductorViewSet, TenistaViewSet,
    OrigenViewSet, DestinoViewSet, SolicitudViewSet, ReservaViewSet
)
from app.webhooks import (
    solicitud_detail,
   
)
from app.webhooks import tenista_por_numero

router = DefaultRouter()
router.register(r'coordinadores', CoordinadorViewSet, basename='coordinador')
router.register(r'conductores', ConductorViewSet, basename='conductor')
router.register(r'tenistas', TenistaViewSet, basename='tenista')
router.register(r'origenes', OrigenViewSet, basename='origen')
router.register(r'destinos', DestinoViewSet, basename='destino')
router.register(r'solicitudes', SolicitudViewSet, basename='solicitud')
router.register(r'reservas',   ReservaViewSet,   basename='reserva')

urlpatterns = [
    path('api/', include(router.urls)),
    path("webhooks/whatsapp/", whatsapp_webhook, name="whatsapp_webhook"),
    path("solicitudes/<int:pk>/", solicitud_detail),
    path("api/tenistas/por-numero/", tenista_por_numero), 
    path("api/tenistas/por-numero/<path:numero>/", tenista_por_numero), 
]

