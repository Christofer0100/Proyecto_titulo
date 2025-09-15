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
from django.urls import path
from app.views import SolicitudListAPI, ConductorListAPI
from django.urls import path
from app.views import coordinador_login
from django.urls import path
from app.views import (
    ConductoresListView,
    asignar_conductor_a_solicitud,
)
from .views import ReservaViewSet
from .views import ReservaListView
from .views import TenistaListView

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
    path("solicitudes/", SolicitudListAPI.as_view()),
    path("conductores/", ConductorListAPI.as_view()),
    path("auth/coordinador/login/", coordinador_login),
    path("solicitudes/<int:pk>/asignar/", asignar_conductor_a_solicitud, name="solicitud-asignar"),
    path("conductores/", ConductoresListView.as_view(), name="conductores-list"),
   path("reservas/", ReservaListView.as_view(), name="reserva-list"),
   path("tenistas/", TenistaListView.as_view(), name="tenista-list"),
]

