# app/admin.py
from django.contrib import admin
from .models import Tenista, Origen, Destino, Solicitud, Reserva, Coordinador, Conductor

admin.site.register(Tenista)
admin.site.register(Origen)
admin.site.register(Destino)
admin.site.register(Solicitud)
admin.site.register(Reserva)
admin.site.register(Coordinador)
admin.site.register(Conductor)
