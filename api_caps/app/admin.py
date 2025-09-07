# app/admin.py
from django.contrib import admin
from .models import Tenista, Origen, Destino, Solicitud, Reserva, Coordinador, Conductor
from django.contrib import admin
from django import forms
from django.contrib.auth.hashers import make_password
from .models import Coordinador
from django.contrib import admin
from app.models import CoordinadorToken

admin.site.register(Tenista)
admin.site.register(Origen)
admin.site.register(Destino)
admin.site.register(Solicitud)
admin.site.register(Reserva)
admin.site.register(Conductor)

class CoordinadorAdminForm(forms.ModelForm):
    # Campo para escribir la clave en el admin
    password = forms.CharField(
        label="Password",
        required=False,
        widget=forms.PasswordInput,
        help_text="Se guardará hasheada. Déjala vacía para no cambiarla."
    )

    class Meta:
        model = Coordinador
        fields = ("nombre", "correo", "password", "created_at")

    def save(self, commit=True):
        obj = super().save(commit=False)
        pwd = self.cleaned_data.get("password")
        if pwd:
            obj.password_hash = make_password(pwd)
        if commit:
            obj.save()
        return obj

@admin.register(Coordinador)
class CoordinadorAdmin(admin.ModelAdmin):
    form = CoordinadorAdminForm
    list_display = ("nombre", "correo")
    # Si quieres ocultar password_hash del formulario:
    # exclude = ("password_hash",)


class CoordinadorTokenAdmin(admin.ModelAdmin):
    list_display = ("id", "coordinador", "key", "is_active", "expires_at", "created_at")
    search_fields = ("coordinador__correo", "key")