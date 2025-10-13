from django.contrib import admin
from django import forms
from django.contrib.auth.hashers import make_password
from .models import (
    Tenista, Origen, Destino, Solicitud, Reserva,
    Coordinador, Conductor, CoordinadorToken
)

# --- Modelos básicos ---
admin.site.register(Tenista)
admin.site.register(Origen)
admin.site.register(Destino)
admin.site.register(Solicitud)
admin.site.register(Reserva)

# --- Coordinador ---
class CoordinadorAdminForm(forms.ModelForm):
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


@admin.register(CoordinadorToken)
class CoordinadorTokenAdmin(admin.ModelAdmin):
    list_display = ("id", "coordinador", "key", "is_active", "expires_at", "created_at")
    search_fields = ("coordinador__correo", "key")


# --- Conductor ---
@admin.register(Conductor)
class ConductorAdmin(admin.ModelAdmin):
    list_display = ("nombre", "apellido", "mail", "activo")

    def save_model(self, request, obj, form, change):
        raw_password = form.cleaned_data.get("password_hash")
        if raw_password and not str(raw_password).startswith("pbkdf2_"):
            obj.password_hash = make_password(raw_password)
        super().save_model(request, obj, form, change)
