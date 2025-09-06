# app/models.py
from django.db import models
from django.utils import timezone


# ---------- ENUMs como TextChoices (en Django)
class SolicitudEstado(models.TextChoices):
    NUEVA       = "NUEVA",       "NUEVA"
    EN_REVISION = "EN_REVISION", "EN_REVISION"
    RECHAZADA   = "RECHAZADA",   "RECHAZADA"
    CONFIRMADA  = "CONFIRMADA",  "CONFIRMADA"


class ReservaEstado(models.TextChoices):
    PENDIENTE  = "PENDIENTE",  "PENDIENTE"
    ASIGNADA   = "ASIGNADA",   "ASIGNADA"
    EN_CURSO   = "EN_CURSO",   "EN_CURSO"
    COMPLETADA = "COMPLETADA", "COMPLETADA"
    CANCELADA  = "CANCELADA",  "CANCELADA"


# ---------- MODELOS
class Coordinador(models.Model):
    id         = models.BigAutoField(primary_key=True)
    nombre     = models.TextField()
    correo     = models.TextField(unique=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        managed  = True
        db_table = "coordinador"
        ordering = ("-id",)

    def __str__(self):
        return f"{self.nombre} <{self.correo}>"


class Conductor(models.Model):
    id         = models.BigAutoField(primary_key=True)
    nombre     = models.TextField()
    apellido   = models.TextField()
    patente    = models.TextField(blank=True, null=True)
    mail       = models.TextField(unique=True)
    telefono   = models.TextField(blank=True, null=True)
    activo     = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        managed  = True
        db_table = "conductor"
        ordering = ("-id",)

    def __str__(self):
        return f"{self.nombre} {self.apellido} ({self.patente or 'sin patente'})"


class Tenista(models.Model):
    id       = models.BigAutoField(primary_key=True)
    nombre   = models.TextField()
    apellido = models.TextField()
    correo   = models.TextField(blank=True, null=True)
    numero   = models.TextField(unique=True)

    class Meta:
        managed  = True
        db_table = "tenista"
        indexes  = [models.Index(fields=["numero"], name="tenista_numero_idx")]
        ordering = ("-id",)

    def __str__(self):
        return f"{self.nombre} {self.apellido} ({self.numero})"


class Origen(models.Model):
    id     = models.BigAutoField(primary_key=True)
    salida = models.TextField(unique=True)

    class Meta:
        managed  = True
        db_table = "origen"
        ordering = ("salida",)

    def __str__(self):
        return self.salida


class Destino(models.Model):
    id    = models.BigAutoField(primary_key=True)
    lugar = models.TextField(unique=True)

    class Meta:
        managed  = True
        db_table = "destino"
        ordering = ("lugar",)

    def __str__(self):
        return self.lugar


class Solicitud(models.Model):
    id             = models.BigAutoField(primary_key=True)

    # Datos ingresados por el formulario (lo que verás rápido en admin)
    form_nombres   = models.TextField()
    form_apellidos = models.TextField()
    form_correo    = models.TextField(blank=True, null=True)
    form_telefono  = models.TextField()

    pasajeros      = models.SmallIntegerField()
    hora_salida    = models.TimeField(blank=True, null=True)
    observaciones  = models.TextField(blank=True, null=True)

    # Relaciones
    origen  = models.ForeignKey(
        Origen, models.DO_NOTHING,
        db_column="origen_id", blank=True, null=True,
        related_name="solicitudes_origen",
    )
    destino = models.ForeignKey(
        Destino, models.DO_NOTHING,
        db_column="destino_id", blank=True, null=True,
        related_name="solicitudes_destino",
    )
    tenista = models.ForeignKey(
        Tenista, models.DO_NOTHING,
        db_column="tenista_id", blank=True, null=True,
        related_name="solicitudes",
    )

    # Auxiliares
    idioma_detectado = models.CharField(max_length=8, blank=True, null=True)
    raw_form         = models.JSONField(blank=True, null=True)

    # Estado y tiempos
    estado     = models.CharField(
        max_length=20, choices=SolicitudEstado.choices, default=SolicitudEstado.NUEVA
    )
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        managed  = True
        db_table = "solicitud"
        indexes  = [
            models.Index(fields=["created_at"], name="solicitud_created_idx"),
            models.Index(fields=["estado"],     name="solicitud_estado_idx"),
        ]
        ordering = ("-created_at", "-id")

    def __str__(self):
        base = f"Solicitud #{self.id} ({self.estado})"
        quien = f"{self.form_nombres} {self.form_apellidos}".strip()
        return f"{base} - {quien or 'sin nombre'}"


class Reserva(models.Model):
    id        = models.BigAutoField(primary_key=True)

    solicitud   = models.OneToOneField(
        Solicitud, models.CASCADE,
        db_column="solicitud_id", related_name="reserva",
    )
    coordinador = models.ForeignKey(
        Coordinador, models.DO_NOTHING,
        db_column="coordinador_id", blank=True, null=True,
        related_name="reservas",
    )
    conductor   = models.ForeignKey(
        Conductor, models.DO_NOTHING,
        db_column="conductor_id", blank=True, null=True,
        related_name="reservas",
    )

    fecha_hora_agendada = models.DateTimeField()
    estado              = models.CharField(
        max_length=20, choices=ReservaEstado.choices, default=ReservaEstado.PENDIENTE
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed  = True
        db_table = "reserva"
        ordering = ("-fecha_hora_agendada", "-id")

    def __str__(self):
        return f"Reserva #{self.id} - {self.estado}"
