from django.db import models

# ---------- ENUMs como TextChoices (en Django)
class SolicitudEstado(models.TextChoices):
    NUEVA = "NUEVA", "NUEVA"
    EN_REVISION = "EN_REVISION", "EN_REVISION"
    RECHAZADA = "RECHAZADA", "RECHAZADA"
    CONFIRMADA = "CONFIRMADA", "CONFIRMADA"

class ReservaEstado(models.TextChoices):
    PENDIENTE = "PENDIENTE", "PENDIENTE"
    ASIGNADA = "ASIGNADA", "ASIGNADA"
    EN_CURSO = "EN_CURSO", "EN_CURSO"
    COMPLETADA = "COMPLETADA", "COMPLETADA"
    CANCELADA = "CANCELADA", "CANCELADA"


class Coordinador(models.Model):
    id = models.BigAutoField(primary_key=True)
    nombre = models.TextField()
    correo = models.TextField(unique=True)
    created_at = models.DateTimeField()

    class Meta:
        managed = True
        db_table = 'coordinador'


class Conductor(models.Model):
    id = models.BigAutoField(primary_key=True)
    nombre = models.TextField()
    apellido = models.TextField()
    patente = models.TextField(blank=True, null=True)
    mail = models.TextField(unique=True)
    telefono = models.TextField(blank=True, null=True)
    activo = models.BooleanField()
    created_at = models.DateTimeField()

    class Meta:
        managed = True
        db_table = 'conductor'


class Tenista(models.Model):
    id = models.BigAutoField(primary_key=True)
    nombre = models.TextField()
    apellido = models.TextField()
    correo = models.TextField(blank=True, null=True)
    numero = models.TextField(unique=True)

    class Meta:
        managed = True
        db_table = 'tenista'


class Origen(models.Model):
    id = models.BigAutoField(primary_key=True)
    salida = models.TextField(unique=True)

    class Meta:
        managed = True
        db_table = 'origen'


class Destino(models.Model):
    id = models.BigAutoField(primary_key=True)
    lugar = models.TextField(unique=True)

    class Meta:
        managed = True
        db_table = 'destino'


class Solicitud(models.Model):
    id = models.BigAutoField(primary_key=True)
    form_nombres = models.TextField()
    form_apellidos = models.TextField()
    form_correo = models.TextField(blank=True, null=True)
    form_telefono = models.TextField()
    pasajeros = models.SmallIntegerField()
    hora_salida = models.TimeField(blank=True, null=True)
    observaciones = models.TextField(blank=True, null=True)

    origen = models.ForeignKey(Origen, models.DO_NOTHING, db_column='origen_id', blank=True, null=True, related_name='solicitudes_origen')
    destino = models.ForeignKey(Destino, models.DO_NOTHING, db_column='destino_id', blank=True, null=True, related_name='solicitudes_destino')
    tenista = models.ForeignKey(Tenista, models.DO_NOTHING, db_column='tenista_id', blank=True, null=True, related_name='solicitudes')

    idioma_detectado = models.CharField(max_length=8, blank=True, null=True)
    raw_form = models.JSONField(blank=True, null=True)

    estado = models.CharField(max_length=20, choices=SolicitudEstado.choices, default=SolicitudEstado.NUEVA)
    created_at = models.DateTimeField()

    class Meta:
        managed = True
        db_table = 'solicitud'


class Reserva(models.Model):
    id = models.BigAutoField(primary_key=True)
    solicitud = models.OneToOneField(Solicitud, models.CASCADE, db_column='solicitud_id', related_name='reserva')
    coordinador = models.ForeignKey(Coordinador, models.DO_NOTHING, db_column='coordinador_id', blank=True, null=True, related_name='reservas')
    conductor = models.ForeignKey(Conductor, models.DO_NOTHING, db_column='conductor_id', blank=True, null=True, related_name='reservas')

    fecha_hora_agendada = models.DateTimeField()
    estado = models.CharField(max_length=20, choices=ReservaEstado.choices, default=ReservaEstado.PENDIENTE)

    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        managed = True
        db_table = 'reserva'
