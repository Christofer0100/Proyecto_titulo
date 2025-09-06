from rest_framework import serializers
from .models import (
    Coordinador, Conductor, Tenista, Origen, Destino,
    Solicitud, Reserva
)

class CoordinadorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Coordinador
        fields = "__all__"


class ConductorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conductor
        fields = "__all__"


class TenistaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tenista
        fields = "__all__"


class OrigenSerializer(serializers.ModelSerializer):
    class Meta:
        model = Origen
        fields = "__all__"


class DestinoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Destino
        fields = "__all__"


# --- Solicitud ---

class SolicitudReadNestedSerializer(serializers.ModelSerializer):
    origen = OrigenSerializer(read_only=True)
    destino = DestinoSerializer(read_only=True)
    tenista = TenistaSerializer(read_only=True)

    class Meta:
        model = Solicitud
        fields = "__all__"


class SolicitudWriteSerializer(serializers.ModelSerializer):
    # Escritura por IDs
    origen_id = serializers.PrimaryKeyRelatedField(
        source='origen', queryset=Origen.objects.all(), allow_null=True, required=False
    )
    destino_id = serializers.PrimaryKeyRelatedField(
        source='destino', queryset=Destino.objects.all(), allow_null=True, required=False
    )
    tenista_id = serializers.PrimaryKeyRelatedField(
        source='tenista', queryset=Tenista.objects.all(), allow_null=True, required=False
    )

    class Meta:
        model = Solicitud
        # Permitimos setear por id y el resto de campos del snapshot
        fields = [
            "id",
            "form_nombres", "form_apellidos", "form_correo", "form_telefono",
            "pasajeros", "hora_salida", "observaciones",
            "origen_id", "destino_id", "tenista_id",
            "idioma_detectado", "raw_form",
            "estado", "created_at",
        ]


# --- Reserva ---

class ReservaReadNestedSerializer(serializers.ModelSerializer):
    solicitud = SolicitudReadNestedSerializer(read_only=True)
    coordinador = CoordinadorSerializer(read_only=True)
    conductor = ConductorSerializer(read_only=True)

    class Meta:
        model = Reserva
        fields = "__all__"


class ReservaWriteSerializer(serializers.ModelSerializer):
    solicitud_id = serializers.PrimaryKeyRelatedField(
        source='solicitud', queryset=Solicitud.objects.all()
    )
    coordinador_id = serializers.PrimaryKeyRelatedField(
        source='coordinador', queryset=Coordinador.objects.all(), allow_null=True, required=False
    )
    conductor_id = serializers.PrimaryKeyRelatedField(
        source='conductor', queryset=Conductor.objects.all(), allow_null=True, required=False
    )

    class Meta:
        model = Reserva
        fields = [
            "id",
            "solicitud_id", "coordinador_id", "conductor_id",
            "fecha_hora_agendada", "estado",
            "created_at", "updated_at",
        ]

    def validate(self, attrs):
        # (opcional) ejemplo de regla: minutos solo 00 o 30
        # fh = attrs.get("fecha_hora_agendada")
        # if fh and fh.minute not in (0, 30):
        #     raise serializers.ValidationError("La hora debe ser en punto o y media.")
        return attrs
