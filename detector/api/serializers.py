from rest_framework import serializers


class TextPredictionSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=20000)


class FilePredictionSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, value):
        if not value.name.lower().endswith('.txt'):
            raise serializers.ValidationError('Only .txt files are supported.')
        return value
