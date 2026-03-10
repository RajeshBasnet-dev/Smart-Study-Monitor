from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from detector.models import PredictionLog
from detector.services.inference import get_predictor
from .serializers import FilePredictionSerializer, TextPredictionSerializer


class TextPredictAPIView(APIView):
    def post(self, request):
        serializer = TextPredictionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        predictor = get_predictor()
        result = predictor.predict(serializer.validated_data['text'])
        PredictionLog.objects.create(
            source='api_text',
            text_length=len(serializer.validated_data['text']),
            predicted_label=result.label,
            confidence=result.confidence,
            model_used=result.model_used,
        )
        return Response(result.__dict__, status=status.HTTP_200_OK)


class FilePredictAPIView(APIView):
    def post(self, request):
        serializer = FilePredictionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        predictor = get_predictor()
        file_obj = serializer.validated_data['file']
        result = predictor.predict_from_file(file_obj)
        PredictionLog.objects.create(
            source='api_file',
            text_length=file_obj.size,
            predicted_label=result.label,
            confidence=result.confidence,
            model_used=result.model_used,
        )
        return Response(result.__dict__, status=status.HTTP_200_OK)
