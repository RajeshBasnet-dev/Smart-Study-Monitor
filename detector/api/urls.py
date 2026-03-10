from django.urls import path

from .views import FilePredictAPIView, TextPredictAPIView

urlpatterns = [
    path('predict/text/', TextPredictAPIView.as_view(), name='api_predict_text'),
    path('predict/file/', FilePredictAPIView.as_view(), name='api_predict_file'),
]
