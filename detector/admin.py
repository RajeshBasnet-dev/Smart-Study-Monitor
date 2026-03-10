from django.contrib import admin
from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'source', 'predicted_label', 'confidence', 'model_used', 'created_at')
    list_filter = ('predicted_label', 'model_used', 'source')
    search_fields = ('predicted_label', 'model_used')
