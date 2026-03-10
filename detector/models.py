from django.db import models


class PredictionLog(models.Model):
    """Stores prediction requests for analytics and auditing."""

    source = models.CharField(max_length=20, default='text')
    text_length = models.PositiveIntegerField()
    predicted_label = models.CharField(max_length=10)
    confidence = models.FloatField()
    model_used = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f"{self.predicted_label} ({self.confidence:.3f})"
