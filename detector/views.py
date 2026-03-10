from django.contrib import messages
from django.shortcuts import render

from .forms import PredictionForm
from .models import PredictionLog
from .services.inference import get_predictor


def home(request):
    form = PredictionForm(request.POST or None, request.FILES or None)
    result = None

    if request.method == 'POST' and form.is_valid():
        predictor = get_predictor()
        text = form.cleaned_data.get('text')
        text_file = form.cleaned_data.get('text_file')
        try:
            if text_file:
                result = predictor.predict_from_file(text_file)
                source = 'file'
                text_length = text_file.size
            else:
                result = predictor.predict(text)
                source = 'text'
                text_length = len(text)

            PredictionLog.objects.create(
                source=source,
                text_length=text_length,
                predicted_label=result.label,
                confidence=result.confidence,
                model_used=result.model_used,
            )
        except ValueError as exc:
            messages.error(request, str(exc))

    stats = {
        'total_predictions': PredictionLog.objects.count(),
        'fake_count': PredictionLog.objects.filter(predicted_label='Fake').count(),
        'real_count': PredictionLog.objects.filter(predicted_label='Real').count(),
    }

    return render(request, 'detector/home.html', {'form': form, 'result': result, 'stats': stats})
