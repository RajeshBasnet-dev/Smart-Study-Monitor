from django import forms


class PredictionForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'placeholder': 'Paste a news article...'}),
        required=False,
        max_length=20000,
    )
    text_file = forms.FileField(required=False)

    def clean(self):
        cleaned_data = super().clean()
        text = cleaned_data.get('text')
        text_file = cleaned_data.get('text_file')

        if not text and not text_file:
            raise forms.ValidationError('Please provide either article text or a .txt file.')

        if text_file and not text_file.name.lower().endswith('.txt'):
            raise forms.ValidationError('Only .txt files are supported for upload.')

        return cleaned_data
