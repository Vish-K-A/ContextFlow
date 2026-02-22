from django.core.exceptions import ValidationError
from django import forms


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('widget', MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result


class FileFieldForm(forms.Form):
    text_prompt = forms.CharField(widget=forms.Textarea, required=False)
    file_field = MultipleFileField(
        required=False,
        label='PDF Attachments',
        widget=MultipleFileInput(attrs={'accept': '.pdf'}),
    )

    def clean_file_field(self):
        files = [
        f for f in self.cleaned_data.get('file_field', [])
        if f is not None
    ]

        for file in files:
            if not file.name.lower().endswith('.pdf'):
                raise ValidationError(
                    f'{file.name} is not a PDF file. Only PDFs are supported.'
                )

            if (
                hasattr(file, 'content_type')
                and file.content_type != 'application/pdf'
            ):
                raise ValidationError(
                    f'{file.name} does not appear to be a valid PDF.'
                )

            if file.size > 50 * 1024 * 1024:
                raise ValidationError(
                    f'{file.name} is too large. Maximum size is 50MB.'
                )

        total_size = sum(file.size for file in files)
        if total_size > 200 * 1024 * 1024:
            raise ValidationError('Total upload size exceeds 200MB limit.')

        return files

    def clean(self):
        cleaned_data = super().clean()
        text_prompt = cleaned_data.get('text_prompt', '').strip()
        file_field = cleaned_data.get('file_field', [])

        if not text_prompt and not file_field:
            raise ValidationError(
                'Please provide either a message or upload at least one PDF.'
            )

        return cleaned_data
