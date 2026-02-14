from django import forms
from django.core.exceptions import ValidationError

class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result
    
class FileFieldForm(forms.Form):
    text_prompt = forms.CharField(widget=forms.Textarea,required = False)
    file_field = MultipleFileField(required=False, label='Attachments')

    def clean_file_field(self):
        files = self.cleaned_data.get('file_field', [])
        
        max_size = 50 * 1024 * 1024  
        for file in files:
            if file.size > max_size:
                raise ValidationError(
                    f'File {file.name} is too large. Maximum size is 50MB.'
                )
        
        total_size = sum(file.size for file in files)
        max_total_size = 200 * 1024 * 1024  
        if total_size > max_total_size:
            raise ValidationError(
                f'Total upload size exceeds 200MB limit.'
            )
        
        return files

    def clean(self):
        cleaned_data = super().clean()
        text_prompt = cleaned_data.get('text_prompt', '')
        file_field = cleaned_data.get('file_field', [])
        
        if not text_prompt and not file_field:
            raise ValidationError(
                'Please provide either a message or upload at least one file.'
            )
        
        return cleaned_data