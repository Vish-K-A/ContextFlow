from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import FileFieldForm
from .models import Message, Attachment
from django.views.generic.edit import FormView

class ChatUploadView(FormView):
    form_class = FileFieldForm
    template_name = "documents/document.html"
    success_url = "/documents/"

    def form_valid(self, form):
        prompt = form.cleaned_data.get('text_prompt', '')
        files = form.cleaned_data.get('file_field', [])

        message_instance = Message.objects.create(
            text=prompt,
            user=self.request.user if self.request.user.is_authenticated else None
        )

        for f in files:
            Attachment.objects.create(
                message=message_instance,
                user=self.request.user if self.request.user.is_authenticated else None,
                file=f
            )

        return super().form_valid(form)

@login_required    
def user_file_views(request):
    user_files = Attachment.objects.filter(user=request.user).select_related('message')
    context = {
        'files': user_files,
        'total_files': user_files.count()
    }
    return render(request, 'documents/files.html', context)