import os
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import FileFieldForm
from .models import Message, Attachment
from django.views.generic.edit import FormView
from django.http import JsonResponse
from .utils import setup_vectorstore, get_advanced_chain

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

        attachment_paths = []

        for f in files:
            attachment = Attachment.objects.create(
                message=message_instance,
                user=self.request.user if self.request.user.is_authenticated else None,
                file=f
            )
            attachment_paths.append(attachment.file.path)

        if prompt and attachment_paths:
            try:
                pdf_paths = [path for path in attachment_paths if path.lower().endswith('.pdf')]

                if pdf_paths:
                    retriever = setup_vectorstore(pdf_paths[0])
                    chain = get_advanced_chain(retriever)

                    response = chain({
                        "question": prompt,
                        "chat_history":[]
                    })

                    ai_message = Message.objects.create(
                        text=response['answer'],
                        user=None
                    )
                    
                    self.request.session['last_retriever'] = pdf_paths[0]
                    self.request.session['chat_history'] = [
                        (prompt, response['answer'])
                    ]
            
            except Exception as e:
                print(f"RAG processing error: {e}")


        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['messages'] = Message.objects.filter(
            user=self.request.user if self.request.user.is_authenticated else None
        )[:25]
        return context

@login_required    
def user_file_views(request):
    user_files = Attachment.objects.filter(user=request.user).select_related('message')
    context = {
        'files': user_files,
        'total_files': user_files.count()
    }
    return render(request, 'documents/files.html', context)

