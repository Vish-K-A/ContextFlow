import os
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import FileFieldForm
from .models import Message, Attachment, Conversation
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

        conversation = Conversation.objects.create(
            user=self.request.user if self.request.user.is_authenticated else None
        )

        user_message = Message.objects.create(
            conversation=conversation,
            text=prompt,
            is_ai_response=False,
            user=self.request.user if self.request.user.is_authenticated else None
        )

        attachment_paths = []
        for f in files:
            attachment = Attachment.objects.create(
                message=user_message,
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
                        "chat_history": []
                    })

                    ai_answer = response['answer']

                    Message.objects.create(
                        conversation=conversation,
                        text=ai_answer,
                        is_ai_response=True,
                        user=None
                    )

                    conversation.last_pdf_path = pdf_paths[0]
                    conversation.save()

                    self.request.session['last_retriever'] = pdf_paths[0]
                    self.request.session['chat_history'] = [[prompt, ai_answer]]

            except Exception as e:
                print(f"RAG processing error: {e}")
                Message.objects.create(
                    conversation=conversation,
                    text=f"Error processing document: {str(e)}",
                    is_ai_response=True,
                    user=None
                )

        self.request.session['active_conversation_id'] = conversation.id
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        active_conv_id = self.request.session.get('active_conversation_id')
        if active_conv_id:
            try:
                conversation = Conversation.objects.get(id=active_conv_id)
                context['messages'] = conversation.messages.all().order_by('created_at')
                context['conversation'] = conversation
            except Conversation.DoesNotExist:
                context['messages'] = []
        else:
            context['messages'] = []
        return context


@login_required
def user_file_views(request):
    user_files = Attachment.objects.filter(user=request.user).select_related('message')
    context = {
        'files': user_files,
        'total_files': user_files.count()
    }
    return render(request, 'documents/files.html', context)


@login_required
def chat_followup(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')

        if not question:
            return JsonResponse({'error': 'No question provided'}, status=400)

        try:
            last_pdf = request.session.get('last_retriever')
            raw_history = request.session.get('chat_history', [])
            chat_history = [tuple(pair) for pair in raw_history]
            active_conv_id = request.session.get('active_conversation_id')

            if not last_pdf:
                return JsonResponse({'error': 'No document context found. Please upload a PDF first.'}, status=400)

            retriever = setup_vectorstore(last_pdf)
            chain = get_advanced_chain(retriever)

            response = chain({
                "question": question,
                "chat_history": chat_history
            })

            answer = response['answer']
            chat_history.append((question, answer))
            request.session['chat_history'] = [list(pair) for pair in chat_history]

            if active_conv_id:
                try:
                    conversation = Conversation.objects.get(id=active_conv_id)
                    Message.objects.create(
                        conversation=conversation,
                        text=question,
                        is_ai_response=False,
                        user=request.user
                    )
                    Message.objects.create(
                        conversation=conversation,
                        text=answer,
                        is_ai_response=True,
                        user=None
                    )
                except Conversation.DoesNotExist:
                    pass

            return JsonResponse({
                'answer': answer,
                'sources': [doc.page_content[:200] for doc in response.get('source_documents', [])]
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid method'}, status=405)