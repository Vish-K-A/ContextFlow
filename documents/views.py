import os
import re
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.generic.edit import FormView

from .forms import FileFieldForm
from .models import Message, Attachment, Conversation
from .utils import setup_vectorstore, get_advanced_chain, answer_without_pdf


def _build_title(prompt, files):
    if prompt:
        return prompt[:50]
    if files:
        raw_name = getattr(files[0], 'name', '') or ''
        base_name = raw_name.replace('\\', '/').split('/')[-1]
        clean = re.sub(r'_[A-Za-z0-9]{6,8}(?=[.][^.]+$)', '', base_name)
        return (clean or base_name)[:50] or 'Uploaded document'
    return 'New conversation'


def _pdf_basename(path):
    if not path:
        return None
    return os.path.basename(path)


class ChatUploadView(FormView):
    form_class = FileFieldForm
    template_name = 'documents/document.html'
    success_url = '/documents/'

    def form_valid(self, form):
        prompt = form.cleaned_data.get('text_prompt', '').strip()
        files = form.cleaned_data.get('file_field', [])

        conversation = Conversation.objects.create(
            user=(
                self.request.user
                if self.request.user.is_authenticated else None
            ),
            title=_build_title(prompt, files),
        )

        user_message = Message.objects.create(
            conversation=conversation,
            text=prompt,
            is_ai_response=False,
            user=(
                self.request.user
                if self.request.user.is_authenticated else None
            ),
        )

        attachment_paths = []
        for file in files:
            attachment = Attachment.objects.create(
                message=user_message,
                user=(
                    self.request.user
                    if self.request.user.is_authenticated
                    else None
                ),
                file=file,
            )
            attachment_paths.append(attachment.file.path)

        ai_answer = None

        if prompt:
            try:
                pdf_paths = [
                    p for p in attachment_paths if p.lower().endswith('.pdf')
                ]

                if pdf_paths:
                    retriever = setup_vectorstore(pdf_paths[0])
                    chain = get_advanced_chain(retriever)
                    response = chain.invoke(
                        {'question': prompt, 'chat_history': []}
                    )
                    ai_answer = response['answer']
                    conversation.last_pdf_path = pdf_paths[0]
                    conversation.save()
                    self.request.session['last_retriever'] = pdf_paths[0]
                    self.request.session['last_retriever_name'] = _pdf_basename(pdf_paths[0])
                else:
                    ai_answer = answer_without_pdf(prompt, [])

                self.request.session['chat_history'] = [[prompt, ai_answer]]

                Message.objects.create(
                    conversation=conversation,
                    text=ai_answer,
                    is_ai_response=True,
                    user=None,
                )

            except Exception as exc:
                print(f'Processing error: {exc}')
                ai_answer = f'Error: {str(exc)}'
                Message.objects.create(
                    conversation=conversation,
                    text=ai_answer,
                    is_ai_response=True,
                    user=None,
                )

        elif attachment_paths:
            pdf_paths = [
                p for p in attachment_paths if p.lower().endswith('.pdf')
            ]
            if pdf_paths:
                conversation.last_pdf_path = pdf_paths[0]
                conversation.save()
                self.request.session['last_retriever'] = pdf_paths[0]
                self.request.session['last_retriever_name'] = _pdf_basename(pdf_paths[0])

        self.request.session['active_conversation_id'] = conversation.id

        return JsonResponse({
            'answer': ai_answer or 'Document uploaded.',
            'conversation_id': conversation.id,
        })

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        active_conv_id = self.request.session.get('active_conversation_id')

        if self.request.user.is_authenticated:
            context['conversations'] = Conversation.objects.filter(
                user=self.request.user
            ).order_by('-created_at')[:20]

        if active_conv_id:
            try:
                conversation = Conversation.objects.get(id=active_conv_id)
                context['messages'] = (
                    conversation.messages.all().order_by('created_at')
                )
                context['conversation'] = conversation
            except Conversation.DoesNotExist:
                context['messages'] = []
        else:
            context['messages'] = []

        last_pdf = self.request.session.get('last_retriever')
        last_pdf_name = self.request.session.get('last_retriever_name')
        if last_pdf and os.path.exists(last_pdf):
            context['active_pdf_name'] = last_pdf_name or _pdf_basename(last_pdf)
        else:
            context['active_pdf_name'] = None

        return context


@login_required
def load_conversation(request, conversation_id):
    try:
        conversation = Conversation.objects.get(
            id=conversation_id, user=request.user
        )

        request.session['active_conversation_id'] = conversation.id

        if conversation.last_pdf_path:
            request.session['last_retriever'] = conversation.last_pdf_path
            request.session['last_retriever_name'] = _pdf_basename(conversation.last_pdf_path)
        else:
            request.session.pop('last_retriever', None)
            request.session.pop('last_retriever_name', None)

        chat_history = []
        user_messages = conversation.messages.filter(
            is_ai_response=False
        ).order_by('created_at')

        for msg in user_messages:
            ai_response = conversation.messages.filter(
                is_ai_response=True,
                created_at__gt=msg.created_at,
            ).first()
            if ai_response:
                chat_history.append([msg.text, ai_response.text])

        request.session['chat_history'] = chat_history
        return redirect('documents')

    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found'}, status=404)


@login_required
def delete_conversation(request, conversation_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        conversation = Conversation.objects.get(
            id=conversation_id, user=request.user
        )
        conversation.delete()

        if request.session.get('active_conversation_id') == conversation_id:
            request.session.pop('active_conversation_id', None)
            request.session.pop('last_retriever', None)
            request.session.pop('last_retriever_name', None)
            request.session.pop('chat_history', None)

        return JsonResponse({'success': True})

    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found'}, status=404)


@login_required
def new_conversation(request):
    request.session.pop('active_conversation_id', None)
    request.session.pop('last_retriever', None)
    request.session.pop('last_retriever_name', None)
    request.session.pop('chat_history', None)
    return redirect('documents')


@login_required
def user_file_views(request):
    user_files = Attachment.objects.filter(
        user=request.user
    ).select_related('message')
    context = {
        'files': user_files,
        'total_files': user_files.count(),
    }
    return render(request, 'documents/files.html', context)


def chat_followup(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    question = request.POST.get('question', '').strip()
    if not question:
        return JsonResponse({'error': 'No question provided'}, status=400)

    try:
        last_pdf = request.session.get('last_retriever')
        raw_history = request.session.get('chat_history', [])
        chat_history = [tuple(pair) for pair in raw_history]
        active_conv_id = request.session.get('active_conversation_id')

        if last_pdf and os.path.exists(last_pdf):
            retriever = setup_vectorstore(last_pdf)
            chain = get_advanced_chain(retriever)
            response = chain.invoke(
                {'question': question, 'chat_history': list(chat_history)}
            )
            answer = response['answer']
        else:
            answer = answer_without_pdf(question, list(chat_history))

        chat_history = list(chat_history) + [(question, answer)]
        request.session['chat_history'] = [list(pair) for pair in chat_history]

        if not active_conv_id:
            conversation = Conversation.objects.create(
                user=(
                    request.user if request.user.is_authenticated else None
                ),
                title=question[:50],
            )
            active_conv_id = conversation.id
            request.session['active_conversation_id'] = active_conv_id

        try:
            conversation = Conversation.objects.get(id=active_conv_id)
            Message.objects.create(
                conversation=conversation,
                text=question,
                is_ai_response=False,
                user=(
                    request.user if request.user.is_authenticated else None
                ),
            )
            Message.objects.create(
                conversation=conversation,
                text=answer,
                is_ai_response=True,
                user=None,
            )
        except Conversation.DoesNotExist:
            pass

        return JsonResponse({'answer': answer})

    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


@login_required
def delete_file(request, file_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        attachment = Attachment.objects.get(id=file_id, user=request.user)
        if attachment.file and os.path.exists(attachment.file.path):
            os.remove(attachment.file.path)
        attachment.delete()
        return JsonResponse({'success': True})

    except Attachment.DoesNotExist:
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


@login_required
def chats_history(request):
    conversations = Conversation.objects.filter(
        user=request.user
    ).order_by('-created_at')

    conv_data = []
    for conv in conversations:
        first_user_msg = (
            conv.messages.filter(is_ai_response=False)
            .order_by('created_at')
            .first()
        )
        preview = (
            first_user_msg.text[:80]
            if first_user_msg and first_user_msg.text
            else ''
        )
        is_default_title = not conv.title or conv.title == 'New conversation'
        conv_data.append({
            'id': conv.id,
            'title': (
                conv.title if not is_default_title else (preview or 'Untitled')
            ),
            'preview': preview,
            'created_at': conv.created_at,
            'message_count': conv.messages.count(),
        })

    context = {
        'conv_data': conv_data,
        'total_chats': conversations.count(),
    }
    return render(request, 'documents/chats.html', context)


@login_required
def load_file_in_chat(request, file_id):
    try:
        attachment = Attachment.objects.get(id=file_id, user=request.user)
        request.session.pop('active_conversation_id', None)
        request.session.pop('chat_history', None)
        request.session['last_retriever'] = attachment.file.path
        request.session['last_retriever_name'] = _pdf_basename(attachment.file.path)
        return redirect('documents')

    except Attachment.DoesNotExist:
        return redirect('files')
