from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.urls import reverse
from .forms import FileFieldForm
from .models import Conversation, Message

User = get_user_model()


class ConversationModelTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )

    def test_str_with_user(self):
        conv = Conversation.objects.create(
            user=self.user,
            title='My Chat',
        )
        self.assertEqual(str(conv), 'My Chat - alice')

    def test_str_without_user(self):
        conv = Conversation.objects.create(title='Anon Chat')
        self.assertEqual(str(conv), 'Anon Chat - Anonymous')

    def test_default_title(self):
        conv = Conversation.objects.create(user=self.user)
        self.assertEqual(conv.title, 'New conversation')

    def test_deleted_when_user_deleted(self):
        Conversation.objects.create(user=self.user, title='Chat')
        self.user.delete()
        self.assertEqual(Conversation.objects.count(), 0)


class MessageModelTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.conv = Conversation.objects.create(
            user=self.user,
            title='Test Chat',
        )

    def test_str_shows_user(self):
        msg = Message.objects.create(
            conversation=self.conv,
            user=self.user,
            text='Hello',
        )
        self.assertIn('alice', str(msg))

    def test_str_shows_ai(self):
        msg = Message.objects.create(
            conversation=self.conv,
            text='I am the AI',
            is_ai_response=True,
        )
        self.assertIn('AI', str(msg))

    def test_default_is_not_ai(self):
        msg = Message.objects.create(
            conversation=self.conv,
            text='Hello',
        )
        self.assertFalse(msg.is_ai_response)

    def test_deleted_when_conversation_deleted(self):
        Message.objects.create(conversation=self.conv, text='Hello')
        self.conv.delete()
        self.assertEqual(Message.objects.count(), 0)


class FileFieldFormTest(TestCase):

    def test_text_only_is_valid(self):
        form = FileFieldForm(
            data={'text_prompt': 'Hello'},
            files={},
        )
        self.assertTrue(form.is_valid())

    def test_empty_form_is_invalid(self):
        form = FileFieldForm(
            data={'text_prompt': ''},
            files={},
        )
        self.assertFalse(form.is_valid())

    def test_non_pdf_is_rejected(self):
        bad_file = SimpleUploadedFile(
            'note.txt', b'hello', content_type='text/plain'
        )
        form = FileFieldForm(
            data={'text_prompt': ''},
            files={'file_field': bad_file},
        )
        self.assertFalse(form.is_valid())

    def test_oversized_file_is_rejected(self):
        big_file = SimpleUploadedFile(
            'big.pdf',
            b'%PDF ' + b'x' * (51 * 1024 * 1024),
            content_type='application/pdf',
        )
        form = FileFieldForm(
            data={'text_prompt': ''},
            files={'file_field': big_file},
        )
        self.assertFalse(form.is_valid())


class NewConversationViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.client.login(username='alice', password='password123')

    def test_redirects_to_documents(self):
        response = self.client.get(reverse('new_conversation'))
        self.assertRedirects(response, reverse('documents'))

    def test_clears_session_data(self):
        session = self.client.session
        session['active_conversation_id'] = 99
        session['chat_history'] = [['q', 'a']]
        session.save()
        self.client.get(reverse('new_conversation'))
        self.assertNotIn('active_conversation_id', self.client.session)
        self.assertNotIn('chat_history', self.client.session)


class LoadConversationViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.client.login(username='alice', password='password123')
        self.conv = Conversation.objects.create(
            user=self.user,
            title='My Chat',
        )

    def test_loads_and_redirects(self):
        url = reverse('load_conversation', args=[self.conv.id])
        response = self.client.get(url)
        self.assertRedirects(response, reverse('documents'))

    def test_sets_session(self):
        url = reverse('load_conversation', args=[self.conv.id])
        self.client.get(url)
        self.assertEqual(
            self.client.session['active_conversation_id'],
            self.conv.id,
        )

    def test_other_users_conversation_is_404(self):
        other = User.objects.create_user(
            username='bob',
            email='bob@example.com',
            password='pass123',
        )
        other_conv = Conversation.objects.create(
            user=other, title='Private'
        )
        url = reverse('load_conversation', args=[other_conv.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)


class DeleteConversationViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.client.login(username='alice', password='password123')
        self.conv = Conversation.objects.create(
            user=self.user,
            title='To Delete',
        )

    def test_post_deletes_conversation(self):
        url = reverse('delete_conversation', args=[self.conv.id])
        self.client.post(url)
        self.assertFalse(
            Conversation.objects.filter(id=self.conv.id).exists()
        )

    def test_returns_success_json(self):
        url = reverse('delete_conversation', args=[self.conv.id])
        response = self.client.post(url)
        self.assertJSONEqual(response.content, {'success': True})

    def test_get_request_returns_405(self):
        url = reverse('delete_conversation', args=[self.conv.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 405)

    def test_other_users_conversation_is_404(self):
        other = User.objects.create_user(
            username='bob',
            email='bob@example.com',
            password='pass123',
        )
        other_conv = Conversation.objects.create(
            user=other, title='Private'
        )
        url = reverse('delete_conversation', args=[other_conv.id])
        response = self.client.post(url)
        self.assertEqual(response.status_code, 404)


class FilesPageTest(TestCase):

    def setUp(self):
        self.client = Client()
        User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.client.login(username='alice', password='password123')

    def test_page_loads(self):
        response = self.client.get(reverse('files'))
        self.assertEqual(response.status_code, 200)

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse('files'))
        self.assertEqual(response.status_code, 302)


class ChatsPageTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.client.login(username='alice', password='password123')

    def test_page_loads(self):
        response = self.client.get(reverse('chats'))
        self.assertEqual(response.status_code, 200)

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse('chats'))
        self.assertEqual(response.status_code, 302)

    def test_only_shows_own_chats(self):
        Conversation.objects.create(user=self.user, title='Mine')
        other = User.objects.create_user(
            username='bob',
            email='bob@example.com',
            password='pass123',
        )
        Conversation.objects.create(user=other, title='Not Mine')
        response = self.client.get(reverse('chats'))
        titles = [c['title'] for c in response.context['conv_data']]
        self.assertIn('Mine', titles)
        self.assertNotIn('Not Mine', titles)
