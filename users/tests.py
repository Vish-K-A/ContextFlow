from django.contrib.auth import get_user_model
from django.test import TestCase, Client
from django.urls import reverse
from .forms import UserRegistrationForm, LoginForm
from .models import Profile

User = get_user_model()


class ProfileModelTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )

    def test_str_shows_username(self):
        profile = Profile.objects.create(user=self.user)
        self.assertEqual(str(profile), "alice's Profile")

    def test_profile_belongs_to_user(self):
        profile = Profile.objects.create(user=self.user)
        self.assertEqual(profile.user, self.user)

    def test_profile_gone_when_user_deleted(self):
        Profile.objects.create(user=self.user)
        self.user.delete()
        self.assertEqual(Profile.objects.count(), 0)


class RegistrationFormTest(TestCase):

    def test_valid_data_passes(self):
        form = UserRegistrationForm(data={
            'username': 'bob',
            'email': 'bob@example.com',
            'password': 'pass1234',
            'password2': 'pass1234',
        })
        self.assertTrue(form.is_valid())

    def test_mismatched_passwords_fail(self):
        form = UserRegistrationForm(data={
            'username': 'bob',
            'email': 'bob@example.com',
            'password': 'pass1234',
            'password2': 'different',
        })
        self.assertFalse(form.is_valid())

    def test_duplicate_email_fails(self):
        User.objects.create_user(
            username='existing',
            email='taken@example.com',
            password='pass123',
        )
        form = UserRegistrationForm(data={
            'username': 'bob',
            'email': 'taken@example.com',
            'password': 'pass1234',
            'password2': 'pass1234',
        })
        self.assertFalse(form.is_valid())

    def test_blank_username_fails(self):
        form = UserRegistrationForm(data={
            'username': '',
            'email': 'bob@example.com',
            'password': 'pass1234',
            'password2': 'pass1234',
        })
        self.assertFalse(form.is_valid())


class LoginFormTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )

    def test_correct_credentials_pass(self):
        form = LoginForm(data={
            'email': 'alice@example.com',
            'password': 'password123',
        })
        self.assertTrue(form.is_valid())

    def test_wrong_password_fails(self):
        form = LoginForm(data={
            'email': 'alice@example.com',
            'password': 'wrongpass',
        })
        self.assertFalse(form.is_valid())

    def test_unknown_email_fails(self):
        form = LoginForm(data={
            'email': 'nobody@example.com',
            'password': 'password123',
        })
        self.assertFalse(form.is_valid())

    def test_blank_email_fails(self):
        form = LoginForm(data={
            'email': '',
            'password': 'password123',
        })
        self.assertFalse(form.is_valid())


class SignupViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('signup')

    def test_page_loads(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)

    def test_signup_creates_user(self):
        self.client.post(self.url, {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'pass1234',
            'password2': 'pass1234',
        })
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_signup_creates_profile(self):
        self.client.post(self.url, {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'pass1234',
            'password2': 'pass1234',
        })
        user = User.objects.get(username='newuser')
        self.assertTrue(Profile.objects.filter(user=user).exists())

    def test_signup_redirects_to_login(self):
        response = self.client.post(self.url, {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'pass1234',
            'password2': 'pass1234',
        })
        self.assertRedirects(response, reverse('login'))

    def test_bad_data_does_not_create_user(self):
        self.client.post(self.url, {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'pass1234',
            'password2': 'mismatch',
        })
        self.assertFalse(User.objects.filter(username='newuser').exists())


class LoginViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('login')
        self.user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )

    def test_page_loads(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)

    def test_correct_login_redirects(self):
        response = self.client.post(self.url, {
            'email': 'alice@example.com',
            'password': 'password123',
        })
        self.assertRedirects(
            response,
            reverse('homepage'),
            fetch_redirect_response=False,
        )

    def test_user_is_logged_in_after_login(self):
        self.client.post(self.url, {
            'email': 'alice@example.com',
            'password': 'password123',
        })
        self.assertEqual(
            int(self.client.session['_auth_user_id']),
            self.user.pk,
        )

    def test_wrong_password_stays_on_page(self):
        response = self.client.post(self.url, {
            'email': 'alice@example.com',
            'password': 'wrongpass',
        })
        self.assertEqual(response.status_code, 200)


class LogoutViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('logout')
        User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='password123',
        )
        self.client.login(username='alice', password='password123')

    def test_get_shows_logout_page(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)

    def test_post_logs_out_and_redirects(self):
        response = self.client.post(self.url)
        self.assertRedirects(
            response,
            reverse('homepage'),
            fetch_redirect_response=False,
        )

    def test_session_cleared_after_logout(self):
        self.client.post(self.url)
        self.assertNotIn('_auth_user_id', self.client.session)
