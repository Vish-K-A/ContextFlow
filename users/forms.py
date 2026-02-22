from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate
from django import forms

User = get_user_model()


class UserRegistrationForm(forms.Form):
    username = forms.CharField(max_length=50, required=True)
    email = forms.EmailField(required=True)
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput,
    )

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError(
                'An account with this email already exists.'
            )
        return email

    def clean_password2(self):
        password = self.cleaned_data.get('password')
        password2 = self.cleaned_data.get('password2')
        if password and password2 and password != password2:
            raise forms.ValidationError("Passwords don't match")
        return password2


class LoginForm(forms.Form):
    email = forms.EmailField(
        label='Email Address',
        widget=forms.EmailInput(),
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(),
    )

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')
        password = cleaned_data.get('password')

        if email and password:
            try:
                user_obj = User.objects.get(email=email)
            except User.DoesNotExist as exc:
                raise forms.ValidationError(
                    'Invalid Email or Password. Please Try Again'
                ) from exc

            self.user_cache = authenticate(
                username=user_obj.username,
                password=password,
            )

            if self.user_cache is None:
                raise forms.ValidationError(
                    'Invalid Email or Password. Please Try Again'
                )

        return cleaned_data
