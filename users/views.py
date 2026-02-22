from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib import messages
from django.shortcuts import render, redirect
from .forms import UserRegistrationForm, LoginForm
from .models import Profile

User = get_user_model()


def signup(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password'],
            )
            profile = Profile.objects.create(user=user)
            profile.save()
            messages.success(
                request,
                'Your account has been created! You are now able to log in',
            )
            return redirect('login')
    else:
        form = UserRegistrationForm()

    return render(request, 'users/signup.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = form.user_cache
            if user is not None:
                auth_login(request, user)
                messages.success(
                    request,
                    'You have been successfully logged in!',
                )
                return redirect('homepage')
    else:
        form = LoginForm()

    return render(request, 'users/login.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        auth_logout(request)
        messages.success(request, 'You have been successfully logged out!')
        return redirect('homepage')

    return render(request, 'users/logout.html')
